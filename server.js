const express = require('express')
const multer = require('multer')
const cors = require('cors')
const path = require('path')
const fs = require('fs')
const { v4: uuidv4 } = require('uuid')
// const { Jimp } = require('jimp')
const { fft } = require('fft-js')

let sharp = null
let sharpLoadError = null
try {
  sharp = require('sharp')
} catch (error) {
  sharpLoadError = error
  console.warn(
    'sharp failed to load; running in degraded mode without sharp-dependent processing:',
    error.message
  )
}

const app = express()
const PORT = 3000

app.use(cors({ origin: '*' }))

app.use((req, res, next) => {
  res.setHeader('ngrok-skip-browser-warning', 'true')
  next()
})

app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(
  '/textures',
  (req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('ngrok-skip-browser-warning', 'true')
    next()
  },
  express.static(path.join(__dirname, 'public/textures'))
)

const outputDir = path.join(__dirname, 'public/textures')
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true })
}

const upload = multer({ storage: multer.memoryStorage() })

let latestTexturePath = null
let latestGarment = 'sweatshirt'
let latestTextureMeta = null
let textureVersion = 0

const CANDIDATE_SIZE = 256
const FINAL_TILE_SIZE = 1024
const FINAL_TILE_GRID = 6
const MIN_SEAMLESS_SCORE = 0.42
const DEFAULT_DEBUG_MODE = false

const GARMENT_SAMPLING_PROFILES = {
  tshirt: { scales: [0.36, 0.42, 0.48], grid: 7 },
  sweatshirt: { scales: [0.42, 0.5, 0.58], grid: 7 },
  hoodie: { scales: [0.42, 0.5, 0.58], grid: 7 },
  jacket: { scales: [0.46, 0.54, 0.62], grid: 7 },
  coat: { scales: [0.5, 0.58, 0.66], grid: 7 },
  default: { scales: [0.4, 0.48, 0.56], grid: 7 }
}

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function smoothstep(t) {
  const x = clamp(t, 0, 1)
  return x * x * (3 - 2 * x)
}

function srgbToLinear(v) {
  const x = v / 255
  if (x <= 0.04045) return x / 12.92
  return Math.pow((x + 0.055) / 1.055, 2.4)
}

function rgbToLab(r, g, b) {
  const lr = srgbToLinear(r)
  const lg = srgbToLinear(g)
  const lb = srgbToLinear(b)

  const x = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375
  const y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.072175
  const z = lr * 0.0193339 + lg * 0.119192 + lb * 0.9503041

  const xn = 0.95047
  const yn = 1.0
  const zn = 1.08883

  const fx = xyzToLabF(x / xn)
  const fy = xyzToLabF(y / yn)
  const fz = xyzToLabF(z / zn)

  return {
    l: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz)
  }
}

function xyzToLabF(t) {
  const delta = 6 / 29
  const delta3 = delta * delta * delta
  if (t > delta3) return Math.cbrt(t)
  return t / (3 * delta * delta) + 4 / 29
}

function deltaE76(c1, c2) {
  const dl = c1.l - c2.l
  const da = c1.a - c2.a
  const db = c1.b - c2.b
  return Math.sqrt(dl * dl + da * da + db * db)
}

function lumaAt(rawBuffer, idx) {
  return (
    0.2126 * rawBuffer[idx] +
    0.7152 * rawBuffer[idx + 1] +
    0.0722 * rawBuffer[idx + 2]
  )
}

function rgbDistanceSq(r1, g1, b1, r2, g2, b2) {
  const dr = r1 - r2
  const dg = g1 - g2
  const db = b1 - b2
  return dr * dr + dg * dg + db * db
}

function luminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function saturationApprox(r, g, b) {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  return max === 0 ? 0 : (max - min) / max
}

function estimateBorderColor(rawBuffer, width, height, channels) {
  let r = 0
  let g = 0
  let b = 0
  let count = 0

  const border = Math.max(1, Math.floor(Math.min(width, height) * 0.03))

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (
        x >= border &&
        x < width - border &&
        y >= border &&
        y < height - border
      ) {
        continue
      }
      const idx = (y * width + x) * channels
      r += rawBuffer[idx]
      g += rawBuffer[idx + 1]
      b += rawBuffer[idx + 2]
      count++
    }
  }

  if (count === 0) return { r: 255, g: 255, b: 255 }

  return {
    r: Math.round(r / count),
    g: Math.round(g / count),
    b: Math.round(b / count)
  }
}

async function removeBackgroundFromFullImage(imageBuffer) {
  if (!sharp) {
    throw new Error(
      `sharp unavailable for background normalization: ${
        sharpLoadError ? sharpLoadError.message : 'unknown sharp load error'
      }`
    )
  }

  const prepared = await sharp(imageBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  const { data, info } = prepared
  const bg = estimateBorderColor(data, info.width, info.height, info.channels)
  const out = Buffer.from(data)
  const alphaMap = new Uint8Array(info.width * info.height)

  const hardCutSq = 12 * 12
  const softCutSq = 32 * 32
  const brightBackdrop = bg.r > 180 && bg.g > 180 && bg.b > 180

  for (let i = 0, p = 0; i < out.length; i += info.channels, p++) {
    const r = out[i]
    const g = out[i + 1]
    const b = out[i + 2]

    let distSq = rgbDistanceSq(r, g, b, bg.r, bg.g, bg.b)

    if (brightBackdrop) {
      const max = Math.max(r, g, b)
      const min = Math.min(r, g, b)
      const sat = max === 0 ? 0 : (max - min) / max
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

      // Be more aggressive for bright low-saturation studio backgrounds.
      if (lum > 185 && sat < 0.12) {
        distSq *= 0.6
      }
    }

    let alpha = 255
    if (distSq <= hardCutSq) {
      alpha = 0
    } else if (distSq < softCutSq) {
      const t = (distSq - hardCutSq) / (softCutSq - hardCutSq)
      alpha = Math.round(t * 255)
    }

    // Undo white-matte edge contamination so transparent fringes don't look like a white shadow.
    if (alpha > 0 && alpha < 255) {
      const a = alpha / 255
      const invA = 1 - a
      out[i] = clamp(Math.round((r - bg.r * invA) / a), 0, 255)
      out[i + 1] = clamp(Math.round((g - bg.g * invA) / a), 0, 255)
      out[i + 2] = clamp(Math.round((b - bg.b * invA) / a), 0, 255)
    }

    out[i + 3] = alpha
    alphaMap[p] = alpha
  }

  // Remove bright desaturated fringe that often survives around product cutouts.
  for (let y = 1; y < info.height - 1; y++) {
    for (let x = 1; x < info.width - 1; x++) {
      const p = y * info.width + x
      const alpha = alphaMap[p]
      if (alpha === 0) continue

      let touchesTransparent = false
      for (let oy = -1; oy <= 1 && !touchesTransparent; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue
          if (alphaMap[(y + oy) * info.width + (x + ox)] < 16) {
            touchesTransparent = true
            break
          }
        }
      }
      if (!touchesTransparent) continue

      const idx = p * info.channels
      const r = out[idx]
      const g = out[idx + 1]
      const b = out[idx + 2]
      const lum = luminance(r, g, b)
      const sat = saturationApprox(r, g, b)

      // Target near-white/gray edge spill only.
      if (lum > 170 && sat < 0.2) {
        const lumT = clamp((lum - 170) / 85, 0, 1)
        const satT = 1 - clamp(sat / 0.2, 0, 1)
        const suppress = lumT * satT
        const nextAlpha = Math.round(alpha * (1 - 0.98 * suppress))
        out[idx + 3] = nextAlpha < 12 ? 0 : nextAlpha
      }
    }
  }

  return sharp(out, {
    raw: {
      width: info.width,
      height: info.height,
      channels: info.channels
    }
  })
    .png()
    .toBuffer()
}

function edgeSignal(rawBuffer, width, height, channels, side) {
  const signal = []

  if (side === 'left') {
    for (let y = 0; y < height; y++) {
      signal.push(lumaAt(rawBuffer, y * width * channels))
    }
    return signal
  }

  if (side === 'right') {
    for (let y = 0; y < height; y++) {
      signal.push(lumaAt(rawBuffer, (y * width + (width - 1)) * channels))
    }
    return signal
  }

  if (side === 'top') {
    for (let x = 0; x < width; x++) {
      signal.push(lumaAt(rawBuffer, x * channels))
    }
    return signal
  }

  for (let x = 0; x < width; x++) {
    signal.push(lumaAt(rawBuffer, ((height - 1) * width + x) * channels))
  }
  return signal
}

function nextPowerOfTwo(n) {
  let p = 1
  while (p < n) p <<= 1
  return p
}

function normalizeSignalForFft(signal) {
  if (!Array.isArray(signal) || signal.length === 0) return []

  const size = nextPowerOfTwo(signal.length)
  if (size === signal.length) return signal

  const out = signal.slice()
  while (out.length < size) out.push(0)
  return out
}

function periodicityIndexFromSignal(signal) {
  if (signal.length < 8) return 0

  const normalized = normalizeSignalForFft(signal)
  if (normalized.length < 8) return 0

  const spectrum = fft(normalized)
  const half = Math.floor(spectrum.length / 2)
  if (half <= 1) return 0

  let maxMag = 0
  let sumMag = 0
  let count = 0

  for (let i = 1; i < half; i++) {
    const re = spectrum[i][0]
    const im = spectrum[i][1]
    const mag = Math.sqrt(re * re + im * im)
    maxMag = Math.max(maxMag, mag)
    sumMag += mag
    count++
  }

  if (count === 0) return 0
  const avgMag = sumMag / count
  if (avgMag <= 1e-6) return 0

  return clamp((maxMag / avgMag - 1) / 6, 0, 1)
}

function phaseCoherence(a, b) {
  if (a.length !== b.length || a.length < 8) return 0

  const na = normalizeSignalForFft(a)
  const nb = normalizeSignalForFft(b)
  if (na.length !== nb.length || na.length < 8) return 0

  const fa = fft(na)
  const fb = fft(nb)
  const half = Math.floor(fa.length / 2)

  let phaseDiffWeighted = 0
  let magnitudeMismatchWeighted = 0
  let weightSum = 0

  for (let i = 1; i < half; i++) {
    const ar = fa[i][0]
    const ai = fa[i][1]
    const br = fb[i][0]
    const bi = fb[i][1]

    const ma = Math.sqrt(ar * ar + ai * ai)
    const mb = Math.sqrt(br * br + bi * bi)
    const weight = (ma + mb) * 0.5
    if (weight < 1e-4) continue

    const pa = Math.atan2(ai, ar)
    const pb = Math.atan2(bi, br)
    const d = Math.atan2(Math.sin(pa - pb), Math.cos(pa - pb))

    phaseDiffWeighted += (Math.abs(d) / Math.PI) * weight
    magnitudeMismatchWeighted += (Math.abs(ma - mb) / (ma + mb + 1e-6)) * weight
    weightSum += weight
  }

  if (weightSum <= 1e-6) return 0

  const phaseMatch = 1 - clamp(phaseDiffWeighted / weightSum, 0, 1)
  const magnitudeMatch = 1 - clamp(magnitudeMismatchWeighted / weightSum, 0, 1)

  return clamp(phaseMatch * 0.7 + magnitudeMatch * 0.3, 0, 1)
}

function scoreCandidate(rawBuffer, width, height, channels, options = {}) {
  let lumSum = 0
  let lumSqSum = 0
  let backgroundLikeCount = 0
  const pixelCount = width * height
  const bgColor =
    options.bgColor || estimateBorderColor(rawBuffer, width, height, channels)
  const bgThresholdSq = 20 * 20

  for (let i = 0; i < rawBuffer.length; i += channels) {
    const r = rawBuffer[i]
    const g = rawBuffer[i + 1]
    const b = rawBuffer[i + 2]
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lumSum += lum
    lumSqSum += lum * lum

    if (
      rgbDistanceSq(r, g, b, bgColor.r, bgColor.g, bgColor.b) < bgThresholdSq
    ) {
      backgroundLikeCount++
    }
  }

  const lumMean = lumSum / pixelCount
  const lumVariance = Math.max(0, lumSqSum / pixelCount - lumMean * lumMean)
  const detailNorm = clamp(Math.sqrt(lumVariance) / 64, 0, 1)

  let edgeDeltaESum = 0
  let edgeDeltaECount = 0

  for (let y = 0; y < height; y++) {
    const leftIdx = y * width * channels
    const rightIdx = (y * width + (width - 1)) * channels
    const leftLab = rgbToLab(
      rawBuffer[leftIdx],
      rawBuffer[leftIdx + 1],
      rawBuffer[leftIdx + 2]
    )
    const rightLab = rgbToLab(
      rawBuffer[rightIdx],
      rawBuffer[rightIdx + 1],
      rawBuffer[rightIdx + 2]
    )
    edgeDeltaESum += deltaE76(leftLab, rightLab)
    edgeDeltaECount++
  }

  for (let x = 0; x < width; x++) {
    const topIdx = x * channels
    const bottomIdx = ((height - 1) * width + x) * channels
    const topLab = rgbToLab(
      rawBuffer[topIdx],
      rawBuffer[topIdx + 1],
      rawBuffer[topIdx + 2]
    )
    const bottomLab = rgbToLab(
      rawBuffer[bottomIdx],
      rawBuffer[bottomIdx + 1],
      rawBuffer[bottomIdx + 2]
    )
    edgeDeltaESum += deltaE76(topLab, bottomLab)
    edgeDeltaECount++
  }

  const avgEdgeDeltaE =
    edgeDeltaECount > 0 ? edgeDeltaESum / edgeDeltaECount : 100
  const edgeMatchNorm = 1 - clamp(avgEdgeDeltaE / 50, 0, 1)

  const leftSignal = edgeSignal(rawBuffer, width, height, channels, 'left')
  const rightSignal = edgeSignal(rawBuffer, width, height, channels, 'right')
  const topSignal = edgeSignal(rawBuffer, width, height, channels, 'top')
  const bottomSignal = edgeSignal(rawBuffer, width, height, channels, 'bottom')

  const phaseLeftRight = phaseCoherence(leftSignal, rightSignal)
  const phaseTopBottom = phaseCoherence(topSignal, bottomSignal)
  const phaseNorm = (phaseLeftRight + phaseTopBottom) * 0.5

  const rowSignal = []
  for (let y = 0; y < height; y++) {
    let rowSum = 0
    for (let x = 0; x < width; x++) {
      rowSum += lumaAt(rawBuffer, (y * width + x) * channels)
    }
    rowSignal.push(rowSum / width)
  }

  const colSignal = []
  for (let x = 0; x < width; x++) {
    let colSum = 0
    for (let y = 0; y < height; y++) {
      colSum += lumaAt(rawBuffer, (y * width + x) * channels)
    }
    colSignal.push(colSum / height)
  }

  const periodicPenalty = Math.max(
    periodicityIndexFromSignal(rowSignal),
    periodicityIndexFromSignal(colSignal)
  )

  const measuredBgRatio = backgroundLikeCount / pixelCount
  const hintBgRatio =
    typeof options.foregroundCoverage === 'number'
      ? clamp(1 - options.foregroundCoverage, 0, 1)
      : measuredBgRatio
  const backgroundRatio = Math.max(measuredBgRatio, hintBgRatio)
  const backgroundPenalty = clamp((backgroundRatio - 0.25) / 0.75, 0, 1)

  // Favor perceptual seam quality and frequency-phase alignment for complex prints.
  const score =
    edgeMatchNorm * 0.36 +
    phaseNorm * 0.36 +
    detailNorm * 0.28 -
    periodicPenalty * 0.1 -
    backgroundPenalty * 0.35

  return clamp(score, 0, 1)
}

function refineTileEdges(rawBuffer, width, height, channels) {
  const out = Buffer.from(rawBuffer)
  const blendX = Math.max(1, Math.floor(width * 0.08))
  const blendY = Math.max(1, Math.floor(height * 0.08))

  for (let x = 0; x < blendX; x++) {
    const t = blendX === 1 ? 1 : smoothstep(x / (blendX - 1))
    for (let y = 0; y < height; y++) {
      const leftIdx = (y * width + x) * channels
      const rightIdx = (y * width + (width - 1 - x)) * channels
      for (let c = 0; c < 3; c++) {
        const left = out[leftIdx + c]
        const right = out[rightIdx + c]
        const avg = (left + right) * 0.5
        out[leftIdx + c] = Math.round(left * (1 - t) + avg * t)
        out[rightIdx + c] = Math.round(right * (1 - t) + avg * t)
      }
    }
  }

  for (let y = 0; y < blendY; y++) {
    const t = blendY === 1 ? 1 : smoothstep(y / (blendY - 1))
    for (let x = 0; x < width; x++) {
      const topIdx = (y * width + x) * channels
      const bottomIdx = ((height - 1 - y) * width + x) * channels
      for (let c = 0; c < 3; c++) {
        const top = out[topIdx + c]
        const bottom = out[bottomIdx + c]
        const avg = (top + bottom) * 0.5
        out[topIdx + c] = Math.round(top * (1 - t) + avg * t)
        out[bottomIdx + c] = Math.round(bottom * (1 - t) + avg * t)
      }
    }
  }

  return out
}

function detectOpaqueBounds(rawBuffer, width, height, channels) {
  if (channels < 4) {
    return {
      minX: 0,
      minY: 0,
      maxX: width - 1,
      maxY: height - 1,
      opaqueRatio: 1
    }
  }

  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  let opaqueCount = 0

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * channels
      if (rawBuffer[idx + 3] < 24) continue

      opaqueCount++
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }

  if (opaqueCount === 0 || maxX < minX || maxY < minY) {
    return {
      minX: 0,
      minY: 0,
      maxX: width - 1,
      maxY: height - 1,
      opaqueRatio: 0
    }
  }

  return {
    minX,
    minY,
    maxX,
    maxY,
    opaqueRatio: opaqueCount / (width * height)
  }
}

function summarizePatch(rawBuffer, width, height, channels) {
  let opaque = 0
  let sumR = 0
  let sumG = 0
  let sumB = 0
  let sumL = 0
  let sumL2 = 0
  let sumC = 0
  let sumC2 = 0

  const rowSignal = new Array(height).fill(0)
  const colSignal = new Array(width).fill(0)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * channels
      const a = channels > 3 ? rawBuffer[idx + 3] : 255
      if (a < 24) continue

      const r = rawBuffer[idx]
      const g = rawBuffer[idx + 1]
      const b = rawBuffer[idx + 2]
      const l = luminance(r, g, b)
      const c = Math.max(r, g, b) - Math.min(r, g, b)

      opaque++
      sumR += r
      sumG += g
      sumB += b
      sumL += l
      sumL2 += l * l
      sumC += c
      sumC2 += c * c

      rowSignal[y] += l
      colSignal[x] += l
    }
  }

  if (opaque === 0) {
    return {
      coverage: 0,
      meanColor: { r: 128, g: 128, b: 128 },
      lStd: 100,
      chromaStd: 100,
      rowPeriodicity: 0,
      colPeriodicity: 0
    }
  }

  for (let y = 0; y < height; y++) rowSignal[y] /= width
  for (let x = 0; x < width; x++) colSignal[x] /= height

  const meanL = sumL / opaque
  const meanC = sumC / opaque
  const lVar = Math.max(0, sumL2 / opaque - meanL * meanL)
  const cVar = Math.max(0, sumC2 / opaque - meanC * meanC)

  return {
    coverage: opaque / (width * height),
    meanColor: {
      r: Math.round(sumR / opaque),
      g: Math.round(sumG / opaque),
      b: Math.round(sumB / opaque)
    },
    lStd: Math.sqrt(lVar),
    chromaStd: Math.sqrt(cVar),
    rowPeriodicity: periodicityIndexFromSignal(rowSignal),
    colPeriodicity: periodicityIndexFromSignal(colSignal)
  }
}

function normalizePatchLighting(rawBuffer, width, height, channels, meanColor) {
  const out = Buffer.from(rawBuffer)

  let lumSum = 0
  let lumCount = 0
  let lumMin = 255
  let lumMax = 0

  for (let i = 0; i < out.length; i += channels) {
    const a = channels > 3 ? out[i + 3] : 255
    if (a < 24) continue
    const lum = luminance(out[i], out[i + 1], out[i + 2])
    lumSum += lum
    lumCount++
    if (lum < lumMin) lumMin = lum
    if (lum > lumMax) lumMax = lum
  }

  const meanLum = lumCount > 0 ? lumSum / lumCount : 128
  const lumRange = Math.max(16, lumMax - lumMin)

  for (let i = 0; i < out.length; i += channels) {
    const a = channels > 3 ? out[i + 3] : 255

    if (a < 24) {
      out[i] = meanColor.r
      out[i + 1] = meanColor.g
      out[i + 2] = meanColor.b
      if (channels > 3) out[i + 3] = 255
      continue
    }

    const r = out[i]
    const g = out[i + 1]
    const b = out[i + 2]
    const lum = luminance(r, g, b)
    const sat = saturationApprox(r, g, b)

    let targetLum = meanLum + (lum - meanLum) * 0.74

    // Gently compress very bright low-saturation pixels to avoid residual studio glare.
    if (lum > meanLum + lumRange * 0.45 && sat < 0.2) {
      targetLum -= (lum - meanLum) * 0.1
    }

    targetLum = clamp(targetLum, 14, 242)
    const scale = targetLum / Math.max(1, lum)

    out[i] = clamp(Math.round(r * scale), 0, 255)
    out[i + 1] = clamp(Math.round(g * scale), 0, 255)
    out[i + 2] = clamp(Math.round(b * scale), 0, 255)
    if (channels > 3) out[i + 3] = 255
  }

  return out
}

function detectPatternType(stats) {
  const dominantPeriodicity = Math.max(
    stats.rowPeriodicity,
    stats.colPeriodicity
  )
  const periodicityDelta = Math.abs(stats.rowPeriodicity - stats.colPeriodicity)

  if (stats.lStd < 6 && stats.chromaStd < 5) return 'solid'
  if (dominantPeriodicity > 0.18 && periodicityDelta > 0.12) return 'stripe'
  if (dominantPeriodicity > 0.16) return 'weave'
  return 'print'
}

function toBooleanFlag(value, fallback = false) {
  if (value === undefined || value === null || value === '') return fallback
  if (typeof value === 'boolean') return value
  const text = String(value).toLowerCase().trim()
  return text === '1' || text === 'true' || text === 'yes' || text === 'on'
}

function parseMinScore(value, fallback) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return clamp(parsed, 0, 1)
}

function getSamplingProfile(garmentType) {
  const key = String(garmentType || '').toLowerCase()
  return GARMENT_SAMPLING_PROFILES[key] || GARMENT_SAMPLING_PROFILES.default
}

function quantizeColor(color, bucketSize = 18) {
  return {
    r: clamp(Math.round(color.r / bucketSize) * bucketSize, 0, 255),
    g: clamp(Math.round(color.g / bucketSize) * bucketSize, 0, 255),
    b: clamp(Math.round(color.b / bucketSize) * bucketSize, 0, 255)
  }
}

function colorDistance(c1, c2) {
  const dr = c1.r - c2.r
  const dg = c1.g - c2.g
  const db = c1.b - c2.b
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

function retainDiverseCandidates(candidates, limit = 24) {
  if (candidates.length <= limit) return candidates

  const sorted = candidates.slice().sort((a, b) => b.score - a.score)
  const selected = []
  const minColorDistance = 28

  for (const candidate of sorted) {
    if (selected.length >= limit) break

    const tooSimilar = selected.some(
      (item) =>
        item.patternType === candidate.patternType &&
        colorDistance(item.signatureColor, candidate.signatureColor) <
          minColorDistance
    )

    if (!tooSimilar || selected.length < Math.min(6, limit)) {
      selected.push(candidate)
    }
  }

  while (selected.length < limit && selected.length < sorted.length) {
    selected.push(sorted[selected.length])
  }

  return selected
}

async function extractFabricPatchCandidates(cleanBuffer, garmentType) {
  const prepared = await sharp(cleanBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  const { data, info } = prepared
  const bounds = detectOpaqueBounds(
    data,
    info.width,
    info.height,
    info.channels
  )

  const roiWidth = Math.max(1, bounds.maxX - bounds.minX + 1)
  const roiHeight = Math.max(1, bounds.maxY - bounds.minY + 1)
  const base = Math.min(roiWidth, roiHeight)

  const profile = getSamplingProfile(garmentType)
  const grid = profile.grid
  const centerWeightBand = 0.24
  const candidates = []
  let candidateIndex = 0

  for (const scale of profile.scales) {
    const patchSize = clamp(Math.floor(base * scale), 96, base)
    const maxLeft = Math.max(bounds.minX, bounds.maxX - patchSize + 1)
    const maxTop = Math.max(bounds.minY, bounds.maxY - patchSize + 1)
    const centerX = bounds.minX + (maxLeft - bounds.minX) * 0.5
    const centerY = bounds.minY + (maxTop - bounds.minY) * 0.5

    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const left =
          grid === 1
            ? bounds.minX
            : Math.round(
                bounds.minX + (gx / (grid - 1)) * (maxLeft - bounds.minX)
              )
        const top =
          grid === 1
            ? bounds.minY
            : Math.round(
                bounds.minY + (gy / (grid - 1)) * (maxTop - bounds.minY)
              )

        const sample = await sharp(cleanBuffer)
          .extract({ left, top, width: patchSize, height: patchSize })
          .resize(CANDIDATE_SIZE, CANDIDATE_SIZE, {
            fit: 'fill',
            kernel: sharp.kernel.lanczos3
          })
          .ensureAlpha()
          .raw()
          .toBuffer({ resolveWithObject: true })

        const initialStats = summarizePatch(
          sample.data,
          sample.info.width,
          sample.info.height,
          sample.info.channels
        )

        if (initialStats.coverage < 0.72) continue

        const normalized = normalizePatchLighting(
          sample.data,
          sample.info.width,
          sample.info.height,
          sample.info.channels,
          initialStats.meanColor
        )

        const normalizedStats = summarizePatch(
          normalized,
          sample.info.width,
          sample.info.height,
          sample.info.channels
        )

        const patternType = detectPatternType(normalizedStats)
        const baseScore = scoreCandidate(
          normalized,
          sample.info.width,
          sample.info.height,
          sample.info.channels,
          { bgColor: normalizedStats.meanColor, foregroundCoverage: 1 }
        )

        const patternBonus =
          patternType === 'solid'
            ? 0.03
            : patternType === 'weave'
              ? 0.02
              : patternType === 'stripe'
                ? 0.015
                : 0

        const centerDx = maxLeft === bounds.minX ? 0 : Math.abs(left - centerX)
        const centerDy = maxTop === bounds.minY ? 0 : Math.abs(top - centerY)
        const normalizedCenterDx =
          maxLeft === bounds.minX
            ? 0
            : centerDx / Math.max(1, maxLeft - bounds.minX)
        const normalizedCenterDy =
          maxTop === bounds.minY
            ? 0
            : centerDy / Math.max(1, maxTop - bounds.minY)
        const centerDistance = Math.sqrt(
          normalizedCenterDx * normalizedCenterDx +
            normalizedCenterDy * normalizedCenterDy
        )
        const centerBonus =
          clamp(1 - centerDistance * 1.2, 0, 1) * centerWeightBand
        const score = clamp(baseScore + patternBonus + centerBonus, 0, 1)

        candidates.push({
          index: candidateIndex++,
          buffer: normalized,
          width: sample.info.width,
          height: sample.info.height,
          channels: sample.info.channels,
          left,
          top,
          patchSize,
          coverage: Number(normalizedStats.coverage.toFixed(4)),
          score: Number(score.toFixed(4)),
          scoreBreakdown: {
            base: Number(baseScore.toFixed(4)),
            patternBonus: Number(patternBonus.toFixed(4)),
            centerBonus: Number(centerBonus.toFixed(4))
          },
          patternType,
          signatureColor: quantizeColor(normalizedStats.meanColor)
        })
      }
    }
  }

  return retainDiverseCandidates(candidates, 24)
}

async function renderFinalTile(candidate) {
  const refined = refineTileEdges(
    candidate.buffer,
    candidate.width,
    candidate.height,
    candidate.channels
  )

  const seamlessScore = scoreCandidate(
    refined,
    candidate.width,
    candidate.height,
    candidate.channels,
    { foregroundCoverage: 1 }
  )

  const perTileSize = Math.max(1, Math.floor(FINAL_TILE_SIZE / FINAL_TILE_GRID))

  const singleTileBuffer = await sharp(refined, {
    raw: {
      width: candidate.width,
      height: candidate.height,
      channels: candidate.channels
    }
  })
    .resize(perTileSize, perTileSize, {
      fit: 'fill',
      kernel: sharp.kernel.lanczos3
    })
    .png()
    .toBuffer()

  const composites = []
  for (let row = 0; row < FINAL_TILE_GRID; row++) {
    for (let col = 0; col < FINAL_TILE_GRID; col++) {
      composites.push({
        input: singleTileBuffer,
        left: col * perTileSize,
        top: row * perTileSize
      })
    }
  }

  let buffer = await sharp({
    create: {
      width: perTileSize * FINAL_TILE_GRID,
      height: perTileSize * FINAL_TILE_GRID,
      channels: 4,
      background: { r: 255, g: 255, b: 255, alpha: 1 }
    }
  })
    .composite(composites)
    .png()
    .toBuffer()

  // Keep output dimensions stable for consumers if integer division changes edge size.
  if (perTileSize * FINAL_TILE_GRID !== FINAL_TILE_SIZE) {
    buffer = await sharp(buffer)
      .resize(FINAL_TILE_SIZE, FINAL_TILE_SIZE, {
        fit: 'fill',
        kernel: sharp.kernel.lanczos3
      })
      .png()
      .toBuffer()
  }

  return {
    buffer,
    patternType: candidate.patternType,
    seamlessScore
  }
}

async function extractTextureWithComfyUI(
  imageBuffer,
  garmentType,
  options = {}
) {
  if (!sharp) {
    throw new Error(
      `sharp unavailable for background removal: ${
        sharpLoadError ? sharpLoadError.message : 'unknown sharp load error'
      }`
    )
  }

  const telemetry = {
    timingsMs: {},
    candidateCount: 0
  }

  const bgStart = Date.now()
  const cleanBuffer = await removeBackgroundFromFullImage(imageBuffer)
  telemetry.timingsMs.backgroundRemoval = Date.now() - bgStart

  const candidateStart = Date.now()
  const candidates = await extractFabricPatchCandidates(
    cleanBuffer,
    garmentType
  )
  telemetry.timingsMs.candidateExtraction = Date.now() - candidateStart
  telemetry.candidateCount = candidates.length

  if (candidates.length === 0) {
    return {
      buffer: cleanBuffer,
      metadata: {
        pipeline: 'background-only',
        patternType: 'unknown',
        seamlessScore: 0,
        telemetry: options.debug ? telemetry : undefined
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score)
  const minSeamlessScore = parseMinScore(
    options.minSeamlessScore,
    MIN_SEAMLESS_SCORE
  )
  const selected =
    candidates.find((candidate) => candidate.score >= minSeamlessScore) ||
    candidates[0]

  const renderStart = Date.now()
  const finalTile = await renderFinalTile(selected)
  telemetry.timingsMs.finalRender = Date.now() - renderStart
  telemetry.timingsMs.total =
    telemetry.timingsMs.backgroundRemoval +
    telemetry.timingsMs.candidateExtraction +
    telemetry.timingsMs.finalRender

  const topCandidates = candidates.slice(0, 6).map((candidate) => ({
    index: candidate.index,
    left: candidate.left,
    top: candidate.top,
    patchSize: candidate.patchSize,
    coverage: candidate.coverage,
    patternType: candidate.patternType,
    score: candidate.score,
    scoreBreakdown: candidate.scoreBreakdown
  }))

  return {
    buffer: finalTile.buffer,
    metadata: {
      pipeline: 'patch-tile',
      patternType: finalTile.patternType,
      seamlessScore: Number(finalTile.seamlessScore.toFixed(3)),
      tileSize: FINAL_TILE_SIZE,
      tileGrid: FINAL_TILE_GRID,
      minSeamlessScore,
      selectedCandidate: {
        index: selected.index,
        left: selected.left,
        top: selected.top,
        patchSize: selected.patchSize,
        coverage: selected.coverage,
        score: selected.score,
        scoreBreakdown: selected.scoreBreakdown,
        patternType: selected.patternType
      },
      telemetry: options.debug
        ? {
            ...telemetry,
            topCandidates
          }
        : undefined
    }
  }
}
// -------------------------------------------------------
// Routes
// -------------------------------------------------------

app.post('/process-garment', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' })

    if (req.body.garment) {
      latestGarment = req.body.garment

      console.log('Garment type:', latestGarment)
    }

    const debug = toBooleanFlag(req.body.debug, DEFAULT_DEBUG_MODE)
    const minSeamlessScore = parseMinScore(
      req.body.minSeamlessScore,
      MIN_SEAMLESS_SCORE
    )

    console.log('Extracting texture...')
    const result = await extractTextureWithComfyUI(
      req.file.buffer,
      latestGarment,
      { debug, minSeamlessScore }
    )
    const textureBuffer = result.buffer

    const filename = `${uuidv4()}.png`
    const filepath = path.join(outputDir, filename)
    fs.writeFileSync(filepath, textureBuffer)
    latestTexturePath = filepath
    latestTextureMeta = result.metadata
    textureVersion++

    console.log('Saved:', filename)

    const BASE_URL =
      'https://nickolas-aciniform-misunderstandingly.ngrok-free.dev'

    const textureUrl = `${BASE_URL}/textures/${filename}`

    res.json({
      success: true,
      texture: textureUrl,
      patternType: result.metadata.patternType,
      seamlessScore: result.metadata.seamlessScore,
      pipeline: result.metadata.pipeline,
      minSeamlessScore: result.metadata.minSeamlessScore,
      selectedCandidate: result.metadata.selectedCandidate,
      telemetry: debug ? result.metadata.telemetry : undefined
    })
  } catch (error) {
    console.error('Failed:', error)
    res.status(500).json({ error: error.message })
  }
})

app.get('/latest-texture', (req, res) => {
  if (!latestTexturePath || !fs.existsSync(latestTexturePath)) {
    return res.status(404).send('No texture yet')
  }
  res.sendFile(latestTexturePath)
})

app.get('/latest-config', (req, res) => {
  res.json({
    garment: latestGarment,
    textureVersion,
    textureMeta: latestTextureMeta
  })
})

app.post('/latest-config', (req, res) => {
  if (req.body.garment) {
    latestGarment = req.body.garment
    console.log('Garment updated to:', latestGarment)
  }
  res.json({ success: true, garment: latestGarment })
})

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`)
})
