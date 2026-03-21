const express = require('express')
const multer = require('multer')
const cors = require('cors')
const path = require('path')
const fs = require('fs')
const { v4: uuidv4 } = require('uuid')
const { removeBackground } = require('@imgly/background-removal-node')
const { Jimp } = require('jimp')
const { Vibrant } = require('node-vibrant/node')

const app = express()
const PORT = 3000

app.use(cors())
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use('/textures', express.static(path.join(__dirname, 'public/textures')))

const outputDir = path.join(__dirname, 'public/textures')
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true })
}

const upload = multer({ storage: multer.memoryStorage() })

let latestTexturePath = null
let latestGarment = 'sweatshirt'
let textureVersion = 0 // ← add this

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------

function getAverageColor(img) {
  const data = img.bitmap.data
  let r = 0,
    g = 0,
    b = 0,
    count = 0
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 128) {
      r += data[i]
      g += data[i + 1]
      b += data[i + 2]
      count++
    }
  }
  if (count === 0) return { r: 128, g: 128, b: 128 }
  return {
    r: Math.round(r / count),
    g: Math.round(g / count),
    b: Math.round(b / count)
  }
}

function findGarmentBounds(img) {
  const data = img.bitmap.data
  const w = img.bitmap.width
  const h = img.bitmap.height
  let minX = w,
    maxX = 0,
    minY = h,
    maxY = 0

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4
      if (data[idx + 3] > 10) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }
  return { minX, maxX, minY, maxY }
}

function flattenAlpha(img, avgColor) {
  const data = img.bitmap.data
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) {
      data[i] = avgColor.r
      data[i + 1] = avgColor.g
      data[i + 2] = avgColor.b
      data[i + 3] = 255
    } else {
      data[i + 3] = 255
    }
  }
  return img
}

function makeSeamless(img) {
  const w = img.bitmap.width
  const h = img.bitmap.height
  const data = img.bitmap.data
  const blendW = Math.floor(w * 0.15)
  const blendH = Math.floor(h * 0.15)

  // Blend left/right edges
  for (let x = 0; x < blendW; x++) {
    const t = x / blendW
    for (let y = 0; y < h; y++) {
      const idxL = (y * w + x) * 4
      const idxR = (y * w + (w - 1 - x)) * 4
      for (let c = 0; c < 3; c++) {
        data[idxL + c] = Math.round(
          data[idxL + c] * (1 - t) + data[idxR + c] * t
        )
      }
    }
  }

  // Blend top/bottom edges
  for (let y = 0; y < blendH; y++) {
    const t = y / blendH
    for (let x = 0; x < w; x++) {
      const idxT = (y * w + x) * 4
      const idxB = ((h - 1 - y) * w + x) * 4
      for (let c = 0; c < 3; c++) {
        data[idxT + c] = Math.round(
          data[idxT + c] * (1 - t) + data[idxB + c] * t
        )
      }
    }
  }

  return img
}

async function extractTexture(imageBuffer) {
  const img = await Jimp.read(imageBuffer)
  const w = img.bitmap.width
  const h = img.bitmap.height

  // Crop center patch
  const patchSize = Math.floor(Math.min(w, h) * 0.25)
  const patchX = Math.floor(w / 2) - Math.floor(patchSize / 2)
  const patchY = Math.floor(h * 0.35) - Math.floor(patchSize / 2)

  console.log(`Patch: ${patchX},${patchY} size ${patchSize}x${patchSize}`)

  const patch = img.clone().crop({
    x: patchX,
    y: patchY,
    w: patchSize,
    h: patchSize
  })

  // Resize patch to 512x512
  patch.resize({ w: 512, h: 512 })

  // Tile 2x2 into 1024x1024
  const final = new Jimp({ width: 1024, height: 1024, color: 0xffffffff })
  final.composite(patch, 0, 0)
  final.composite(patch, 512, 0)
  final.composite(patch, 0, 512)
  final.composite(patch, 512, 512)

  return await final.getBuffer('image/png')
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

    // console.log('Removing background...')
    // const blob = new Blob([req.file.buffer], { type: req.file.mimetype })
    // const resultBlob = await removeBackground(blob)
    // const cleanBuffer = Buffer.from(await resultBlob.arrayBuffer())

    console.log('Extracting texture...')
    // const textureBuffer = await extractTexture(cleanBuffer)
    const textureBuffer = await extractTexture(req.file.buffer)

    const filename = `${uuidv4()}.png`
    const filepath = path.join(outputDir, filename)
    fs.writeFileSync(filepath, textureBuffer)
    latestTexturePath = filepath
    textureVersion++ // ← increment on every new upload

    console.log('Saved:', filename)

    const textureUrl = `http://localhost:${PORT}/textures/${filename}`
    res.json({ success: true, texture: textureUrl })
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
    textureVersion // ← include in config response
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
