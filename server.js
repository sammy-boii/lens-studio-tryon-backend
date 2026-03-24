const express = require('express')
const multer = require('multer')
const cors = require('cors')
const path = require('path')
const fs = require('fs')
const { v4: uuidv4 } = require('uuid')
// const { Jimp } = require('jimp')

const sharp = require('sharp')

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

async function cropCenter(imageBuffer) {
  const image = sharp(imageBuffer)
  const metadata = await image.metadata()

  // Smaller ratio means a tighter center crop (more zoom-in).
  const CENTER_CROP_RATIO = 0.25
  const baseSize = Math.min(metadata.width, metadata.height)
  const size = Math.max(1, Math.floor(baseSize * CENTER_CROP_RATIO))

  const croppedBuffer = await image
    .extract({
      left: Math.floor((metadata.width - size) / 2),
      top: Math.floor((metadata.height - size) / 2),
      width: size,
      height: size
    })
    .resize(512, 512)
    .png()
    .toBuffer()

  // 🔥 DEBUG: save cropped image
  const filepath = path.join(outputDir, 'cropped.png')
  fs.writeFileSync(filepath, croppedBuffer)

  return croppedBuffer
}

async function extractTextureWithComfyUI(imageBuffer) {
  // 🔥 STEP 0: Crop center (VERY IMPORTANT)
  const cropped = await cropCenter(imageBuffer)
  console.log('CROPPED')

  // Convert to base64
  const base64 = cropped.toString('base64')

  // Upload image
  const formData = new FormData()
  const blob = new Blob([cropped], { type: 'image/png' })
  formData.append('image', blob, 'input.png')

  const uploadRes = await fetch('http://127.0.0.1:8188/upload/image', {
    method: 'POST',
    body: formData
  })

  const uploadData = await uploadRes.json()
  const imageName = uploadData.name

  // 🔥 WORKFLOW
  const workflow = {
    1: {
      class_type: 'CheckpointLoaderSimple',
      inputs: {
        ckpt_name: 'v1-5-pruned-emaonly.safetensors'
      }
    },

    2: {
      class_type: 'LoadImage',
      inputs: { image: imageName }
    },

    3: {
      class_type: 'VAEEncode',
      inputs: { pixels: ['2', 0], vae: ['1', 2] }
    },

    // ✅ Positive prompt
    4: {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: 'seamless repeating checkerboard fabric texture, black and gray squares, perfectly tileable, flat scan, no folds, no clothing shape, uniform lighting, texture map',
        clip: ['1', 1]
      }
    },

    // ❌ Negative prompt
    5: {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: '3d, folds, wrinkles, hoodie, sleeves, collar, perspective, shadows, lighting gradient, depth, object shape',
        clip: ['1', 1]
      }
    },

    // 🔥 TILE CONTROLNET
    6: {
      class_type: 'ControlNetLoader',
      inputs: {
        control_net_name: 'control_v11f1e_sd15_tile_fp16.safetensors'
      }
    },

    7: {
      class_type: 'ControlNetApplyAdvanced',
      inputs: {
        positive: ['4', 0],
        negative: ['5', 0],
        control_net: ['6', 0],
        image: ['2', 0],
        strength: 1.0,
        start_percent: 0.0,
        end_percent: 1.0
      }
    },

    // 🔥 SAMPLER (KEY FIXES HERE)
    8: {
      class_type: 'KSampler',
      inputs: {
        model: ['1', 0],
        positive: ['7', 0],
        negative: ['7', 1],
        latent_image: ['3', 0],
        seed: Math.floor(Math.random() * 1000000),
        steps: 30,
        cfg: 7,
        sampler_name: 'dpmpp_2m',
        scheduler: 'karras',
        denoise: 0.8 // 🔥 VERY IMPORTANT
      }
    },

    9: {
      class_type: 'VAEDecode',
      inputs: { samples: ['8', 0], vae: ['1', 2] }
    },

    10: {
      class_type: 'SaveImage',
      inputs: {
        images: ['9', 0],
        filename_prefix: 'texture'
      }
    }
  }

  // Send prompt
  const promptRes = await fetch('http://127.0.0.1:8188/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: workflow })
  })

  const promptData = await promptRes.json()
  const promptId = promptData.prompt_id

  // Poll result
  while (true) {
    await new Promise((r) => setTimeout(r, 1000))

    const historyRes = await fetch(`http://127.0.0.1:8188/history/${promptId}`)
    const history = await historyRes.json()

    if (history[promptId]) {
      const outputs = history[promptId].outputs
      const imageInfo = outputs['10'].images[0]

      const imgRes = await fetch(
        `http://127.0.0.1:8188/view?filename=${imageInfo.filename}&subfolder=${imageInfo.subfolder}&type=${imageInfo.type}`
      )

      return Buffer.from(await imgRes.arrayBuffer())
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

    // console.log('Removing background...')
    // const blob = new Blob([req.file.buffer], { type: req.file.mimetype })
    // const resultBlob = await removeBackground(blob)
    // const cleanBuffer = Buffer.from(await resultBlob.arrayBuffer())

    console.log('Extracting texture...')
    // const textureBuffer = await extractTexture(cleanBuffer)
    const textureBuffer = await extractTextureWithComfyUI(req.file.buffer)

    const filename = `${uuidv4()}.png`
    const filepath = path.join(outputDir, filename)
    fs.writeFileSync(filepath, textureBuffer)
    latestTexturePath = filepath
    textureVersion++ // ← increment on every new upload

    console.log('Saved:', filename)

    const BASE_URL =
      'https://nickolas-aciniform-misunderstandingly.ngrok-free.dev'

    const textureUrl = `${BASE_URL}/textures/${filename}`

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
