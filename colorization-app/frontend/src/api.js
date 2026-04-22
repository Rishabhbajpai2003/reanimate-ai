import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  timeout: 600_000, // 10 minutes for heavy models (e.g. SadTalker on M2)
})

/**
 * Upload an image and run the pipeline.
 * @param {File}    imageFile
 * @param {Object}  options   - { restore, super_res, colorize, enhance, animate, sr_compare, sr_models }
 * @param {File}    [audioFile] - optional audio for animation
 * @param {Function} [onProgress] - Axios upload progress callback
 */
export async function processImage(imageFile, options, audioFile = null, onProgress) {
  const form = new FormData()
  form.append('image', imageFile)

  const {
    sr_models = ['realesrgan'],
    ...boolOptions
  } = options

  Object.entries(boolOptions).forEach(([k, v]) => form.append(k, v ? 'true' : 'false'))
  form.append('sr_models', JSON.stringify(sr_models))

  if (audioFile) form.append('audio', audioFile)

  const { data } = await api.post('/api/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  })
  return data
}

/**
 * Health check.
 */
export async function checkHealth() {
  const { data } = await api.get('/api/health')
  return data
}
