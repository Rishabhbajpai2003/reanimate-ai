import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const ALLOWED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff']
const MAX_MB  = 20

export default function UploadBox({ onFileSelected, disabled }) {
  const [dragActive, setDragActive] = useState(false)

  const onDrop = useCallback((accepted, rejected) => {
    setDragActive(false)
    if (rejected.length) {
      alert(rejected[0]?.errors?.[0]?.message || 'Invalid file')
      return
    }
    if (accepted.length) onFileSelected(accepted[0])
  }, [onFileSelected])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    accept: Object.fromEntries(ALLOWED.map(t => [t, []])),
    maxFiles: 1,
    maxSize: MAX_MB * 1024 * 1024,
    disabled,
  })

  return (
    <div
      {...getRootProps()}
      className={`upload-box ${isDragActive || dragActive ? 'drag-active' : ''}`}
      id="upload-dropzone"
      role="button"
      aria-label="Upload portrait image"
      tabIndex={disabled ? -1 : 0}
    >
      <input {...getInputProps()} id="file-input" />
      <span className="upload-icon">🖼️</span>
      <h3>
        {isDragActive ? 'Drop your portrait here' : 'Drag & drop a portrait'}
      </h3>
      <p>or click to browse — JPG, PNG, WEBP up to {MAX_MB} MB</p>
    </div>
  )
}
