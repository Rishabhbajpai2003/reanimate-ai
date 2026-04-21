export default function ImagePreview({ src, label = 'Original' }) {
  if (!src) {
    return (
      <div className="image-placeholder">
        <span className="ph-icon">🖼</span>
        <span>{label} will appear here</span>
      </div>
    )
  }

  return (
    <img
      src={src}
      alt={label}
      className="image-preview"
      id={`preview-${label.toLowerCase().replace(/\s+/g, '-')}`}
    />
  )
}
