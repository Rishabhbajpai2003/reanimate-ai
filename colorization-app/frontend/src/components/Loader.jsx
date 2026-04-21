export default function Loader({ steps = [] }) {
  const substeps = [
    'Denoising and removing artifacts…',
    'Applying super resolution…',
    'Colorizing portrait…',
    'Enhancing facial details…',
    'Finalising output…',
  ]
  const activeIdx = steps.filter(s => !s.skipped).length
  const hint = substeps[Math.min(activeIdx, substeps.length - 1)]

  return (
    <div className="loader-overlay" role="status" aria-live="polite">
      <div className="loader-ring" aria-hidden="true" />
      <div className="loader-text">
        Processing your portrait…
        <div className="loader-substep">{hint}</div>
      </div>
    </div>
  )
}
