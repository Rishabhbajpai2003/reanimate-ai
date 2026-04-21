const OPTIONS = [
  {
    key:  'restore',
    name: 'Restoration',
    desc: 'Denoising + artifact removal',
    icon: '🧹',
  },
  {
    key:  'colorize',
    name: 'Colorization',
    desc: 'Grayscale → colour (Zhang et al.)',
    icon: '🎨',
  },
  {
    key:  'super_res',
    name: 'Super Resolution',
    desc: 'Image upscaling stage',
    icon: '🔭',
  },
  {
    key:  'enhance',
    name: 'Face Enhancement',
    desc: 'CodeFormer identity preservation',
    icon: '✨',
  },
  {
    key:  'animate',
    name: 'Animation',
    desc: 'Bring portrait to life (motion GIF)',
    icon: '🎭',
  },
]

const SR_MODELS = [
  { key: 'realesrgan', label: 'Real-ESRGAN', hint: 'Generative sharp detail' },
  { key: 'swinir', label: 'SwinIR', hint: 'Transformer SR baseline' },
  { key: 'hat', label: 'HAT', hint: 'Hybrid attention transformer' },
]

export default function OptionsPanel({ options, onChange, srModels, onSrModelsChange }) {
  const toggle = (key) => onChange({ ...options, [key]: !options[key] })
  const enabledSrModels = Object.entries(srModels).filter(([, v]) => v).map(([k]) => k)

  const toggleSrModel = (modelKey) => {
    const next = { ...srModels, [modelKey]: !srModels[modelKey] }
    const count = Object.values(next).filter(Boolean).length
    // Keep at least one SR model selected.
    if (count === 0) return
    onSrModelsChange(next)
  }

  return (
    <>
      <div className="options-grid">
        {OPTIONS.map(({ key, name, desc, icon }) => {
          const enabled = !!options[key]
          return (
            <div
              key={key}
              className={`option-toggle ${enabled ? 'enabled' : ''}`}
              onClick={() => toggle(key)}
              role="switch"
              aria-checked={enabled}
              aria-label={`${name} toggle`}
              tabIndex={0}
              onKeyDown={e => e.key === 'Enter' || e.key === ' ' ? toggle(key) : null}
              id={`toggle-${key}`}
            >
              <div className="option-info">
                <span className="option-name">{icon} {name}</span>
                <span className="option-desc">{desc}</span>
              </div>
              <div className={`toggle-switch ${enabled ? 'on' : ''}`} aria-hidden="true" />
            </div>
          )
        })}
      </div>

      {options.super_res && (
        <div className="sr-controls">
          <div className="sr-headline">🔬 SR Ablation Controls</div>
          <div className="sr-compare-toggle" onClick={() => toggle('sr_compare')}>
            <div className="option-info">
              <span className="option-name">Compare Multiple SR Models</span>
              <span className="option-desc">Generate one output per selected model</span>
            </div>
            <div className={`toggle-switch ${options.sr_compare ? 'on' : ''}`} aria-hidden="true" />
          </div>

          <div className="sr-model-grid">
            {SR_MODELS.map((m) => {
              const selected = !!srModels[m.key]
              const disabled = !selected && enabledSrModels.length === 1
              return (
                <button
                  key={m.key}
                  type="button"
                  className={`sr-model-chip ${selected ? 'selected' : ''}`}
                  onClick={() => toggleSrModel(m.key)}
                  disabled={disabled}
                  aria-pressed={selected}
                >
                  <span>{m.label}</span>
                  <small>{m.hint}</small>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </>
  )
}
