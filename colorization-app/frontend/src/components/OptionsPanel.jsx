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
    desc: 'Real-ESRGAN ×2 upscaling',
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

export default function OptionsPanel({ options, onChange }) {
  const toggle = (key) => onChange({ ...options, [key]: !options[key] })

  return (
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
  )
}
