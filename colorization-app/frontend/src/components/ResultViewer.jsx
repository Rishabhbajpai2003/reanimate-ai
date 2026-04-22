import { useRef, useState, useCallback, useEffect } from 'react'

const STEP_META = {
  restore:   { icon: '🧹', label: 'Restoration' },
  colorize:  { icon: '🎨', label: 'Colorization' },
  super_res: { icon: '🔭', label: 'Super Resolution' },
  enhance:   { icon: '✨', label: 'Face Enhancement' },
  animate:   { icon: '🎭', label: 'Animation' },
}

const SR_MODEL_LABEL = {
  realesrgan: 'Real-ESRGAN',
  swinir: 'SwinIR',
  hat: 'HAT',
}

const SR_BACKEND_LABEL = {
  native: 'native model',
  'native-cpu-fallback': 'native model (CPU fallback)',
  'fallback-opencv': 'OpenCV fallback',
}

const COLOR_MODEL_LABEL = {
  eccv16: 'Zhang (ECCV16)',
  deoldify_artistic: 'DeOldify (Artistic)',
  deoldify_stable: 'DeOldify (Stable)',
  ddcolor: 'DDColor (ICCV23)',
}

const COLOR_BACKEND_LABEL = {
  'opencv-dnn': 'OpenCV DNN',
  deoldify: 'DeOldify',
  'modelscope-ddcolor': 'DDColor (ModelScope)',
  'opencv-dnn-fallback': 'fallback (OpenCV DNN)',
}

function getStepMeta(stepName) {
  if (stepName.startsWith('super_res:')) {
    const model = stepName.split(':')[1] || ''
    return {
      icon: '🔭',
      label: `Super Resolution (${SR_MODEL_LABEL[model] || model})`,
    }
  }
  return STEP_META[stepName] ?? { icon: '⚙️', label: stepName }
}

/* ── Compare Slider ─────────────────────────────────────────────────────────── */
function CompareSlider({ originalSrc, resultSrc }) {
  const [pos, setPos] = useState(50)
  const wrapRef = useRef(null)
  const dragging = useRef(false)

  const onMouseMove = useCallback((e) => {
    if (!dragging.current || !wrapRef.current) return
    const rect = wrapRef.current.getBoundingClientRect()
    const x    = Math.max(0, Math.min(e.clientX - rect.left, rect.width))
    setPos(Math.round((x / rect.width) * 100))
  }, [])

  const startDrag = (e) => { dragging.current = true; e.preventDefault() }
  const stopDrag  = ()   => { dragging.current = false }

  return (
    <div
      ref={wrapRef}
      className="compare-wrap"
      onMouseMove={onMouseMove}
      onMouseUp={stopDrag}
      onMouseLeave={stopDrag}
      aria-label="Before/after comparison slider"
    >
      <img src={originalSrc} alt="Original portrait" />
      <div className="compare-overlay" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={resultSrc} alt="Restored portrait" />
      </div>
      <div
        className="compare-handle"
        style={{ left: `${pos}%` }}
        onMouseDown={startDrag}
        role="slider"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pos}
        aria-label="Comparison divider"
        tabIndex={0}
      />
    </div>
  )
}

/* ── Step Timeline ───────────────────────────────────────────────────────────── */
function StepTimeline({ steps }) {
  return (
    <div className="steps-timeline" aria-label="Pipeline steps">
      {steps.map((s, i) => {
        const meta = getStepMeta(s.step)
        const status = s.skipped ? 'skip' : s.error ? 'err' : 'ok'
        return (
          <div className="step-row" key={i} style={{ animationDelay: `${i * 60}ms` }}>
            <span className="step-icon" aria-hidden="true">{meta.icon}</span>
            <span className="step-label">{meta.label}</span>
            <span className="step-latency">
              {s.skipped ? '—' : `${s.latency_s}s`}
            </span>
            <span className={`step-status ${status}`}>
              {status === 'ok' ? '✓ done' : status === 'skip' ? 'skipped' : '✗ error'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

/* ── Intermediate Grid ───────────────────────────────────────────────────────── */
function IntermediatesGrid({ intermediates }) {
  const entries = Object.entries(intermediates).filter(([k]) => k !== 'animate' && !k.startsWith('super_res_') && !k.startsWith('colorize_'))
  if (!entries.length) return null
  return (
    <div className="intermediates-grid">
      {entries.map(([step, url]) => {
        const meta = STEP_META[step] ?? { label: step }
        return (
          <div className="intermediate-card" key={step}>
            <img src={url} alt={`${meta.label} output`} loading="lazy" />
            <div className="intermediate-label">{meta.icon ?? ''} {meta.label}</div>
          </div>
        )
      })}
    </div>
  )
}

function SrComparisonGrid({ outputs, activeModel, onSelect }) {
  const entries = Object.entries(outputs || {})
  if (!entries.length) return null

  return (
    <div className="sr-comparison-section">
      <div className="card-title" style={{ marginTop: '1.25rem' }}>
        <span>🧪</span> Super-Resolution Model Comparison
      </div>
      <div className="sr-comparison-grid">
        {entries.map(([model, meta]) => {
          const isActive = activeModel === model
          return (
            <button
              key={model}
              type="button"
              className={`sr-output-card ${isActive ? 'active' : ''}`}
              onClick={() => onSelect(model)}
            >
              <img src={meta.url} alt={`${SR_MODEL_LABEL[model] || model} output`} loading="lazy" />
              <div className="sr-output-meta">
                <span>{SR_MODEL_LABEL[model] || model}</span>
                <small>
                  {SR_BACKEND_LABEL[meta.backend] || meta.backend || 'unknown backend'}
                  {typeof meta.latency_s === 'number' ? ` · ${meta.latency_s}s` : ''}
                </small>
                {!!meta.metrics && (
                  <small>
                    {typeof meta.metrics.psnr === 'number' ? `PSNR ${meta.metrics.psnr} dB` : 'PSNR n/a'}
                    {' · '}
                    {typeof meta.metrics.ssim === 'number' ? `SSIM ${meta.metrics.ssim}` : 'SSIM n/a'}
                    {' · '}
                    {typeof meta.metrics.lpips === 'number' ? `LPIPS ${meta.metrics.lpips}` : 'LPIPS n/a'}
                  </small>
                )}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/* ── Animation Viewer ────────────────────────────────────────────────────────── */
function AnimationViewer({ animationUrl }) {
  if (!animationUrl) return null

  const isGif   = animationUrl.endsWith('.gif')
  const isVideo = animationUrl.endsWith('.mp4') || animationUrl.endsWith('.webm')

  return (
    <div className="animation-section">
      <div className="card-title" style={{ marginBottom: '0.75rem' }}>
        <span>🎭</span> Animated Portrait
      </div>
      <div className="animation-display">
        {isGif && (
          <img
            src={animationUrl}
            alt="Animated portrait — head moving, breathing, blinking"
            className="animation-gif"
            id="animation-result"
          />
        )}
        {isVideo && (
          <video
            src={animationUrl}
            autoPlay
            loop
            muted
            playsInline
            className="animation-gif"
            id="animation-result"
          />
        )}
        {!isGif && !isVideo && (
          <img src={animationUrl} alt="Animation output" className="animation-gif" />
        )}
        <div className="animation-badge">LIVE</div>
      </div>
      <a
        href={animationUrl}
        download={`reanimateai_animation.${isGif ? 'gif' : 'mp4'}`}
        className="btn-download"
        style={{ marginTop: '0.75rem' }}
      >
        ⬇ Download Animation ({isGif ? 'GIF' : 'MP4'})
      </a>
    </div>
  )
}

function ColorComparisonGrid({ outputs, activeModel, onSelect }) {
  const entries = Object.entries(outputs || {})
  if (!entries.length) return null

  return (
    <div className="sr-comparison-section">
      <div className="card-title" style={{ marginTop: '1.25rem' }}>
        <span>🎨</span> Colorization Model Comparison
      </div>
      <div className="sr-comparison-grid">
        {entries.map(([model, meta]) => {
          const isActive = activeModel === model
          return (
            <button
              key={model}
              type="button"
              className={`sr-output-card ${isActive ? 'active' : ''}`}
              onClick={() => onSelect(model)}
            >
              <img src={meta.url} alt={`${COLOR_MODEL_LABEL[model] || model} output`} loading="lazy" />
              <div className="sr-output-meta">
                <span>{COLOR_MODEL_LABEL[model] || model}</span>
                <small>
                  {COLOR_BACKEND_LABEL[meta.backend] || meta.backend || 'unknown backend'}
                  {typeof meta.latency_s === 'number' ? ` · ${meta.latency_s}s` : ''}
                </small>
                {!!meta.metrics && (
                  <small>
                    {typeof meta.metrics.psnr === 'number' ? `PSNR ${meta.metrics.psnr} dB` : 'PSNR n/a'}
                    {' · '}
                    {typeof meta.metrics.ssim === 'number' ? `SSIM ${meta.metrics.ssim}` : 'SSIM n/a'}
                    {' · '}
                    {typeof meta.metrics.lpips === 'number' ? `LPIPS ${meta.metrics.lpips}` : 'LPIPS n/a'}
                  </small>
                )}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/* ── Main Result Viewer ─────────────────────────────────────────────────────── */
export default function ResultViewer({ result, originalSrc }) {
  const [showIntermediates, setShowIntermediates] = useState(false)
  const [compareMode, setCompareMode]             = useState(false)
  const [selectedSrModel, setSelectedSrModel]     = useState(() => {
    const models = Object.keys(result?.sr_compare_outputs || {})
    if (models.includes('realesrgan')) return 'realesrgan'
    return models[0] || null
  })

  const [selectedColorModel, setSelectedColorModel] = useState(() => {
    const models = Object.keys(result?.color_compare_outputs || {})
    if (models.includes('eccv16')) return 'eccv16'
    return models[0] || null
  })

  useEffect(() => {
    const models = Object.keys(result?.sr_compare_outputs || {})
    if (!models.length) {
      setSelectedSrModel(null)
      return
    }
    setSelectedSrModel(models.includes('realesrgan') ? 'realesrgan' : models[0])
  }, [result])

  useEffect(() => {
    const models = Object.keys(result?.color_compare_outputs || {})
    if (!models.length) {
      setSelectedColorModel(null)
      return
    }
    setSelectedColorModel(models.includes('eccv16') ? 'eccv16' : models[0])
  }, [result])

  if (!result) return null

  const hasIntermediates = Object.keys(result.intermediates ?? {}).filter(
    k => k !== 'animate'
  ).length > 0
  const hasAnimation = !!result.animation_url
  const hasSrCompare = Object.keys(result.sr_compare_outputs ?? {}).length > 0
  const hasColorCompare = Object.keys(result.color_compare_outputs ?? {}).length > 0
  const activeSrUrl = selectedSrModel && result.sr_compare_outputs?.[selectedSrModel]?.url
  const activeColorUrl = selectedColorModel && result.color_compare_outputs?.[selectedColorModel]?.url
  const displayUrl = activeSrUrl || activeColorUrl || result.result_url
  const metrics = result.metrics || {}

  const metricRows = [
    {
      key: 'psnr',
      label: 'PSNR',
      unit: 'dB',
      hint: 'Higher is better',
    },
    {
      key: 'ssim',
      label: 'SSIM',
      unit: '',
      hint: 'Higher is better',
    },
    {
      key: 'lpips',
      label: 'LPIPS',
      unit: '',
      hint: 'Lower is better',
    },
  ]

  const hasMetricValues = metricRows.some(({ key }) => typeof metrics[key] === 'number')

  return (
    <div className="card result-section" id="result-section">
      <div className="card-title">
        <span>✅</span> Restored Portrait
        <span style={{ marginLeft: 'auto', fontSize: '0.78rem', fontWeight: 400, color: 'var(--clr-text-muted)' }}>
          Total: {result.elapsed_seconds}s
        </span>
      </div>

      {/* Compare / plain toggle */}
      {originalSrc && (
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem', fontSize: '0.8rem' }}>
          <button
            onClick={() => setCompareMode(false)}
            style={{
              padding: '0.3rem 0.75rem',
              borderRadius: '99px',
              border: `1px solid ${compareMode ? 'var(--clr-border)' : 'var(--clr-primary)'}`,
              background: compareMode ? 'transparent' : 'rgba(79,140,247,0.1)',
              color: compareMode ? 'var(--clr-text-muted)' : 'var(--clr-primary)',
              cursor: 'pointer',
            }}
          >Result</button>
          <button
            onClick={() => setCompareMode(true)}
            style={{
              padding: '0.3rem 0.75rem',
              borderRadius: '99px',
              border: `1px solid ${compareMode ? 'var(--clr-primary)' : 'var(--clr-border)'}`,
              background: compareMode ? 'rgba(79,140,247,0.1)' : 'transparent',
              color: compareMode ? 'var(--clr-primary)' : 'var(--clr-text-muted)',
              cursor: 'pointer',
            }}
          >Compare ↔</button>
        </div>
      )}

      {compareMode && originalSrc ? (
        <CompareSlider originalSrc={originalSrc} resultSrc={displayUrl} />
      ) : (
        <div className="result-image-wrap">
          <img src={displayUrl} alt="Restored portrait" id="result-img" />
          <div className="result-glow" aria-hidden="true" />
        </div>
      )}

      {/* Download enhanced image */}
      <a
        href={displayUrl}
        download="reanimateai_result.png"
        className="btn-download"
        id="btn-download"
        rel="noreferrer"
      >
        ⬇ Download Enhanced Portrait
      </a>

      <div className="metrics-section">
        <div className="card-title" style={{ marginBottom: '0.55rem' }}>
          <span>📊</span> Evaluation Metrics
        </div>
        <p className="metrics-caption">Computed against the uploaded input image.</p>
        <div className="metrics-grid">
          {metricRows.map((m) => {
            const raw = metrics[m.key]
            const hasValue = typeof raw === 'number' && Number.isFinite(raw)
            const display = hasValue ? `${raw}${m.unit ? ` ${m.unit}` : ''}` : 'n/a'

            return (
              <div className="metric-card" key={m.key}>
                <div className="metric-label">{m.label}</div>
                <div className="metric-value">{display}</div>
                <div className="metric-hint">{m.hint}</div>
              </div>
            )
          })}
        </div>
        {!hasMetricValues && (
          <p className="metrics-warning">
            Metrics are unavailable for this run. Install optional metric dependencies to enable full evaluation.
          </p>
        )}
      </div>

      {/* ── Animation GIF / Video ──────────────────────────────────────────── */}
      {hasAnimation && (
        <AnimationViewer animationUrl={result.animation_url} />
      )}

      {hasSrCompare && (
        <SrComparisonGrid
          outputs={result.sr_compare_outputs}
          activeModel={selectedSrModel}
          onSelect={setSelectedSrModel}
        />
      )}

      {hasColorCompare && (
        <ColorComparisonGrid
          outputs={result.color_compare_outputs}
          activeModel={selectedColorModel}
          onSelect={setSelectedColorModel}
        />
      )}

      {/* Steps */}
      {result.steps?.length > 0 && (
        <div style={{ marginTop: '1.25rem' }}>
          <div className="card-title"><span>⚡</span> Pipeline Timing</div>
          <StepTimeline steps={result.steps} />
        </div>
      )}

      {/* Intermediates */}
      {hasIntermediates && (
        <>
          <button
            className="intermediates-toggle"
            onClick={() => setShowIntermediates(v => !v)}
            aria-expanded={showIntermediates}
          >
            {showIntermediates ? '▾' : '▸'}&nbsp;
            {showIntermediates ? 'Hide' : 'Show'} intermediate outputs
          </button>
          {showIntermediates && <IntermediatesGrid intermediates={result.intermediates} />}
        </>
      )}
    </div>
  )
}
