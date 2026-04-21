import { useState, useEffect } from 'react'
import { Toaster, toast } from 'react-hot-toast'
import UploadBox    from './components/UploadBox.jsx'
import ImagePreview from './components/ImagePreview.jsx'
import Loader       from './components/Loader.jsx'
import ResultViewer from './components/ResultViewer.jsx'
import OptionsPanel from './components/OptionsPanel.jsx'
import { processImage, checkHealth } from './api.js'

const PIPELINE_STEPS = [
  { key: 'restore',   label: 'Restore' },
  { key: 'colorize',  label: 'Colorize' },
  { key: 'super_res', label: 'Super-Res' },
  { key: 'enhance',   label: 'Enhance' },
  { key: 'animate',   label: 'Animate' },
]

const DEFAULT_OPTIONS = {
  restore:   true,
  colorize:  true,
  super_res: true,
  enhance:   true,
  animate:   false,
}

export default function App() {
  const [file,      setFile]      = useState(null)
  const [preview,   setPreview]   = useState(null)
  const [options,   setOptions]   = useState(DEFAULT_OPTIONS)
  const [loading,   setLoading]   = useState(false)
  const [result,    setResult]    = useState(null)
  const [steps,     setSteps]     = useState([])
  const [health,    setHealth]    = useState(null)

  // ── Health check on mount ──────────────────────────────────────────────
  useEffect(() => {
    checkHealth()
      .then(d => setHealth(d.status))
      .catch(() => setHealth('offline'))
  }, [])

  // ── File selection ─────────────────────────────────────────────────────
  const handleFileSelected = (f) => {
    setFile(f)
    setResult(null)
    setSteps([])
    const url = URL.createObjectURL(f)
    setPreview(url)
    toast.success(`Portrait loaded: ${f.name}`, { className: 'toast-custom' })
  }

  // ── Submit ─────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!file) {
      toast.error('Please upload a portrait first.', { className: 'toast-custom' })
      return
    }
    setLoading(true)
    setResult(null)
    setSteps([])
    try {
      const data = await processImage(file, options)
      setResult(data)
      setSteps(data.steps ?? [])
      toast.success('Portrait restored successfully! 🎉', { className: 'toast-custom' })
      // Smooth scroll to result
      setTimeout(() => {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (err) {
      const msg = err?.response?.data?.error || err.message || 'Unknown error'
      toast.error(`Error: ${msg}`, { className: 'toast-custom', duration: 6000 })
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = !!file && !loading

  return (
    <div className="app-shell">
      <Toaster position="top-right" />

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="header" role="banner">
        <a href="/" className="logo" aria-label="ReAnimateAI home">
          <div className="logo-icon" aria-hidden="true">🎞</div>
          <span className="logo-text">ReAnimateAI</span>
        </a>
        <span className="header-badge" title={`Backend: ${health}`}>
          {health === 'ok' ? '● Live' : health === 'offline' ? '○ Offline' : '◌ Connecting'}
        </span>
      </header>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <main className="main-content" role="main">

        {/* Hero */}
        <section className="hero" aria-labelledby="hero-heading">
          <h1 id="hero-heading">AI Portrait Restoration</h1>
          <p>
            Upload a portrait and let our modular AI pipeline restore, enhance,
            and animate it — powered by Real-ESRGAN, CodeFormer, and more.
          </p>
        </section>

        {/* Pipeline steps */}
        <nav className="pipeline-steps" aria-label="Pipeline overview">
          {PIPELINE_STEPS.map((s, i) => (
            <span key={s.key}>
              <span className={`step-chip ${options[s.key] ? 'active' : ''}`}>
                <span className="dot" aria-hidden="true" />
                {s.label}
              </span>
              {i < PIPELINE_STEPS.length - 1 && (
                <span className="step-arrow" aria-hidden="true">→</span>
              )}
            </span>
          ))}
        </nav>

        {/* ── Workspace Grid ──────────────────────────────────────────── */}
        <div className="workspace-grid">

          {/* Upload Card */}
          <div className="card" id="upload-card">
            <div className="card-title">
              <span>📤</span> Upload Portrait
            </div>
            <UploadBox onFileSelected={handleFileSelected} disabled={loading} />

            <button
              className="btn-submit"
              onClick={handleSubmit}
              disabled={!canSubmit}
              id="btn-submit"
              aria-label="Restore portrait"
            >
              {loading ? 'Processing…' : '✨ Restore Portrait'}
            </button>
          </div>

          {/* Preview Card */}
          <div className="card" id="preview-card">
            <div className="card-title"><span>👁</span> Original</div>
            <ImagePreview src={preview} label="Original" />
          </div>

          {/* Options Panel */}
          <div className="card options-panel" id="options-card">
            <div className="card-title"><span>⚙️</span> Pipeline Options</div>
            <OptionsPanel options={options} onChange={setOptions} />
          </div>

          {/* Loader (replaces result while processing) */}
          {loading && (
            <div className="card result-section" id="loader-card">
              <Loader steps={steps} />
            </div>
          )}

          {/* Result Viewer */}
          {!loading && result && (
            <ResultViewer result={result} originalSrc={preview} />
          )}
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer className="footer" role="contentinfo">
        ReAnimateAI — AI Portrait Restoration &nbsp;·&nbsp; Built with Flask + React + Vite
      </footer>
    </div>
  )
}
