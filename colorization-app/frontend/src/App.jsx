import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const BACKEND_URL =
    import.meta?.env?.VITE_BACKEND_URL?.replace(/\/+$/, "") ||
    "http://127.0.0.1:5000";
  const fileInputRef = useRef(null);
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [quality, setQuality] = useState("balanced");
  const [clahe, setClahe] = useState(false);
  const [saturation, setSaturation] = useState(1.08);
  const [sharpness, setSharpness] = useState(-1);
  const [stylePreset, setStylePreset] = useState("clean_anime");
  const [styleStrength, setStyleStrength] = useState(0.75);
  const [mode, setMode] = useState("colorize"); // colorize | stylize
  const [comparePos, setComparePos] = useState(50);
  const [compareBeforeSrc, setCompareBeforeSrc] = useState(null);
  const [compareDims, setCompareDims] = useState(null);
  const [result, setResult] = useState(null);
  const [stylized, setStylized] = useState(null);
  const [stylizeLoading, setStylizeLoading] = useState(false);
  const [loading, setLoading] = useState(false);

  const effectiveSharpness = sharpness < 0 ? null : sharpness;

  /** Automatic color strength per quality preset (Advanced can still override). */
  useEffect(() => {
    if (quality === "fast") setSaturation(1.05);
    else if (quality === "best") setSaturation(1.12);
    else setSaturation(1.08);
  }, [quality]);

  /**
   * Resize "before" to match output pixel size. Also record output w×h so the
   * compare frame uses the same aspect ratio as the JPEG (fixed 4:5 + .card img
   * max-width was breaking alignment).
   */
  useEffect(() => {
    if (!result || !preview) {
      setCompareBeforeSrc(null);
      setCompareDims(null);
      return;
    }

    let cancelled = false;

    const run = async () => {
      const outImg = new Image();
      outImg.src = result;
      try {
        await outImg.decode();
      } catch {
        if (!cancelled) {
          setCompareBeforeSrc(null);
          setCompareDims(null);
        }
        return;
      }
      if (cancelled) return;

      const w = outImg.naturalWidth;
      const h = outImg.naturalHeight;
      if (!w || !h) return;

      setCompareDims({ w, h });

      const prevImg = new Image();
      prevImg.src = preview;
      try {
        await prevImg.decode();
      } catch {
        if (!cancelled) setCompareBeforeSrc(null);
        return;
      }
      if (cancelled) return;

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(prevImg, 0, 0, w, h);
      if (!cancelled) {
        setCompareBeforeSrc(canvas.toDataURL("image/jpeg", 0.92));
      }
    };

    run();

    return () => {
      cancelled = true;
    };
  }, [result, preview]);

  /** Always derive a luminance preview — matches backend (color uploads are treated as B&W before colorization). */
  useEffect(() => {
    if (!image) {
      setPreview(null);
      return;
    }

    let cancelled = false;
    const objectUrl = URL.createObjectURL(image);
    // For Stylize: show true color preview (matches what the backend stylizes).
    // For Colorize: show a luminance preview (matches what the backend colorizes).
    if (mode === "stylize") {
      setPreview(objectUrl);
    } else {
      const im = new Image();
      im.onload = () => {
        if (cancelled) return;
        const canvas = document.createElement("canvas");
        canvas.width = im.naturalWidth;
        canvas.height = im.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(im, 0, 0);
        const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = d.data;
        for (let i = 0; i < data.length; i += 4) {
          const g = Math.round(
            0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
          );
          data[i] = g;
          data[i + 1] = g;
          data[i + 2] = g;
        }
        ctx.putImageData(d, 0, 0);
        setPreview(canvas.toDataURL("image/jpeg", 0.92));
      };
      im.onerror = () => {};
      im.src = objectUrl;
    }

    return () => {
      cancelled = true;
      URL.revokeObjectURL(objectUrl);
    };
  }, [image, mode]);

  const handleChange = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImage(file);
    setResult(null);
    if (stylized) URL.revokeObjectURL(stylized);
    setStylized(null);
  };

  const openFilePicker = () => fileInputRef.current?.click();

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleChange(file);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleUpload = async () => {
    if (!image) {
      alert("Please upload an image!");
      return;
    }

    setLoading(true);
    if (result) {
      URL.revokeObjectURL(result);
      setResult(null);
    }

    const formData = new FormData();
    formData.append("image", image);
    formData.append("grayscale_only", "1");
    formData.append("quality", quality);
    formData.append("clahe", clahe ? "1" : "0");
    formData.append("saturation", String(saturation));
    if (effectiveSharpness !== null) {
      formData.append("sharpness", String(effectiveSharpness));
    }

    try {
      const res = await fetch(`${BACKEND_URL}/colorize`, {
        method: "POST",
        body: formData,
      });

      const blob = await res.blob();
      if (!res.ok) {
        let msg = "Colorization failed";
        try {
          const t = await blob.text();
          const j = JSON.parse(t);
          if (j.error) msg = j.error;
        } catch {
          /* ignore */
        }
        alert(msg);
        return;
      }

      setResult(URL.createObjectURL(blob));
    } catch (err) {
      console.error(err);
      alert(
        `Error processing image — is the backend running at ${BACKEND_URL}?`
      );
    } finally {
      setLoading(false);
    }
  };

  const handleStylize = async () => {
    if (!image) {
      alert("Please upload an image!");
      return;
    }

    setStylizeLoading(true);
    if (stylized) {
      URL.revokeObjectURL(stylized);
      setStylized(null);
    }

    const formData = new FormData();
    formData.append("image", image);
    formData.append("max_side", "1200");
    const useAi = stylePreset === "ai_cartoon";
    if (!useAi) {
      formData.append("preset", stylePreset);
      formData.append("strength", String(styleStrength));
    }

    try {
      const res = await fetch(
        `${BACKEND_URL}/${useAi ? "stylize_ai" : "cartoonize"}`,
        {
        method: "POST",
        body: formData,
        }
      );

      const blob = await res.blob();
      if (!res.ok) {
        let msg = "Stylization failed";
        try {
          const t = await blob.text();
          const j = JSON.parse(t);
          if (j.error) msg = j.error;
        } catch {
          /* ignore */
        }
        alert(msg);
        return;
      }

      setStylized(URL.createObjectURL(blob));
    } catch (err) {
      console.error(err);
      alert(`Error stylizing image — is the backend running at ${BACKEND_URL}?`);
    } finally {
      setStylizeLoading(false);
    }
  };

  // GIF animation removed (project simplified to: Colorize + Cartoon/Anime)

  const downloadBlobUrl = (url, filename) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const downloadResult = () => {
    if (!result) return;
    const stem =
      image?.name?.replace(/\.[^/.]+$/, "")?.replace(/[^\w.-]+/g, "_") ||
      "portrait";
    downloadBlobUrl(result, `colorized-${stem}.jpg`);
  };

  const downloadStylized = () => {
    if (!stylized) return;
    const stem =
      image?.name?.replace(/\.[^/.]+$/, "")?.replace(/[^\w.-]+/g, "_") ||
      "portrait";
    downloadBlobUrl(stylized, `${stylePreset}-${stem}.png`);
  };

  const resetAll = () => {
    if (result) URL.revokeObjectURL(result);
    if (stylized) URL.revokeObjectURL(stylized);
    setImage(null);
    setResult(null);
    setPreview(null);
    setStylized(null);
    setCompareBeforeSrc(null);
    setCompareDims(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>ReAnimate</h1>
        <p className="tagline">
          Restore and stylize portraits. Pick a mode, upload an image, and export
          a clean result.
        </p>
      </header>

      <div className="mode-toggle" role="tablist" aria-label="Mode">
        <button
          type="button"
          className={`mode-chip ${mode === "colorize" ? "active" : ""}`}
          onClick={() => setMode("colorize")}
        >
          Colorize
        </button>
        <button
          type="button"
          className={`mode-chip ${mode === "stylize" ? "active" : ""}`}
          onClick={() => setMode("stylize")}
        >
          Cartoon / Anime
        </button>
      </div>

      <div
        className="upload-box"
        role="button"
        tabIndex={0}
        onClick={openFilePicker}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openFilePicker();
          }
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="upload-input"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleChange(file);
            e.target.value = "";
          }}
        />
        <p className="upload-title">Drop image here or click to browse</p>
        <p className="upload-hint">JPEG / PNG · portraits work best</p>
      </div>

      <div className="controls">
        {mode === "colorize" ? (
          <>
            <div className="control-row">
              <label className="control-label" htmlFor="quality">
                Quality
              </label>
              <select
                id="quality"
                className="control-select"
                value={quality}
                onChange={(e) => setQuality(e.target.value)}
              >
                <option value="fast">Fast — quick try</option>
                <option value="balanced">Balanced — default</option>
                <option value="best">Best — richer color (slower)</option>
              </select>
            </div>
            <p className="control-hint">
              Saturation updates with quality. Open Advanced for contrast and manual tuning.
            </p>
          </>
        ) : (
          <>
            <div className="control-row">
              <label className="control-label" htmlFor="stylePreset">
                Preset
              </label>
              <select
                id="stylePreset"
                className="control-select"
                value={stylePreset}
                onChange={(e) => setStylePreset(e.target.value)}
              >
                <option value="ai_cartoon">AI Cartoon (best)</option>
                <option value="clean_anime">Clean Anime (best)</option>
                <option value="soft_cartoon">Soft Cartoon</option>
                <option value="ink">Ink / Manga</option>
                <option value="sketch">Sketch (B/W)</option>
              </select>
            </div>

            {stylePreset !== "ai_cartoon" && (
              <div className="control-row slider-row">
                <label className="control-label" htmlFor="styleStrength">
                  Strength
                </label>
                <input
                  id="styleStrength"
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={styleStrength}
                  onChange={(e) => setStyleStrength(Number(e.target.value))}
                />
                <span className="slider-value">{styleStrength.toFixed(2)}</span>
              </div>
            )}
          </>
        )}

        <details className="advanced-panel">
          <summary className="advanced-summary">Advanced tuning (optional)</summary>
          <div className="advanced-body">
            {mode === "colorize" ? (
              <>
                <label className="demo-option">
                  <input
                    type="checkbox"
                    checked={clahe}
                    onChange={(e) => setClahe(e.target.checked)}
                  />
                  <span>Boost local contrast (CLAHE) — use for faded scans</span>
                </label>

                <div className="control-row slider-row">
                  <label className="control-label" htmlFor="sat">
                    Saturation
                  </label>
                  <input
                    id="sat"
                    type="range"
                    min="0.85"
                    max="1.4"
                    step="0.01"
                    value={saturation}
                    onChange={(e) => setSaturation(Number(e.target.value))}
                  />
                  <span className="slider-value">{saturation.toFixed(2)}×</span>
                </div>

                <div className="control-row slider-row">
                  <label className="control-label" htmlFor="sharp">
                    Sharpness
                  </label>
                  <input
                    id="sharp"
                    type="range"
                    min="-1"
                    max="0.5"
                    step="0.05"
                    value={sharpness}
                    onChange={(e) => setSharpness(Number(e.target.value))}
                  />
                  <span className="slider-value">
                    {sharpness < 0 ? "Auto" : sharpness.toFixed(2)}
                  </span>
                </div>
              </>
            ) : (
              <p className="control-hint" style={{ margin: 0 }}>
                Tip: try <b>Anime-like</b> at ~0.7 for clean lines, or <b>Cartoon</b> at ~0.5 for softer shading.
              </p>
            )}
          </div>
        </details>
      </div>

      <div className="actions">
        {mode === "colorize" ? (
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={loading}
          >
            {loading ? "Processing…" : "Colorize"}
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleStylize}
            disabled={stylizeLoading}
            title="Convert the photo to a cartoon/anime/sketch look"
          >
            {stylizeLoading ? "Stylizing…" : "Stylize"}
          </button>
        )}
        {image && (
          <button type="button" className="btn btn-ghost" onClick={resetAll}>
            Clear
          </button>
        )}
      </div>

      {(loading || stylizeLoading) && <div className="loader" aria-hidden />}

      <div className="results">
        {preview && (
          <div className="card">
            <h3>Input</h3>
            <img src={preview} alt="Input" />
          </div>
        )}

        {mode === "colorize" && result && preview && (
          <div className="card card-wide">
            <h3>Compare</h3>
            <div
              className="compare-wrap"
              style={
                compareDims
                  ? { aspectRatio: `${compareDims.w} / ${compareDims.h}` }
                  : undefined
              }
            >
              {/* After = full frame; before = same pixel size, clipped on the left */}
              <img
                className="compare-layer compare-after"
                src={result}
                alt=""
                aria-hidden
              />
              <img
                className="compare-layer compare-before"
                src={compareBeforeSrc || preview}
                alt=""
                aria-hidden
                style={{
                  clipPath: `inset(0 ${100 - comparePos}% 0 0)`,
                }}
              />
              <input
                type="range"
                className="compare-slider"
                min="0"
                max="100"
                value={comparePos}
                onChange={(e) => setComparePos(Number(e.target.value))}
                aria-label="Before and after"
              />
              <div
                className="compare-handle"
                style={{ left: `${comparePos}%` }}
              />
              <span className="compare-label compare-label-left">Before</span>
              <span className="compare-label compare-label-right">After</span>
            </div>
          </div>
        )}

        {mode === "colorize" && result && (
          <div className="card">
            <h3>Colorized</h3>
            <img src={result} alt="Colorized result" />
            <button
              type="button"
              className="btn btn-download"
              onClick={downloadResult}
            >
              Download result
            </button>
          </div>
        )}

        {mode === "stylize" && stylized && (
          <div className="card">
            <h3>Cartoon / Anime</h3>
            <img src={stylized} alt="Stylized result" />
            <button
              type="button"
              className="btn btn-download"
              onClick={downloadStylized}
            >
              Download PNG
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
