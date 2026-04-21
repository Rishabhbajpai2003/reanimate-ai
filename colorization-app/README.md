# ReAnimateAI 🎞

**AI-powered portrait restoration pipeline** — denoising, super resolution, colorization,
face enhancement, and optional talking-head animation in one production-ready web application.

---

## ✨ Features

| Module | Technology | Fallback |
|---|---|---|
| Restoration | GFPGAN v1.4 | OpenCV NLM denoising |
| Colorization | Zhang et al. (OpenCV DNN) | Vibrance boost |
| Super Resolution | Real-ESRGAN ×2 | Lanczos4 + unsharp mask |
| Face Enhancement | CodeFormer | CLAHE + bilateral filter |
| Animation | SadTalker | FFmpeg static loop |

Every module gracefully degrades to a CPU-friendly fallback if the heavy model is not installed.

---

## 📁 Project Structure

```
reanimateai/
├── backend/
│   ├── app.py                  ← Flask REST API
│   ├── download_models.py      ← Model downloader CLI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/                 ← Downloaded weights go here
│   ├── static/
│   │   ├── uploads/            ← Input images
│   │   └── results/            ← Processed outputs
│   └── pipeline/
│       ├── __init__.py
│       ├── main.py             ← PipelineController
│       ├── restore.py          ← GFPGAN / OpenCV
│       ├── super_res.py        ← Real-ESRGAN / Lanczos
│       ├── colorize.py         ← Zhang DNN / vibrance
│       ├── enhance.py          ← CodeFormer / CLAHE
│       └── animate.py          ← SadTalker / FFmpeg
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── index.css
│   │   └── components/
│   │       ├── UploadBox.jsx   ← Drag-and-drop
│   │       ├── ImagePreview.jsx
│   │       ├── ResultViewer.jsx← Compare slider
│   │       ├── Loader.jsx
│   │       └── OptionsPanel.jsx← Per-module toggles
│   ├── Dockerfile
│   ├── vite.config.js
│   └── package.json
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models (optional but recommended)
python download_models.py --all

# Start the Flask development server
python app.py
# → http://localhost:5000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

Open `http://localhost:3000` in your browser.

---

## 🧠 Model Download Details

Use the included downloader to fetch model weights:

```bash
cd backend

# Download everything
python download_models.py --all

# Or selectively:
python download_models.py --colorize      # Zhang et al. colorization DNN
python download_models.py --gfpgan        # GFPGAN v1.4 restoration
python download_models.py --realesrgan    # Real-ESRGAN x2 + x4
python download_models.py --codeformer    # CodeFormer face enhancement
```

Models are saved in `backend/models/`:

| File | Size | Purpose |
|---|---|---|
| `colorization_deploy_v2.prototxt` | ~3 KB | Colorization architecture |
| `colorization_release_v2.caffemodel` | ~130 MB | Colorization weights |
| `pts_in_hull.npy` | ~2 KB | Cluster centres |
| `GFPGANv1.4.pth` | ~332 MB | Restoration |
| `RealESRGAN_x2plus.pth` | ~67 MB | Super Resolution ×2 |
| `RealESRGAN_x4plus.pth` | ~67 MB | Super Resolution ×4 |
| `codeformer.pth` | ~375 MB | Face Enhancement |

### SadTalker (Animation)

SadTalker requires additional setup:

```bash
git clone https://github.com/OpenTalker/SadTalker.git ../SadTalker
cd ../SadTalker
pip install -r requirements.txt
bash scripts/download_models.sh
```

---

## 🐳 Docker Deployment

### Run with Docker Compose

```bash
# Build and start all services
docker compose up --build

# Frontend → http://localhost:80
# Backend  → http://localhost:5000
```

### With GPU (NVIDIA)

Uncomment the `deploy.resources` block in `docker-compose.yml` and ensure
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
is installed.

---

## ☁️ Production Deployment

### Backend (Railway / Render / AWS EC2)

**Railway:**
```bash
railway login
railway init
railway up --service backend
# Set env: PORT=5000, FLASK_ENV=production
```

**Render:**
- Connect GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --timeout 300 app:app`
- Add persistent disk for `/app/models` and `/app/static`

**AWS EC2 (Ubuntu):**
```bash
sudo apt update && sudo apt install -y python3-venv ffmpeg libgl1
git clone https://github.com/youruser/reanimateai.git
cd reanimateai/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 app:app &
```

### Frontend (Vercel / Netlify)

**Vercel:**
```bash
cd frontend
npm run build
vercel --prod
# Set env: VITE_API_URL=https://your-backend-url.com
```

**Netlify:**
- Connect repo, set build command: `npm run build`, publish dir: `dist`
- Add `_redirects`: `/api/* https://your-backend.com/api/:splat 200`

---

## 🛡️ API Reference

### `POST /api/upload`

Upload an image and run the pipeline.

**Form Data:**

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | File | required | Portrait image (JPG, PNG, WEBP, BMP) |
| `restore` | bool | true | Enable restoration |
| `colorize` | bool | true | Enable colorization |
| `super_res` | bool | true | Enable super resolution |
| `enhance` | bool | true | Enable face enhancement |
| `animate` | bool | false | Enable animation |
| `audio` | File | — | Audio file for animation |

**Response:**
```json
{
  "job_id": "abc123",
  "elapsed_seconds": 4.2,
  "original_url": "/static/uploads/abc123_input.jpg",
  "result_url": "/static/results/abc123_final.png",
  "steps": [
    { "step": "restore",   "latency_s": 0.8,  "skipped": false },
    { "step": "colorize",  "latency_s": 1.2,  "skipped": false },
    { "step": "super_res", "latency_s": 1.4,  "skipped": false },
    { "step": "enhance",   "latency_s": 0.8,  "skipped": false }
  ],
  "intermediates": {
    "restore":   "/static/results/abc123_restore.png",
    "colorize":  "/static/results/abc123_colorize.png",
    "super_res": "/static/results/abc123_super_res.png",
    "enhance":   "/static/results/abc123_enhance.png"
  }
}
```

### `GET /api/health`

Returns `{ "status": "ok", "service": "ReAnimateAI", "version": "1.0.0" }`.

### `GET /api/result/<job_id>`

Returns result URL for a previously processed job.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Backend server port |
| `FLASK_ENV` | `development` | `development` or `production` |
| `VITE_API_URL` | `` (empty) | Frontend API base URL for production |

---

## 📊 Performance Notes

- All models are loaded **once** at startup (singleton pattern)
- GPU is used automatically when `torch.cuda.is_available()`
- Images > 1024px are resized before processing for memory safety
- Pipeline logs per-module latency to stdout

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `libGL.so.1: cannot open` | `apt install libgl1` |
| Out of memory | Reduce `max_dim` in `restore.py` or use CPU |
| Colorize DNN not working | Run `python download_models.py --colorize` |
| Port already in use | Change `PORT` env var |
| CORS error in browser | Ensure Flask is running on `localhost:5000` |
