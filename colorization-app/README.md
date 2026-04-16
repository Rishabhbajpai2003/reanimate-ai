# ReAnimate AI (Colorize + Cartoon/Anime)

Restore old portraits in two steps:

- **Colorize**: turn a grayscale portrait into a plausible color photo
- **Cartoon/Anime**: convert a photo into a clean stylized look (anime-like / cartoon / sketch)

This solves a real-world need for **archival photo restoration** (family albums, museums, historical records) and also creates **shareable stylized portraits** for social media and creative projects.

## Run locally (Windows / PowerShell)

### Backend (Flask)

```powershell
cd "colorization-app\backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend URL: `http://127.0.0.1:5000`  
Health check: `http://127.0.0.1:5000/health`

### Frontend (Vite + React)

Open a second terminal:

```powershell
cd "colorization-app\frontend"
npm install
npm run dev
```

Frontend URL: `http://localhost:3000`

## Model files

Place these files in `colorization-app/backend/model/`:

- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

## Notes

- **PowerShell tip**: if you test with curl, use `curl.exe` (PowerShell aliases `curl` to `Invoke-WebRequest`).
- **Best results**: portraits with clear faces work best for both colorization and stylization.
