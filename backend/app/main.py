from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path

from backend.app.api.multimodal_chat_api import router as multimodal_router

app = FastAPI(
    title="Smart Agro-Cure API",
    version="0.1.0",
)

# --------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Resolve project root
# backend/app/main.py → parents[0]=app, [1]=backend, [2]=project root
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# HARD FAIL if UI folder is wrong (no silent bug)
if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend directory not found: {FRONTEND_DIR}")

# --------------------------------------------------------------------
# Serve UI
# --------------------------------------------------------------------
app.mount(
    "/ui",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="ui",
)

# --------------------------------------------------------------------
# Redirect root → UI
# --------------------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/ui/index.html")

# --------------------------------------------------------------------
# APIs
# --------------------------------------------------------------------
app.include_router(multimodal_router, prefix="/api")
