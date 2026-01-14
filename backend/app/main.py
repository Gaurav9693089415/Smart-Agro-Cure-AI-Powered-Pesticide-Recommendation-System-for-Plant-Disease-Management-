from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.multimodal_chat_api import router as multimodal_router

app = FastAPI(
    title="Smart Agro-Cure API",
    version="0.1.0",
)

# --------------------------------------------------------------------
# CORS (so your frontend JS can talk to FastAPI without issues)
# --------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Routers
# --------------------------------------------------------------------
app.include_router(multimodal_router, prefix="/api")
