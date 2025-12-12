# src/api/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import ingest_router, ask_router

app = FastAPI(title="PolicyNavigator AI API")

# CORS – allow your React web page to call the API (adjust origin if you deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, tighten this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router.router)
app.include_router(ask_router.router)


@app.get("/")
def root():
    return {"status": "ok", "app": "PolicyNavigator AI API"}
