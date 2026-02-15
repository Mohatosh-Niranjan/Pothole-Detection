from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from .realtime_store import upsert_gps


app = FastAPI(title="PotholeTrack GPS API")

# Add CORS middleware to allow requests from any origin (needed for phone browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GPSPing(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    timestamp: str | None = None


@app.post("/gps")
def receive_gps(ping: GPSPing):
    ts = ping.timestamp or (datetime.utcnow().isoformat() + "Z")
    upsert_gps(ping.device_id, ping.latitude, ping.longitude, ts)
    return {"status": "ok"}


