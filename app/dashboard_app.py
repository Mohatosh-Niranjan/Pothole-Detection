import os
import sys
import time
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.realtime_store import get_recent_detections
from app.heatmap import build_map


st.set_page_config(page_title="PotholeTrack Dashboard", layout="wide")
st.title("🛰️ PotholeTrack Live Dashboard")
st.caption("Public-facing live map of pothole detections. Auto-refreshes.")

refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 2, 60, 5)
max_points = st.sidebar.slider("Max points", 100, 5000, 1000, step=100)

placeholder = st.empty()

def render_once():
    detections = get_recent_detections(limit=max_points)
    df = pd.DataFrame(detections)
    if df.empty:
        st.info("No detections yet.")
        return
    # Summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Recent Detections", len(df))
    with c2:
        st.metric("High Risk", int((df["risk_level"] == "High").sum()))
    with c3:
        st.metric("Unique Frames", df["frame_id"].nunique())

    # Map
    st.subheader("Live Map")
    m = build_map(df)
    st_folium(m, width=None, height=650, key=f"live_map_{len(df)}")

    # Table
    st.subheader("Recent Detections")
    st.dataframe(df.tail(200))

while True:
    with placeholder.container():
        render_once()
    time.sleep(refresh_seconds)


