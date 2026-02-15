from typing import Dict

import folium
from folium.plugins import HeatMap
import pandas as pd


RISK_COLOR: Dict[str, str] = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
}


def _risk_weight(risk: str) -> float:
    if risk == "High":
        return 1.0
    if risk == "Medium":
        return 0.6
    return 0.3


def build_map(df: pd.DataFrame, default_center=(28.6139, 77.2090)) -> folium.Map:
    if df.empty:
        m = folium.Map(location=default_center, zoom_start=12, control_scale=True)
        return m

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    m = folium.Map(location=(center_lat, center_lon), zoom_start=13, control_scale=True)

    # Heatmap layer (weighted by risk)
    heat_data = [
        [row["latitude"], row["longitude"], _risk_weight(str(row["risk_level"]))]
        for _, row in df.iterrows()
        if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]) and str(row["risk_level"]) in RISK_COLOR
    ]
    if heat_data:
        HeatMap(heat_data, radius=12, blur=18, min_opacity=0.3).add_to(m)

    # Color-coded markers
    for _, row in df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isnull(lat) or pd.isnull(lon):
            continue
        risk = str(row.get("risk_level", "Low"))
        color = RISK_COLOR.get(risk, "#3498db")
        popup = folium.Popup(
            html=(
                f"Risk: <b>{risk}</b><br>"
                f"Conf: {float(row.get('confidence', 0.0)):.2f}<br>"
                f"Area(px): {float(row.get('area_px', 0.0)):.0f}<br>"
                f"Frame: {int(row.get('frame_id', 0))}<br>"
                f"Time: {row.get('detection_time', '')}"
            ),
            max_width=250,
        )
        folium.CircleMarker(
            location=(float(lat), float(lon)),
            radius=6 if risk == "Low" else 7 if risk == "Medium" else 8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
        ).add_to(m)

    return m

