from typing import List, Tuple, Iterable
import math
import random
from datetime import datetime

import gpxpy
import gpxpy.gpx


def meters_to_degrees_lat(meters: float) -> float:
    return meters / 111_320.0


def meters_to_degrees_lon(meters: float, latitude_deg: float) -> float:
    # Adjust for Earth's curvature at given latitude
    return meters / (111_320.0 * math.cos(math.radians(latitude_deg)) or 1e-6)


def generate_gps_track(
    num_points: int,
    base_latitude: float = 28.6139,
    base_longitude: float = 77.2090,
    step_meters_mean: float = 2.0,
    step_meters_stddev: float = 0.5,
    bearing_degrees: float = 0.0,
    jitter_meters: float = 0.5,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """Generate a simple forward-moving GPS track with slight jitter.

    The path advances roughly along the provided bearing with random step sizes
    around the mean, plus small lateral jitter to emulate real motion.
    """
    random.seed(seed)
    lat = base_latitude
    lon = base_longitude
    points: List[Tuple[float, float]] = []

    bearing_rad = math.radians(bearing_degrees)
    forward_unit_lat = math.cos(bearing_rad)
    forward_unit_lon = math.sin(bearing_rad)

    for _ in range(max(1, num_points)):
        step = max(0.2, random.gauss(step_meters_mean, step_meters_stddev))
        jitter_lat_m = random.uniform(-jitter_meters, jitter_meters)
        jitter_lon_m = random.uniform(-jitter_meters, jitter_meters)

        # Forward movement components
        step_lat_m = step * forward_unit_lat
        step_lon_m = step * forward_unit_lon

        # Convert to degrees
        lat += meters_to_degrees_lat(step_lat_m + jitter_lat_m)
        lon += meters_to_degrees_lon(step_lon_m + jitter_lon_m, lat)

        points.append((lat, lon))

    return points


def parse_gpx_file(gpx_bytes: bytes) -> Tuple[List[Tuple[float, float]], List[datetime]]:
    """Parse a GPX file and return list of (lat, lon) and timestamps.

    Returns empty lists if no track points are found.
    """
    points: List[Tuple[float, float]] = []
    times: List[datetime] = []
    try:
        gpx = gpxpy.parse(gpx_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return points, times
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append((float(p.latitude), float(p.longitude)))
                times.append(p.time if p.time is not None else datetime.utcnow())
    if not points:
        for rte in gpx.routes:
            for p in rte.points:
                points.append((float(p.latitude), float(p.longitude)))
                times.append(datetime.utcnow())
    return points, times


def resample_track_to_length(points: List[Tuple[float, float]], target_len: int) -> List[Tuple[float, float]]:
    """Resample or repeat a GPS track to exactly target_len points.

    If points < target_len, repeats last coordinate. If points > target_len,
    down-samples uniformly.
    """
    if target_len <= 0:
        return []
    if not points:
        return [(float('nan'), float('nan'))] * target_len
    n = len(points)
    if n == target_len:
        return list(points)
    if n > target_len:
        # Uniformly pick indices
        indices = [int(i * (n - 1) / (target_len - 1)) for i in range(target_len)]
        return [points[i] for i in indices]
    # n < target_len: pad with last
    out = list(points)
    out.extend([points[-1]] * (target_len - n))
    return out

