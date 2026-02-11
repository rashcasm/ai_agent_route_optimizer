import json
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from openai import OpenAI

load_dotenv()
# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

LLM_MODEL = "google/gemini-2.0-flash-lite-001"


def _llm_call(prompt, retries=3):
    """Safe LLM wrapper — always returns a dict (JSON mode)."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception as e:
            print(f"  ⚠️ LLM attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return {}


# ─────────────────────────────────────────────
# 1. DETECT CITY
# ─────────────────────────────────────────────
def llm_detect_city(stop_names):
    """Ask LLM which Indian city these stops belong to."""
    prompt = f"""
You are a geography expert for Indian cities.
Given these bus/metro stop names, identify which single Indian city they belong to.
Return ONLY JSON: {{"city": "CityName"}}

Stop names: {stop_names}
"""
    result = _llm_call(prompt)
    return result.get("city", "India") if isinstance(result, dict) else "India"


# ─────────────────────────────────────────────
# 2. STANDARDIZE NAMES
# ─────────────────────────────────────────────
def llm_standardize_names(messy_names_list):
    """LLM cleans up messy stop names."""
    prompt = f"""
You are a Transport Data Expert. Standardize these messy bus stop names.

Rules:
1. Fix typos and abbreviations: "Stn" -> "Station", "Sq" -> "Square",
   "Ngr" -> "Nagar", "Rd" -> "Road", "Intl" -> "International",
   "Arprt" -> "Airport", "Opp." -> "Opposite"
2. Remove noise like "(Gate 1)", "(Platform 1)" unless it makes the stop distinct.
3. Keep names recognizable — do NOT invent new names.
4. Return ONLY a JSON object mapping original -> clean name.

Input: {messy_names_list}

Output format:
{{
    "original_name_1": "Clean Name 1",
    "original_name_2": "Clean Name 2"
}}
"""
    result = _llm_call(prompt)
    if not result:
        return {name: name for name in messy_names_list}
    return result


# ─────────────────────────────────────────────
# 3. NOMINATIM GEOCODER — 2 real attempts per stop
# ─────────────────────────────────────────────
geolocator = Nominatim(user_agent="transport_agent_robust_v4", timeout=10)


def get_lat_long(stop_name, city="India"):
    """
    Try Nominatim with multiple query strategies. Two real passes:
      Pass A: exact name + city
      Pass B: cleaned/simplified variants
    Returns (lat, lon) or (None, None).
    """
    # Build a list of search queries to try
    queries = []

    # 1. Full name + city
    queries.append(f"{stop_name}, {city}, India")

    # 2. Strip parenthetical noise
    clean = re.sub(r"\s*\(.*?\)", "", stop_name).strip()
    if clean != stop_name:
        queries.append(f"{clean}, {city}, India")

    # 3. Common Indian word swaps
    swaps = {
        "Chowk": "Square",
        "Square": "Chowk",
        "Nagar": "Colony",
        "Colony": "Nagar",
        "Station": "",
        "Road": "Rd",
    }
    for old, new in swaps.items():
        if old in clean:
            queries.append(f"{clean.replace(old, new).strip()}, {city}, India")
            break

    # 4. First 1-2 significant words only
    words = clean.split()
    if len(words) >= 2:
        queries.append(f"{' '.join(words[:2])}, {city}, India")
    if len(words) >= 1:
        queries.append(f"{words[0]}, {city}, India")

    # De-duplicate preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    # Try each query (this gives ~2-5 real attempts)
    for query in unique:
        try:
            print(f"  📍 Nominatim: '{query}'")
            location = geolocator.geocode(query)
            time.sleep(1.2)  # respect rate limit
            if location:
                lat, lon = float(location.latitude), float(location.longitude)
                # Sanity: must be in India
                if 6.0 <= lat <= 37.0 and 67.0 <= lon <= 98.0:
                    print(f"  ✅ Found: {lat:.4f}, {lon:.4f}")
                    return lat, lon
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            time.sleep(2)

    print(f"  ❌ Nominatim failed: {stop_name}")
    return None, None


# ─────────────────────────────────────────────
# 4. LLM GEOCODING FALLBACK — for stops Nominatim couldn't find
# ─────────────────────────────────────────────
def llm_geocode_all(stop_names, city):
    """
    Ask the LLM for lat/lon of a list of stops.
    Used as FALLBACK after Nominatim fails.
    """
    prompt = f"""
You are a geography expert with precise knowledge of {city}, India.
I need the latitude and longitude for each of these locations in {city}.

Be as ACCURATE as possible. These are real bus stops / landmarks in {city}.
Use your knowledge of the city. Every location MUST get coordinates —
if you're not 100% sure, give your best estimate for that area of the city.

Locations: {stop_names}

Return ONLY a JSON object. Every location must be a key, value must have "lat" and "lon" as numbers.
Example:
{{
    "Some Place": {{"lat": 21.1458, "lon": 79.0882}},
    "Another Place": {{"lat": 21.1500, "lon": 79.0900}}
}}

IMPORTANT: You MUST return coordinates for EVERY single location. No nulls. No skipping.
"""
    result = _llm_call(prompt)
    return result


def batch_geocode(df, city, progress_callback=None):
    """
    3-pass geocoding:
      Pass 1: Nominatim (real geocoder, 2+ attempts per stop)
      Pass 2: LLM fallback for anything Nominatim missed
      Pass 3: Interpolate any remaining gaps from neighbors
    """
    df = df.copy()
    df["lat"] = None
    df["lon"] = None

    # ── Pass 1: Nominatim for every stop ──
    for index, row in df.iterrows():
        name = row["clean_stop_name"]
        if pd.isna(name) or name == "":
            name = row["stop_name"]
        if progress_callback:
            progress_callback(f"🌍 Nominatim geocoding: {name}...")
        lat, lon = get_lat_long(name, city)
        df.at[index, "lat"] = lat
        df.at[index, "lon"] = lon

    # ── Pass 2: LLM fallback for failures ──
    missing_mask = df["lat"].isna()
    missing_names = df.loc[missing_mask, "clean_stop_name"].tolist()

    if missing_names:
        if progress_callback:
            progress_callback(
                f"🤖 Nominatim missed {len(missing_names)} stops — asking LLM..."
            )
        llm_coords = llm_geocode_all(missing_names, city)

        if llm_coords:
            for index, row in df[missing_mask].iterrows():
                name = row["clean_stop_name"]
                if name in llm_coords and isinstance(llm_coords[name], dict):
                    lat = llm_coords[name].get("lat")
                    lon = llm_coords[name].get("lon")
                    if lat is not None and lon is not None:
                        df.at[index, "lat"] = float(lat)
                        df.at[index, "lon"] = float(lon)
                        print(f"  🤖 LLM provided: {name} → {lat}, {lon}")

    # ── Pass 3: Interpolate any remaining from neighbors ──
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = _interpolate_coords(df)

    return df


def _interpolate_coords(df):
    """Linearly interpolate missing coords from neighbors."""
    df = df.copy()
    n = len(df)

    for i in range(n):
        if pd.isna(df.loc[i, "lat"]):
            # Find previous known
            prev_idx = None
            for j in range(i - 1, -1, -1):
                if pd.notna(df.loc[j, "lat"]):
                    prev_idx = j
                    break

            # Find next known
            next_idx = None
            for j in range(i + 1, n):
                if pd.notna(df.loc[j, "lat"]):
                    next_idx = j
                    break

            if prev_idx is not None and next_idx is not None:
                frac = (i - prev_idx) / (next_idx - prev_idx)
                df.loc[i, "lat"] = df.loc[prev_idx, "lat"] + frac * (
                    df.loc[next_idx, "lat"] - df.loc[prev_idx, "lat"]
                )
                df.loc[i, "lon"] = df.loc[prev_idx, "lon"] + frac * (
                    df.loc[next_idx, "lon"] - df.loc[prev_idx, "lon"]
                )
            elif prev_idx is not None:
                df.loc[i, "lat"] = df.loc[prev_idx, "lat"] + 0.002 * (i - prev_idx)
                df.loc[i, "lon"] = df.loc[prev_idx, "lon"] + 0.002 * (i - prev_idx)
            elif next_idx is not None:
                df.loc[i, "lat"] = df.loc[next_idx, "lat"] - 0.002 * (next_idx - i)
                df.loc[i, "lon"] = df.loc[next_idx, "lon"] - 0.002 * (next_idx - i)

    return df


# ─────────────────────────────────────────────
# 4. SIMPLE TIME FILL — Just average between known times
# ─────────────────────────────────────────────
def fill_missing_times(df):
    """
    Dead simple: find known times (anchors), evenly space the gaps between them.
    No coordinates, no distance, no speed calculations. Just linear interpolation.
    """
    time_col = "arrival_time" if "arrival_time" in df.columns else "time"
    df = df.copy()

    # Parse to datetime
    df[time_col] = pd.to_datetime(df[time_col], format="%H:%M", errors="coerce")

    n = len(df)

    # Edge case: no known times at all → generate 08:00 + 5min each
    if df[time_col].notna().sum() == 0:
        base = pd.Timestamp("2024-01-01 08:00:00")
        for i in range(n):
            df.loc[i, time_col] = base + pd.Timedelta(minutes=i * 5)
        df[time_col] = df[time_col].dt.strftime("%H:%M")
        return df

    # Edge case: only 1 known time → space others 5 min apart from it
    if df[time_col].notna().sum() == 1:
        anchor_idx = df[time_col].first_valid_index()
        anchor_time = df.loc[anchor_idx, time_col]
        for i in range(n):
            if pd.isna(df.loc[i, time_col]):
                offset = (i - anchor_idx) * 5  # 5 min per stop
                df.loc[i, time_col] = anchor_time + pd.Timedelta(minutes=offset)
        df[time_col] = df[time_col].dt.strftime("%H:%M")
        return df

    # ── Main logic: fill between consecutive anchors ──
    anchors = df[df[time_col].notna()].index.tolist()

    # Fill gaps BEFORE first anchor
    if anchors[0] > 0:
        # Estimate per-stop gap from first two anchors
        if len(anchors) >= 2:
            span = (
                df.loc[anchors[1], time_col] - df.loc[anchors[0], time_col]
            ).total_seconds() / 60.0
            per_stop = span / (anchors[1] - anchors[0])
        else:
            per_stop = 5.0
        per_stop = max(per_stop, 2.0)  # at least 2 min

        for i in range(anchors[0] - 1, -1, -1):
            df.loc[i, time_col] = df.loc[i + 1, time_col] - pd.Timedelta(
                minutes=per_stop
            )

    # Fill gaps BETWEEN anchors
    for a_pos in range(len(anchors) - 1):
        a_idx = anchors[a_pos]
        b_idx = anchors[a_pos + 1]
        gap_indices = list(range(a_idx + 1, b_idx))

        if not gap_indices:
            continue

        time_a = df.loc[a_idx, time_col]
        time_b = df.loc[b_idx, time_col]
        total_minutes = (time_b - time_a).total_seconds() / 60.0
        total_steps = b_idx - a_idx

        for step, gap_i in enumerate(gap_indices, start=1):
            frac = step / total_steps
            df.loc[gap_i, time_col] = time_a + pd.Timedelta(
                minutes=frac * total_minutes
            )

    # Fill gaps AFTER last anchor
    if anchors[-1] < n - 1:
        if len(anchors) >= 2:
            span = (
                df.loc[anchors[-1], time_col] - df.loc[anchors[-2], time_col]
            ).total_seconds() / 60.0
            per_stop = span / (anchors[-1] - anchors[-2])
        else:
            per_stop = 5.0
        per_stop = max(per_stop, 2.0)

        for i in range(anchors[-1] + 1, n):
            df.loc[i, time_col] = df.loc[i - 1, time_col] + pd.Timedelta(
                minutes=per_stop
            )

    # Convert back to string
    df[time_col] = df[time_col].dt.strftime("%H:%M")
    return df


# ─────────────────────────────────────────────
# 5. ROUTE OPTIMIZATION — Nearest Neighbor + 2-opt
# ─────────────────────────────────────────────
def optimize_route(df):
    """
    Nearest-neighbor TSP with 2-opt improvement.
    Returns (ordered list of df indices, total distance in km).
    """
    from geopy.distance import geodesic

    coords = df[["lat", "lon"]].dropna()
    if len(coords) < 2:
        return list(df.index), 0.0

    valid_indices = coords.index.tolist()
    n = len(valid_indices)

    # Distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic(
                (
                    coords.loc[valid_indices[i], "lat"],
                    coords.loc[valid_indices[i], "lon"],
                ),
                (
                    coords.loc[valid_indices[j], "lat"],
                    coords.loc[valid_indices[j], "lon"],
                ),
            ).km
            dist[i][j] = d
            dist[j][i] = d

    # Nearest neighbor from every start
    best_tour = None
    best_cost = float("inf")

    for start in range(n):
        tour = [start]
        remaining = set(range(n)) - {start}
        while remaining:
            curr = tour[-1]
            nxt = min(remaining, key=lambda x: dist[curr][x])
            tour.append(nxt)
            remaining.remove(nxt)
        cost = sum(dist[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]

    # 2-opt improvement
    if best_tour and n > 3:
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Current cost of edges (i-1,i) and (j, j+1 or end)
                    end_j = j + 1 if j + 1 < n else j
                    old = (
                        dist[best_tour[i - 1]][best_tour[i]]
                        + dist[best_tour[j]][best_tour[end_j]]
                    )
                    new = (
                        dist[best_tour[i - 1]][best_tour[j]]
                        + dist[best_tour[i]][best_tour[end_j]]
                    )
                    if new < old - 1e-10:
                        best_tour[i : j + 1] = reversed(best_tour[i : j + 1])
                        best_cost = sum(
                            dist[best_tour[k]][best_tour[k + 1]] for k in range(n - 1)
                        )
                        improved = True

    # Map back to df indices
    order = [valid_indices[i] for i in best_tour]

    # Append any stops with no coords at the end
    all_idx = set(df.index.tolist())
    missing = sorted(all_idx - set(order))
    order.extend(missing)

    return order, best_cost


def build_route_segments(df, order):
    """Build line segment dicts for map drawing."""
    segments = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if pd.notna(df.loc[a, "lat"]) and pd.notna(df.loc[b, "lat"]):
            segments.append(
                {
                    "start_lat": float(df.loc[a, "lat"]),
                    "start_lon": float(df.loc[a, "lon"]),
                    "end_lat": float(df.loc[b, "lat"]),
                    "end_lon": float(df.loc[b, "lon"]),
                    "from_stop": df.loc[a, "clean_stop_name"],
                    "to_stop": df.loc[b, "clean_stop_name"],
                }
            )
    return segments
