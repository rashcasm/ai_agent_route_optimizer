import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from agent import (
    batch_geocode,
    build_route_segments,
    fill_missing_times,
    llm_detect_city,
    llm_standardize_names,
    optimize_route,
)

st.set_page_config(page_title="AI Transport Agent", layout="wide")

st.title("🤖 Intelligent Route Standardizer & Optimizer")
st.markdown(
    "Upload messy transport CSV → AI cleans names, geocodes every stop, "
    "fills missing times, and finds the **shortest route**."
)

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    default_speed = st.slider(
        "Assumed avg speed (km/h)", min_value=10, max_value=60, value=30, step=5
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline steps:**\n"
        "1. 🧠 LLM standardizes stop names\n"
        "2. 🏙️ LLM detects city\n"
        "3. 🌍 Nominatim geocodes (2 real tries)\n"
        "4. 🤖 LLM fallback for missed stops\n"
        "5. 📐 Interpolate any coord gaps\n"
        "6. ⏱️ Average-based time estimation\n"
        "7. 🗺️ TSP route optimization\n"
        "8. 📊 Interactive map with route lines"
    )
    st.markdown("---")
    if st.button("🗑️ Clear Results"):
        for key in [
            "result_df",
            "optimized_order",
            "total_distance",
            "route_segments",
            "city",
            "df_optimized",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

# ── File Upload ──
uploaded_file = st.file_uploader("📂 Upload Route CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate
    if "stop_name" not in df.columns:
        st.error(f"CSV must have a `stop_name` column. Found: {list(df.columns)}")
        st.stop()

    # Determine time column
    time_col = "arrival_time" if "arrival_time" in df.columns else "time"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ Raw Messy Data")
        st.dataframe(df, width="stretch")

    # Stats
    total = len(df)
    missing_times = df[time_col].isna().sum() if time_col in df.columns else total
    st.info(f"📊 **{total}** stops loaded · **{missing_times}** missing arrival times")

    # ── Run Pipeline ──
    if st.button("🚀 Activate Agent", type="primary"):
        status = st.empty()
        bar = st.progress(0)

        # STEP 1: Standardize names
        status.info("🧠 Step 1/6 — LLM standardizing stop names...")
        unique_stops = df["stop_name"].unique().tolist()
        clean_map = llm_standardize_names(unique_stops)
        df["clean_stop_name"] = df["stop_name"].map(clean_map)
        df["clean_stop_name"] = df["clean_stop_name"].fillna(df["stop_name"])
        bar.progress(10)

        # STEP 2: Detect city
        status.info("🏙️ Step 2/6 — Detecting city...")
        clean_names_list = df["clean_stop_name"].unique().tolist()
        city = llm_detect_city(clean_names_list)
        bar.progress(20)

        # STEP 3: Geocode stops (Nominatim first, then LLM fallback)
        status.info(
            f"🌍 Step 3/6 — Geocoding stops in **{city}** (Nominatim → LLM fallback)..."
        )

        def _progress_cb(msg):
            status.text(msg)

        df = batch_geocode(df, city, progress_callback=_progress_cb)
        bar.progress(50)

        geocoded = df["lat"].notna().sum()

        # STEP 4: Fill missing times
        status.info("⏱️ Step 4/6 — Estimating missing arrival times...")
        df = fill_missing_times(df)
        bar.progress(65)

        filled_times = df[time_col].notna().sum()

        # STEP 5: Route optimization
        status.info("🗺️ Step 5/6 — Computing shortest route (TSP)...")
        optimized_order, total_distance = optimize_route(df)
        bar.progress(80)

        # Build optimized df
        df_optimized = df.loc[optimized_order].reset_index(drop=True)
        df_optimized.index = df_optimized.index + 1
        df_optimized.index.name = "visit_order"

        # STEP 6: Build segments
        status.info("📊 Step 6/6 — Building route map...")
        route_segments = build_route_segments(df, optimized_order)
        bar.progress(100)

        status.success(
            f"✅ Done! Geocoded **{geocoded}/{total}** stops · "
            f"Filled **{filled_times}/{total}** times · "
            f"Optimized route: **{total_distance:.2f} km**"
        )

        # ── Persist everything in session_state ──
        st.session_state["result_df"] = df
        st.session_state["df_optimized"] = df_optimized
        st.session_state["optimized_order"] = optimized_order
        st.session_state["total_distance"] = total_distance
        st.session_state["route_segments"] = route_segments
        st.session_state["city"] = city
        st.session_state["time_col"] = time_col

    # ═══════════════════════════════════════════
    # DISPLAY RESULTS (from session_state so they persist)
    # ═══════════════════════════════════════════
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        df_optimized = st.session_state["df_optimized"]
        optimized_order = st.session_state["optimized_order"]
        total_distance = st.session_state["total_distance"]
        route_segments = st.session_state["route_segments"]
        city = st.session_state["city"]
        tc = st.session_state["time_col"]

        # ── Standardized Data Table ──
        with col2:
            st.subheader("✅ Standardized Data")
            display_cols = [
                c
                for c in ["clean_stop_name", tc, "lat", "lon"]
                if c in result_df.columns
            ]
            st.dataframe(result_df[display_cols], width="stretch")

        # ── Optimized Route Table ──
        st.subheader(f"🗺️ Optimized Route Order — {total_distance:.2f} km total")
        opt_cols = [
            c
            for c in ["clean_stop_name", tc, "lat", "lon"]
            if c in df_optimized.columns
        ]
        st.dataframe(df_optimized[opt_cols], width="stretch")

        # ── Folium Map ──
        st.subheader("📍 Route Visualization")

        valid_coords = result_df.dropna(subset=["lat", "lon"])
        if len(valid_coords) > 0:
            center_lat = valid_coords["lat"].mean()
            center_lon = valid_coords["lon"].mean()
        else:
            center_lat, center_lon = 20.5937, 78.9629

        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
        )

        # Original route — grey dashed
        original_pts = []
        for _, row in result_df.iterrows():
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                original_pts.append([float(row["lat"]), float(row["lon"])])
        if len(original_pts) >= 2:
            folium.PolyLine(
                original_pts,
                color="grey",
                weight=2,
                opacity=0.5,
                dash_array="10",
                tooltip="Original Order",
            ).add_to(m)

        # Optimized route — blue solid
        if route_segments:
            opt_pts = []
            for seg in route_segments:
                if not opt_pts:
                    opt_pts.append([seg["start_lat"], seg["start_lon"]])
                opt_pts.append([seg["end_lat"], seg["end_lon"]])

            if len(opt_pts) >= 2:
                folium.PolyLine(
                    opt_pts,
                    color="#2196F3",
                    weight=4,
                    opacity=0.85,
                    tooltip="Optimized Route",
                ).add_to(m)

        # Stop markers
        colors = [
            "green",
            "blue",
            "purple",
            "orange",
            "darkblue",
            "cadetblue",
            "darkpurple",
            "pink",
            "lightblue",
            "lightgreen",
        ]

        for visit_num, orig_idx in enumerate(optimized_order):
            row = result_df.loc[orig_idx]
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                is_first = visit_num == 0
                is_last = visit_num == len(optimized_order) - 1

                if is_first:
                    icon_color = "green"
                    icon_name = "play"
                elif is_last:
                    icon_color = "red"
                    icon_name = "stop"
                else:
                    icon_color = colors[visit_num % len(colors)]
                    icon_name = "info-sign"

                time_str = row.get(tc, "N/A")
                if pd.isna(time_str):
                    time_str = "N/A"

                popup_html = (
                    f"<b>Stop #{visit_num + 1}</b><br>"
                    f"<b>{row['clean_stop_name']}</b><br>"
                    f"⏱️ {time_str}<br>"
                    f"📍 {float(row['lat']):.4f}, {float(row['lon']):.4f}"
                )

                folium.Marker(
                    location=[float(row["lat"]), float(row["lon"])],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"#{visit_num + 1}: {row['clean_stop_name']}",
                    icon=folium.Icon(
                        color=icon_color, icon=icon_name, prefix="glyphicon"
                    ),
                ).add_to(m)

        st_folium(m, width=None, height=600, use_container_width=True)

        # Legend
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🟢 **Start** · 🔴 **End**")
        with c2:
            st.markdown("━━ **Blue: Optimized Route**")
        with c3:
            st.markdown("╌╌ *Grey: Original Order*")

        # Download
        st.subheader("📥 Download Results")
        csv_data = df_optimized.to_csv(index=True)
        st.download_button(
            label="⬇️ Download Optimized Route CSV",
            data=csv_data,
            file_name="optimized_route.csv",
            mime="text/csv",
        )
