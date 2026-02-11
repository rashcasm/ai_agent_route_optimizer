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

# ── Page Configuration ──
st.set_page_config(page_title="Transport Route Optimizer", layout="wide")

# ── Custom CSS for Professional Styling ──
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #333;
    }
    div.stButton > button:first-child {
        background-color: #0068c9;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #004b91;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──
st.title("Transport Route Optimizer")
st.markdown(
    """
    **Automated Data Standardization & Route Optimization Pipeline**

    Upload raw transport logs to standardize inconsistent names, geocode locations,
    impute missing timestamps, and calculate the mathematically optimal route.
    """
)

# ── Sidebar ──
with st.sidebar:
    st.header("Configuration")
    default_speed = st.slider(
        "Average Vehicle Speed (km/h)", min_value=10, max_value=60, value=30, step=5
    )

    st.divider()

    st.subheader("Pipeline Architecture")
    st.markdown(
        """
        1. **Standardization:** LLM normalizes stop names.
        2. **City Detection:** Identifies route locality.
        3. **Geocoding:** Fetches lat/long coordinates.
        4. **Imputation:** Estimates missing timestamps.
        5. **Optimization:** Solves Traveling Salesperson Problem (TSP).
        6. **Visualization:** Renders interactive map segments.
        """
    )

    st.divider()

    if st.button("Reset Application"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# ── Main Content ──
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validation
    if "stop_name" not in df.columns:
        st.error(
            f"Invalid Schema: CSV must contain a 'stop_name' column. Detected columns: {list(df.columns)}"
        )
        st.stop()

    time_col = "arrival_time" if "arrival_time" in df.columns else "time"

    # Data Overview
    st.subheader("Data Overview")
    c1, c2, c3 = st.columns(3)
    total = len(df)
    missing_times = df[time_col].isna().sum() if time_col in df.columns else total

    c1.metric("Total Stops", total)
    c2.metric("Missing Timestamps", missing_times)
    c3.metric("Data Quality Score", f"{int(((total - missing_times) / total) * 100)}%")

    with st.expander("View Raw Input Data"):
        st.dataframe(df, use_container_width=True)

    # ── Action Area ──
    st.divider()

    if st.button("Start Optimization Pipeline", type="primary"):
        # Use a status container for a cleaner log view
        with st.status("Processing Data Pipeline...", expanded=True) as status:
            # STEP 1
            st.write("Step 1: Standardizing stop names via LLM...")
            unique_stops = df["stop_name"].unique().tolist()
            clean_map = llm_standardize_names(unique_stops)
            df["clean_stop_name"] = df["stop_name"].map(clean_map)
            df["clean_stop_name"] = df["clean_stop_name"].fillna(df["stop_name"])

            # STEP 2
            st.write("Step 2: Detecting city context...")
            clean_names_list = df["clean_stop_name"].unique().tolist()
            city = llm_detect_city(clean_names_list)

            # STEP 3
            st.write(f"Step 3: Geocoding locations in {city}...")
            # We define a dummy callback since we are using st.status
            df = batch_geocode(df, city, progress_callback=lambda x: None)
            geocoded = df["lat"].notna().sum()

            # STEP 4
            st.write("Step 4: Imputing missing timestamps...")
            df = fill_missing_times(df)
            filled_times = df[time_col].notna().sum()

            # STEP 5
            st.write("Step 5: calculating optimal route (TSP)...")
            optimized_order, total_distance = optimize_route(df)

            # Build optimized df
            df_optimized = df.loc[optimized_order].reset_index(drop=True)
            df_optimized.index = df_optimized.index + 1
            df_optimized.index.name = "visit_order"

            # STEP 6
            st.write("Step 6: Generating map layers...")
            route_segments = build_route_segments(df, optimized_order)

            status.update(
                label="Pipeline Completed Successfully",
                state="complete",
                expanded=False,
            )

        # Persist results
        st.session_state["result_df"] = df
        st.session_state["df_optimized"] = df_optimized
        st.session_state["optimized_order"] = optimized_order
        st.session_state["total_distance"] = total_distance
        st.session_state["route_segments"] = route_segments
        st.session_state["city"] = city
        st.session_state["time_col"] = time_col

    # ── Results Display ──
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        df_optimized = st.session_state["df_optimized"]
        optimized_order = st.session_state["optimized_order"]
        total_distance = st.session_state["total_distance"]
        route_segments = st.session_state["route_segments"]
        city = st.session_state["city"]
        tc = st.session_state["time_col"]

        st.divider()
        st.subheader("Optimization Results")

        # Use Tabs for a cleaner interface
        tab1, tab2, tab3 = st.tabs(["Map Visualization", "Data Details", "Exports"])

        with tab1:
            st.markdown(f"**Total Optimized Distance:** {total_distance:.2f} km")

            valid_coords = result_df.dropna(subset=["lat", "lon"])
            if len(valid_coords) > 0:
                center_lat = valid_coords["lat"].mean()
                center_lon = valid_coords["lon"].mean()
            else:
                center_lat, center_lon = 20.5937, 78.9629

            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
            )

            # Route Lines
            # 1. Original (Grey Dashed)
            original_pts = []
            for _, row in result_df.iterrows():
                if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                    original_pts.append([float(row["lat"]), float(row["lon"])])
            if len(original_pts) >= 2:
                folium.PolyLine(
                    original_pts,
                    color="gray",
                    weight=2,
                    opacity=0.5,
                    dash_array="5, 10",
                    tooltip="Original Sequence",
                ).add_to(m)

            # 2. Optimized (Blue Solid)
            if route_segments:
                opt_pts = []
                for seg in route_segments:
                    if not opt_pts:
                        opt_pts.append([seg["start_lat"], seg["start_lon"]])
                    opt_pts.append([seg["end_lat"], seg["end_lon"]])
                if len(opt_pts) >= 2:
                    folium.PolyLine(
                        opt_pts,
                        color="#0068c9",
                        weight=4,
                        opacity=0.9,
                        tooltip="Optimized Route",
                    ).add_to(m)

            # Markers
            colors = [
                "green",
                "blue",
                "purple",
                "orange",
                "darkred",
                "lightred",
                "beige",
                "darkblue",
                "darkgreen",
                "cadetblue",
            ]

            for visit_num, orig_idx in enumerate(optimized_order):
                row = result_df.loc[orig_idx]
                if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                    # Logic for markers
                    is_start = visit_num == 0
                    is_end = visit_num == len(optimized_order) - 1

                    if is_start:
                        icon_color = "green"
                        icon_type = "play"
                    elif is_end:
                        icon_color = "red"
                        icon_type = "stop"
                    else:
                        icon_color = "blue"
                        icon_type = "info-sign"

                    time_val = row.get(tc, "N/A")
                    popup_txt = f"""
                    <div style='font-family:sans-serif; width:150px;'>
                        <b>Stop {visit_num + 1}</b><br>
                        {row["clean_stop_name"]}<br>
                        Time: {time_val}
                    </div>
                    """

                    folium.Marker(
                        location=[float(row["lat"]), float(row["lon"])],
                        popup=folium.Popup(popup_txt, max_width=200),
                        tooltip=f"{visit_num + 1}. {row['clean_stop_name']}",
                        icon=folium.Icon(color=icon_color, icon=icon_type),
                    ).add_to(m)

            st_folium(m, width=None, height=600, use_container_width=True)

            st.caption(
                "Legend: Green = Start, Red = End, Blue Line = Optimized Path, Grey Dashed = Original Input Order"
            )

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Cleaned Data (Input Order)")
                st.dataframe(
                    result_df[["clean_stop_name", tc, "lat", "lon"]],
                    use_container_width=True,
                )
            with col_b:
                st.markdown("#### Optimized Schedule")
                st.dataframe(
                    df_optimized[["clean_stop_name", tc, "lat", "lon"]],
                    use_container_width=True,
                )

        with tab3:
            st.markdown("#### Download Reports")
            csv_data = df_optimized.to_csv(index=True)
            st.download_button(
                label="Download Optimized Route CSV",
                data=csv_data,
                file_name="optimized_route.csv",
                mime="text/csv",
            )
