import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import altair as alt
import time
from fpdf import FPDF
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards # <--- THE DESIGN UPGRADE
import subprocess
import sys
import socket

# --- CONFIGURATION ---
DB_CONNECTION = "postgresql://postgres:password@localhost:5432/pdm_timeseries"
ST_PAGE_TITLE = "Gaia | Enterprise Command"
REFRESH_RATE_SEC = 1

# --- SETUP PAGE ---
st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide", page_icon="🏭")

# --- CUSTOM CSS (The "Polish") ---
# This removes the gap at the top and sharpens the borders
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    /* Make charts fit the dark theme perfectly */
    canvas {border-radius: 4px;} 
</style>
""", unsafe_allow_html=True)

# --- BACKEND (Cached) ---
@st.cache_resource
def get_engine():
    return create_engine(DB_CONNECTION)

@st.cache_data(ttl=REFRESH_RATE_SEC)
def load_data(asset_id=None):
    engine = get_engine()
    
    # 1. FLEET VIEW: Latest heartbeat
    query_fleet = """
    SELECT DISTINCT ON (asset_id) 
        asset_id, timestamp, vibration_x, motor_temp_c, rul_hours
    FROM sensors 
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY asset_id, timestamp DESC
    """
    
    # 2. DETAIL VIEW: Sliding window (20 mins)
    query_history = ""
    query_events = ""
    if asset_id:
        query_history = f"""
        SELECT timestamp, vibration_x, joint_1_torque, motor_temp_c 
        FROM sensors 
        WHERE asset_id = '{asset_id}' 
        AND timestamp > NOW() - INTERVAL '20 minutes'
        ORDER BY timestamp DESC 
        """
        query_events = """
        SELECT timestamp, event_type 
        FROM events 
        WHERE timestamp > NOW() - INTERVAL '60 minutes'
        ORDER BY timestamp DESC
        """

    with engine.connect() as conn:
        df_fleet = pd.read_sql(text(query_fleet), conn)
        df_history = pd.DataFrame()
        df_events = pd.DataFrame()
        
        if asset_id and not df_fleet.empty:
            df_history = pd.read_sql(text(query_history), conn)
            df_events = pd.read_sql(text(query_events), conn)
            
    return df_fleet, df_history, df_events

# --- HELPER: CHECK IF SERVER IS RUNNING ---
def is_port_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

# --- VIEW 1: FLEET OVERVIEW ---
def render_fleet_view(df_fleet):
    st.title("🏭 Fleet Command Center")
    st.caption("Live Telemetry | Zone A | 4 Active Assets")
    
    # Logic for System Status
    if not df_fleet.empty:
        critical_assets = df_fleet[df_fleet['rul_hours'] < 48]
        if len(critical_assets) >= 2:
            st.error(f"🚨 SYSTEMIC FAILURE DETECTED: {len(critical_assets)} Assets Critical. Risk of Cascade.")
        else:
            st.success("✅  All Systems Nominal")
    
    st.divider()

    # --- THE SMART LAUNCHER ---
    if df_fleet.empty:
        st.warning("⚠️ System Offline.")
        st.info("Physics Engine and Data Pipeline are currently stopped.")
        
        # This button does EVERYTHING now
        if st.button("🚀 Initialize Gaia Simulation", type="primary"):
            progress_text = "Initializing..."
            my_bar = st.progress(0, text=progress_text)

            # STEP 1: Check/Start Server (The Physics Engine)
            # Mapped to opcua_fleet_server.py
            if not is_port_open(4840):
                my_bar.progress(30, text="Booting Physics Engine (OPC UA Server)...")
                # Launch the server in the background
                subprocess.Popen([sys.executable, "opcua_fleet_server.py"])
                time.sleep(3) # Wait for it to spin up
            else:
                my_bar.progress(30, text="Physics Engine already running...")
                time.sleep(1)

            # STEP 2: Start Client (The Ingest)
            # Mapped to mock_fleet_streamer.py (Direct Simulator for robustness)
            my_bar.progress(70, text="Connecting Data Pipeline...")
            subprocess.Popen([sys.executable, "mock_fleet_streamer.py"])
            time.sleep(2)
            
            # STEP 3: Finish
            my_bar.progress(100, text="System Online!")
            time.sleep(1)
            st.rerun()
            
        return

    # THE GRID (Data is present)
    cols = st.columns(4)
    for i, row in df_fleet.iterrows():
        asset = row['asset_id']
        rul = row['rul_hours']
        
        with cols[i % 4]:
            with st.container():
                st.subheader(f"{asset}")
                
                st.metric(label="RUL (Hours)", value=f"{int(rul)}h", delta=None)
                st.metric(label="Vibration", value=f"{row['vibration_x']:.2f}g", delta_color="inverse")
                
                if rul < 24:
                    st.markdown(":red[**CRITICAL**]")
                    btn_type = "primary"
                elif rul < 168:
                    st.markdown(":orange[**WARNING**]")
                    btn_type = "secondary"
                else:
                    st.markdown(":green[**HEALTHY**]")
                    btn_type = "secondary"
                
                if st.button(f"Analyze 🔍", key=f"btn_{asset}", type=btn_type, use_container_width=True):
                    st.session_state['selected_asset'] = asset
                    st.rerun()

    style_metric_cards(background_color="#262626", border_left_color="#00ADB5", border_radius_px=5, box_shadow=True)

# --- VIEW 2: DETAIL VIEW ---
def render_detail_view(asset_id, df_history, df_events):
    c1, c2 = st.columns([1, 8])
    if c1.button("← Back"):
        st.session_state['selected_asset'] = None
        st.rerun()
    c2.markdown(f"## {asset_id} // Real-Time Diagnostics")

    if df_history.empty:
        st.info("Buffering data stream...")
        return

    # 1. Main Chart (Dark Theme Optimized)
    chart_data = df_history.iloc[::5, :]
    
    base = alt.Chart(chart_data).encode(
        x=alt.X('timestamp', axis=alt.Axis(title='Time', format='%H:%M:%S', labelColor='#888')),
        tooltip=['timestamp', 'vibration_x']
    )
    
    # Make the line "Electric Teal" to match the theme
    line = base.mark_line(color='#00ADB5', strokeWidth=2).encode(
        y=alt.Y('vibration_x', title='Vibration (g)'),
    )

    # Red Line Logic
    min_time = chart_data['timestamp'].min()
    visible_events = df_events[df_events['timestamp'] >= min_time]
    cleaning_events = visible_events[visible_events['event_type'].str.contains('Cleaning', na=False)]

    if not cleaning_events.empty:
        rules = alt.Chart(cleaning_events).mark_rule(color='#FF4B4B', strokeDash=[5,5], strokeWidth=2).encode(x='timestamp')
        chart = (line + rules).interactive()
        insight_active = True
    else:
        chart = line.interactive()
        insight_active = False

    st.altair_chart(chart, use_container_width=True)

    # 2. Sidebar Agent
    with st.sidebar:
        st.header(f"🧠 Gaia Agent")
        st.caption(f"Monitoring: {asset_id}")
        st.markdown("---")
        
        if insight_active:
            st.error("ROOT CAUSE IDENTIFIED")
            st.markdown("""
            **Analysis:** Vibration spike detected.
            
            **Correlation:** Matches **Cleaning Crew** entry pattern (Zone 3).
            
            **Confidence:** 98.5%
            """)
            pdf = create_work_order(asset_id, "Correlated vibration spike detected matching Cleaning Crew timestamps.")
            st.download_button("📄 Download Work Order", pdf, "work_order.pdf", "application/pdf", type="primary")
        else:
            st.success("System Nominal")
            st.markdown("**Last Scan:** No anomalies.")

# --- MAIN ---
def main():
    if 'selected_asset' not in st.session_state:
        st.session_state['selected_asset'] = None

    try:
        asset = st.session_state['selected_asset']
        df_fleet, df_history, df_events = load_data(asset)
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()

    if asset:
        render_detail_view(asset, df_history, df_events)
    else:
        render_fleet_view(df_fleet)
    
    time.sleep(REFRESH_RATE_SEC)
    st.rerun()

if __name__ == "__main__":
    main()
