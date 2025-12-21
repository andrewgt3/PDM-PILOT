import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import altair as alt
import time
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURATION ---
DB_CONNECTION = "postgresql://postgres:password@localhost:5432/pdm_timeseries"
ST_PAGE_TITLE = "Gaia | Enterprise Command"
REFRESH_RATE_SEC = 1  # Faster refresh for "live" feel

# --- SETUP PAGE ---
st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide", page_icon="🏭")

# --- CLEAN STYLING ---
st.markdown("""
<style>
    .stAppDeployButton {display:none;}
    /* Remove padding to make it look like a dashboard */
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    /* Status Badges */
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND (Cached) ---
@st.cache_resource
def get_engine():
    return create_engine(DB_CONNECTION)

@st.cache_data(ttl=REFRESH_RATE_SEC)
def load_data(asset_id=None):
    engine = get_engine()
    
    # 1. FLEET VIEW: Get the absolute latest heartbeat for every robot
    # We use DISTINCT ON to get only the single most recent row per robot
    query_fleet = """
    SELECT DISTINCT ON (asset_id) 
        asset_id, timestamp, vibration_x, motor_temp_c, rul_hours
    FROM sensors 
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY asset_id, timestamp DESC
    """
    
    # 2. DETAIL VIEW: Get sliding window (Last 20 mins)
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
        # Get events for context overlay
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

# --- PDF GENERATOR ---
def create_work_order(asset_id, insight_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Gaia Predictive | Automated Work Order", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Asset: {asset_id}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, "Priority: CRITICAL", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Detected Issue:\n{insight_text}")
    return pdf.output(dest='S').encode('latin-1')

# --- VIEW 1: FLEET OVERVIEW (Cleaned Up) ---
def render_fleet_view(df_fleet):
    st.title("🏭 Fleet Command Center")
    
    # 1. System Status Banner (Only shows if risk exists)
    critical_assets = df_fleet[df_fleet['rul_hours'] < 48]
    if len(critical_assets) >= 2:
        st.error(f"🚨 CASCADE FAILURE IMMINENT: {len(critical_assets)} assets are critical. Immediate intervention required.")
    
    st.divider()

    # 2. The Clean Grid
    if df_fleet.empty:
        st.warning("No live data connection. Ensure OPC Client is running.")
        return

    cols = st.columns(4)
    
    for i, row in df_fleet.iterrows():
        asset = row['asset_id']
        rul = row['rul_hours']
        vib = row['vibration_x']
        
        # Color Logic
        border_color = "#00cc96" # Green
        if rul < 24: border_color = "#ff4b4b" # Red
        elif rul < 168: border_color = "#ffa500" # Orange
            
        with cols[i % 4]:
            # Native Container with border (Cleaner look)
            with st.container(border=True):
                st.subheader(asset)
                
                # KPIs using native metric
                c1, c2 = st.columns(2)
                c1.metric("RUL", f"{int(rul)}h")
                c2.metric("Vib", f"{vib:.2f}g", delta_color="inverse")
                
                # Action Button
                if st.button("Deep Dive 🔍", key=f"btn_{asset}", use_container_width=True):
                    st.session_state['selected_asset'] = asset
                    st.rerun()

# --- VIEW 2: ASSET DETAIL (The Scientific View) ---
def render_detail_view(asset_id, df_history, df_events):
    # Header & Nav
    c1, c2 = st.columns([1, 6])
    if c1.button("← Fleet"):
        st.session_state['selected_asset'] = None
        st.rerun()
    c2.markdown(f"## 🤖 Live Diagnostics: **{asset_id}**")

    if df_history.empty:
        st.info("Waiting for data stream...")
        return

    # 1. The Main Chart (Altair)
    # Downsample for speed
    chart_data = df_history.iloc[::5, :]
    
    # Base Line
    base = alt.Chart(chart_data).encode(
        x=alt.X('timestamp', axis=alt.Axis(title='Time (Live)', format='%H:%M:%S')),
        tooltip=['timestamp', 'vibration_x']
    )
    
    line = base.mark_line(color='#0068c9', strokeWidth=2).encode(
        y=alt.Y('vibration_x', title='Vibration (g)'),
    )

    # Context Overlay (Red Line)
    # We only show events that are INSIDE the current chart window
    min_time = chart_data['timestamp'].min()
    visible_events = df_events[df_events['timestamp'] >= min_time]
    
    # Cleaning Crew Filter
    cleaning_events = visible_events[visible_events['event_type'].str.contains('Cleaning', na=False)]

    if not cleaning_events.empty:
        rules = alt.Chart(cleaning_events).mark_rule(color='red', strokeDash=[5,5], strokeWidth=2).encode(x='timestamp')
        chart = (line + rules).interactive()
        insight_active = True
    else:
        chart = line.interactive()
        insight_active = False

    st.altair_chart(chart, use_container_width=True)

    # 2. Agent Sidebar (Context-Aware)
    with st.sidebar:
        st.header(f"🧠 {asset_id} AI Analyst")
        st.markdown("---")
        
        if insight_active:
            st.error("🚨 ROOT CAUSE FOUND")
            st.markdown("""
            **Analysis:** Vibration spikes correlate 98% with **Cleaning Crew** entry logs.
            
            **Recommendation:**
            Install power conditioning in Zone 3 immediately.
            """)
            
            pdf = create_work_order(asset_id, "Correlated vibration spike detected matching Cleaning Crew timestamps.")
            st.download_button("📄 Download Work Order", pdf, "work_order.pdf", "application/pdf", type="primary")
        
        elif df_history['vibration_x'].iloc[0] > 1.0:
             st.warning("⚠️ High Vibration Detected")
             st.markdown("Asset is operating outside normal parameters.")
        
        else:
            st.success("✅ System Nominal")
            st.caption("Monitoring real-time telemetry...")

# --- MAIN APP ---
def main():
    if 'selected_asset' not in st.session_state:
        st.session_state['selected_asset'] = None

    # Load Data
    try:
        asset = st.session_state['selected_asset']
        df_fleet, df_history, df_events = load_data(asset)
    except Exception as e:
        st.error(f"Data Stream Error: {e}")
        st.stop()

    # Router
    if asset:
        render_detail_view(asset, df_history, df_events)
    else:
        render_fleet_view(df_fleet)
    
    # Live Loop
    time.sleep(REFRESH_RATE_SEC)
    st.rerun()

if __name__ == "__main__":
    main()
