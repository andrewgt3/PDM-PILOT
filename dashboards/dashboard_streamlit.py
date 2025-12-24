import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import altair as alt
import time
import subprocess
import sys
import socket
from fpdf import FPDF
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie # <--- NEW IMPORT
import json
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
DB_CONNECTION = os.getenv('DATABASE_URL')
if not DB_CONNECTION:
    raise ValueError("DATABASE_URL environment variable is required")
ST_PAGE_TITLE = "Gaia | Enterprise Command"
REFRESH_RATE_SEC = 1

# --- SETUP PAGE ---
st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide", page_icon=None)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    canvas {border-radius: 4px;} 
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    /* Centered Launch Button Styling */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        font-weight: 600;
        background-color: #00ADB5; /* Electric Teal match */
        border: none;
    }
    div.stButton > button:hover {
         background-color: #00c4ce;
    }
</style>
""", unsafe_allow_html=True)

# --- LOTTIE ANIMATION DATA (Neural Network Vector) ---
# Embedded minified JSON for reliable offline loading
lottie_neural_json = json.loads("""{"v":"5.7.5","fr":60,"ip":0,"op":180,"w":500,"h":500,"nm":"Neural Network","ddd":0,"assets":[],"layers":[{"ddd":0,"ind":1,"ty":4,"nm":"Shape Layer 1","sr":1,"ks":{"o":{"a":0,"k":100,"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[250,250,0],"ix":2,"l":2},"a":{"a":0,"k":[0,0,0],"ix":1,"l":2},"s":{"a":0,"k":[100,100,100],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ty":"rc","d":1,"s":{"a":0,"k":[10,10],"ix":2},"p":{"a":0,"k":[0,0],"ix":3},"r":{"a":0,"k":0,"ix":4},"nm":"Rectangle Path 1","mn":"ADBE Vector Shape - Rect","hd":false},{"ty":"fl","c":{"a":0,"k":[0,0.678431391716,0.709803938866,1],"ix":4},"o":{"a":1,"k":[{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":0,"s":[0]},{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":30,"s":[100]},{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":150,"s":[100]},{"t":180,"s":[0]}],"ix":5},"r":1,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":1,"k":[{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":0,"s":[-150,0]},{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":90,"s":[0,0]},{"t":180,"s":[150,0]}],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Rectangle 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false},{"ty":"gr","it":[{"ty":"el","s":{"a":1,"k":[{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":0,"s":[0,0]},{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":45,"s":[20,20]},{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":135,"s":[20,20]},{"t":180,"s":[0,0]}],"ix":2},"p":{"a":0,"k":[0,0],"ix":3},"nm":"Ellipse Path 1","mn":"ADBE Vector Shape - Ellipse","hd":false},{"ty":"st","c":{"a":0,"k":[0,0.678431391716,0.709803938866,1],"ix":3},"o":{"a":0,"k":100,"ix":4},"w":{"a":0,"k":2,"ix":5},"lc":1,"lj":1,"ml":4,"bm":0,"nm":"Stroke 1","mn":"ADBE Vector Graphic - Stroke","hd":false},{"ty":"tr","p":{"a":1,"k":[{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":0,"s":[-150,0]},{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":90,"s":[0,0]},{"t":180,"s":[150,0]}],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":2,"mn":"ADBE Vector Group","hd":false},{"ty":"gr","it":[{"ty":"el","s":{"a":1,"k":[{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":20,"s":[0,0]},{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":65,"s":[15,15]},{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":115,"s":[15,15]},{"t":160,"s":[0,0]}],"ix":2},"p":{"a":0,"k":[0,0],"ix":3},"nm":"Ellipse Path 1","mn":"ADBE Vector Shape - Ellipse","hd":false},{"ty":"st","c":{"a":0,"k":[0,0.678431391716,0.709803938866,1],"ix":3},"o":{"a":0,"k":100,"ix":4},"w":{"a":0,"k":2,"ix":5},"lc":1,"lj":1,"ml":4,"bm":0,"nm":"Stroke 1","mn":"ADBE Vector Graphic - Stroke","hd":false},{"ty":"tr","p":{"a":1,"k":[{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":20,"s":[-100,50]},{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":90,"s":[0,0]},{"t":160,"s":[100,-50]}],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 2","np":3,"cix":2,"bm":0,"ix":3,"mn":"ADBE Vector Group","hd":false},{"ty":"gr","it":[{"ty":"el","s":{"a":1,"k":[{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":20,"s":[0,0]},{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":65,"s":[15,15]},{"i":{"x":[0.6,0.6],"y":[1,1]},"o":{"x":[0.4,0.4],"y":[0,0]},"t":115,"s":[15,15]},{"t":160,"s":[0,0]}],"ix":2},"p":{"a":0,"k":[0,0],"ix":3},"nm":"Ellipse Path 1","mn":"ADBE Vector Shape - Ellipse","hd":false},{"ty":"st","c":{"a":0,"k":[0,0.678431391716,0.709803938866,1],"ix":3},"o":{"a":0,"k":100,"ix":4},"w":{"a":0,"k":2,"ix":5},"lc":1,"lj":1,"ml":4,"bm":0,"nm":"Stroke 1","mn":"ADBE Vector Graphic - Stroke","hd":false},{"ty":"tr","p":{"a":1,"k":[{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":20,"s":[-100,-50]},{"i":{"x":0.6,"y":1},"o":{"x":0.4,"y":0},"t":90,"s":[0,0]},{"t":160,"s":[100,50]}],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 3","np":3,"cix":2,"bm":0,"ix":4,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":180,"st":0,"bm":0}],"markers":[]}""")

# --- BACKEND HELPERS ---
@st.cache_resource
def get_engine():
    return create_engine(DB_CONNECTION)

def is_port_open(port):
    """Checks if the Physics Engine (Server) is already running"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

@st.cache_data(ttl=REFRESH_RATE_SEC)
def load_data(asset_id=None):
    engine = get_engine()
    query_fleet = """
    SELECT DISTINCT ON (asset_id) 
        asset_id, timestamp, vibration_x, motor_temp_c, rul_hours
    FROM sensors WHERE timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY asset_id, timestamp DESC
    """
    query_history = ""
    query_events = ""
    if asset_id:
        query_history = f"""
        SELECT timestamp, vibration_x, joint_1_torque, motor_temp_c 
        FROM sensors WHERE asset_id = '{asset_id}' 
        AND timestamp > NOW() - INTERVAL '20 minutes' ORDER BY timestamp DESC 
        """
        query_events = """
        SELECT timestamp, event_type FROM events 
        WHERE timestamp > NOW() - INTERVAL '60 minutes' ORDER BY timestamp DESC
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

# --- VIEW 1: FLEET OVERVIEW ---
def render_fleet_view(df_fleet):
    st.title("Fleet Command Center")
    st.caption("Live Telemetry | Zone A | 4 Active Assets")
    
    # --- ENTERPRISE LAUNCHER LOGIC ---
    if df_fleet.empty:
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            # Create a placeholder for the loading screen
            loading_placeholder = st.empty()
            
            # The Button
            if loading_placeholder.button("INITIATE ENVIRONMENT"):
                # 1. Clear button and show Animation
                loading_placeholder.empty()
                with loading_placeholder.container():
                    st.markdown("<h3 style='text-align: center;'>INITIALIZING GAIA NEURAL CORE</h3>", unsafe_allow_html=True)
                    st_lottie(lottie_neural_json, height=300, key="loader")
                    st.caption("Booting physics engine and establishing OPC UA handshake...")
                
                # 2. Run Background Tasks (Blocking)
                if not is_port_open(4840):
                    subprocess.Popen([sys.executable, "opcua_fleet_server.py"]) 
                    time.sleep(3) 
                subprocess.Popen([sys.executable, "mock_fleet_streamer.py"])
                time.sleep(2)
                
                # 3. Clear animation and refresh
                loading_placeholder.empty()
                st.rerun()
            
            if not df_fleet.empty: return

            st.write("") 
            st.info("System Offline. Core physics engine suspended.")
        return

    # --- SYSTEM STATUS BANNER ---
    critical_assets = df_fleet[df_fleet['rul_hours'] < 48]
    if len(critical_assets) >= 2:
        st.error(f"SYSTEMIC FAILURE DETECTED: {len(critical_assets)} Assets Critical. Risk of Cascade.")
    else:
        st.success("All Systems Nominal")
    
    st.divider()

    # --- THE GRID ---
    cols = st.columns(4)
    for i, row in df_fleet.iterrows():
        asset = row['asset_id']
        rul = row['rul_hours']
        vib = row['vibration_x']
        with cols[i % 4]:
            with st.container():
                st.subheader(f"{asset}")
                st.metric(label="RUL (Hours)", value=f"{int(rul)}h", delta=None)
                st.metric(label="Vibration", value=f"{vib:.2f}g", delta_color="inverse")
                if rul < 24:
                    st.markdown(":red[**CRITICAL**]")
                    btn_type = "primary"
                elif rul < 168:
                    st.markdown(":orange[**WARNING**]")
                    btn_type = "secondary"
                else:
                    st.markdown(":green[**HEALTHY**]")
                    btn_type = "secondary"
                if st.button(f"Analyze", key=f"btn_{asset}", type=btn_type, use_container_width=True):
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

    chart_data = df_history.iloc[::5, :] # Downsample
    base = alt.Chart(chart_data).encode(
        x=alt.X('timestamp', axis=alt.Axis(title='Time', format='%H:%M:%S', labelColor='#888')),
        tooltip=['timestamp', 'vibration_x']
    )
    line = base.mark_line(color='#00ADB5', strokeWidth=2).encode(
        y=alt.Y('vibration_x', title='Vibration (g)'),
    )
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

    with st.sidebar:
        st.header(f"Gaia Agent")
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
            st.download_button("Download Work Order", pdf, "work_order.pdf", "application/pdf", type="primary")
        else:
            st.success("System Nominal")
            st.markdown("**Last Scan:** No anomalies.")

# --- MAIN APP ---
def main():
    if 'selected_asset' not in st.session_state:
        st.session_state['selected_asset'] = None

    try:
        asset = st.session_state['selected_asset']
        df_fleet, df_history, df_events = load_data(asset)
    except Exception as e:
        df_fleet = pd.DataFrame()
        df_history = pd.DataFrame()
        df_events = pd.DataFrame()

    if asset:
        render_detail_view(asset, df_history, df_events)
    else:
        render_fleet_view(df_fleet)
    
    time.sleep(REFRESH_RATE_SEC)
    st.rerun()

if __name__ == "__main__":
    main()
