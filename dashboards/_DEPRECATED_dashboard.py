"""
Gaia Predictive - Enterprise Platform (Golden Master)
=====================================================
Release: v6.0 Production
Standards: Native Components, Light Mode, ISO Symbology
Author: PlantAGI Enterprise Team
"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
import time
import subprocess
import sys
import socket
import os
from dotenv import load_dotenv
import streamlit_authenticator as stauth
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import graphviz
from streamlit_option_menu import option_menu

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DB_CONNECTION = os.getenv("DATABASE_URL")
if not DB_CONNECTION:
    raise ValueError("DATABASE_URL environment variable is required. Check your .env file.")
ST_PAGE_TITLE = "Gaia Enterprise"
COOKIE_KEY = "gaia_tier1_secure_key"

# Backend scripts
SERVER_SCRIPT = "opcua_fleet_server.py"
INGEST_SCRIPT = "mock_fleet_streamer.py"

# --- PAGE SETUP ---
st.set_page_config(
    page_title=ST_PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚙️"
)

# --- CSS OVERRIDES (Professional Standard) ---
# Minimal overrides. Rely on Native Components.
st.markdown("""
<style>
    /* 1. Global Font Stack (System Native) */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 2. Navbar Customization (Engineering Blue) */
    .nav-link-selected {
        background-color: #003478 !important; /* Ford/Engineering Blue */
        color: white !important;
        border-radius: 4px !important;
    }
    .nav-link {
        font-size: 14px !important;
        border-radius: 4px !important;
        margin: 0px 5px !important;
    }
    
    /* 3. Hide Streamlit Chrome */
    /* 3. Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 4. Adjust Main Container Padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

</style>
""", unsafe_allow_html=True)

# --- BACKEND & DATABASE ---
@st.cache_resource
def get_engine():
    """Create PostgreSQL engine with connection pooling."""
    engine = create_engine(DB_CONNECTION, pool_pre_ping=True)
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

def is_port_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def load_data(limit=500):
    """Load sensor data from PostgreSQL/TimescaleDB."""
    engine = get_engine()
    try:
        query = f"SELECT * FROM sensors ORDER BY timestamp DESC LIMIT {limit}"
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# --- AUTHENTICATION ---
def setup_auth():
    # Fetch credentials from environment variables
    admin_user = os.getenv('DASHBOARD_ADMIN_USER', 'admin')
    admin_password = os.getenv('DASHBOARD_ADMIN_PASSWORD')
    if not admin_password:
        raise ValueError("DASHBOARD_ADMIN_PASSWORD environment variable is required")
    
    names = ['Plant Manager']
    usernames = [admin_user]
    passwords = [admin_password]
    hashed_passwords = stauth.Hasher.hash_list(passwords)
    
    credentials = {
        'usernames': {
            usernames[0]: {'name': names[0], 'password': hashed_passwords[0]}
        }
    }
    
    authenticator = stauth.Authenticate(
        credentials,
        'gaia_enterprise_cookie',
        COOKIE_KEY,
        cookie_expiry_days=0
    )
    return authenticator

# --- VIEW: FLEET OVERVIEW ---
def view_fleet_overview(df):
    # Native Metric Row
    active = df['asset_id'].nunique() if not df.empty else 0
    eff_val = 100 - (df['vibration_x'].mean() * 10) if not df.empty else 0
    risks = len(df[df['rul_hours'] < 48]) if not df.empty else 0
    
    # Use Container for "Card-like" grouping
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Assets", active)
        c2.metric("Overall OEE", f"{eff_val:.1f}%", "+0.4%")
        c3.metric("Critical Alerts", risks, delta_color="inverse")
        c4.metric("Gateway Latency", "12ms")
        
    st.caption("Last synchronized: " + time.strftime("%H:%M:%S UTC"))
    st.subheader("Asset Registry")

    if df.empty:
        st.info("Initializing telemetry stream...")
        return
        
    # AgGrid Preparation
    latest_df = df.sort_values('timestamp').groupby('asset_id').tail(1).copy()
    
    # Text-based Status (No raw HTML in dataframe, we use JS renderer)
    latest_df['status'] = latest_df['rul_hours'].apply(lambda x: "CRITICAL" if x < 48 else "NOMINAL")
    
    display_df = latest_df[['asset_id', 'status', 'rul_hours', 'vibration_x', 'motor_temp_c']]
    display_df.columns = ['Asset Tag', 'Condition', 'RUL (h)', 'Vibration (g)', 'Temp (°C)']

    # -- JS Renderers --
    status_renderer = JsCode("""
    class StatusRenderer {
        init(params) {
            this.eGui = document.createElement('div');
            this.eGui.style.width = '100%';
            this.eGui.style.height = '100%';
            this.eGui.style.display = 'flex';
            this.eGui.style.alignItems = 'center';
            this.eGui.style.justifyContent = 'center';

            if (params.value === "CRITICAL") {
                this.eGui.innerHTML = '<span style="background-color:#ffc9c9; color:#c92a2a; padding:4px 12px; border-radius:12px; font-weight:600; font-size:12px;">⚠️ CRITICAL</span>';
            } else {
                this.eGui.innerHTML = '<span style="background-color:#b2f2bb; color:#2b8a3e; padding:4px 12px; border-radius:12px; font-weight:600; font-size:12px;">● NOMINAL</span>';
            }
        }
        getGui() { return this.eGui; }
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    
    # Apply Renderer
    gb.configure_column("Condition", cellRenderer=status_renderer)
    gb.configure_column("Asset Tag", pinned='left')
    
    gb.configure_selection('single')
    gridOptions = gb.build()

    AgGrid(
        display_df,
        gridOptions=gridOptions,
        theme='alpine', # Light theme logic
        allow_unsafe_jscode=True, # Required for JS injection
        domLayout='autoHeight',
        fit_columns_on_grid_load=True,
        height=400
    )

# --- VIEW: ASSET DIAGNOSTICS ---
def view_asset_diagnostics(df):
    if df.empty:
        st.warning("No telemetry data available for analysis.")
        return

    # 1. Navigation / Selector (Top)
    c_sel, _ = st.columns([1, 4])
    with c_sel:
        assets = sorted(df['asset_id'].unique())
        selected_asset = st.selectbox("Select Asset", assets)

    # Filter Data
    asset_df = df[df['asset_id'] == selected_asset].sort_values('timestamp')
    latest = asset_df.iloc[-1]
    
    st.title(f"{selected_asset} Analysis")
    
    # 2. Key Metrics (Native)
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        
        is_crit = latest['rul_hours'] < 48
        status_label = "CRITICAL FAILURE" if is_crit else "OPERATIONAL"
        
        m1.metric("Current Status", status_label)
        m2.metric("Predicted RUL", f"{int(latest['rul_hours'])} h")
        m3.metric("Vibration (RMS)", f"{latest['vibration_x']:.3f} g")
        m4.metric("Motor Temp", f"{latest['motor_temp_c']:.1f} °C")

    st.markdown("---")
    
    # 3. Charts (Professional Theme)
    c_chart, c_gauge = st.columns([2, 1])
    
    with c_chart:
        st.subheader("Vibration Trend (RMS)")
        fig = px.line(asset_df, x='timestamp', y='vibration_x')
        fig.update_layout(
            template='plotly_white', 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=10, l=10, r=10),
            height=300
        )
        line_color = '#d62728' if is_crit else '#003478'
        fig.update_traces(line_color=line_color, line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_gauge:
        st.subheader("Anomaly Likelihood")
        # Gauge Chart
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest['vibration_x'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 3.0]},
                'bar': {'color': "#003478"},
                'steps': [
                    {'range': [0, 0.5], 'color': "#f8f9fa"},
                    {'range': [0.5, 3.0], 'color': "#ffebee"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2.0
                }
            }
        ))
        fig_g.update_layout(
            template='plotly_white', 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=300, 
            margin=dict(t=10, b=10, l=30, r=30)
        )
        st.plotly_chart(fig_g, use_container_width=True)

# --- VIEW: SYSTEM TOPOLOGY ---
def view_system_topology(df):
    st.header("Infrastructure Topology")
    st.caption("Live dependency graph (Graphviz)")
    
    with st.container(border=True):
        dot = graphviz.Digraph(comment='System Topology')
        dot.attr(rankdir='LR') # Left to Right
        dot.attr(bgcolor='transparent')
        
        # Professional standard nodes
        dot.node('CLOUD', 'Gaia Cloud Core', shape='component', style='filled', fillcolor='white')
        dot.node('EDGE', 'Edge Gateway', shape='folder', style='filled', fillcolor='#f1f3f5')
        
        if not df.empty:
            for asset in sorted(df['asset_id'].unique()):
                latest_rul = df[df['asset_id'] == asset].iloc[-1]['rul_hours']
                
                # Visual Logic
                if latest_rul < 48:
                    color = '#ffc9c9' # Light Red
                    penwidth = '2'
                    label = f"{asset}\n(CRITICAL)"
                else:
                    color = '#e7f5ff' # Light Blue
                    penwidth = '1'
                    label = asset
                    
                dot.node(asset, label, shape='box', style='filled', fillcolor=color, penwidth=penwidth)
                dot.edge('EDGE', asset)
        
        dot.edge('CLOUD', 'EDGE', label=' TCP/IP')
        st.graphviz_chart(dot, use_container_width=True)

# --- MAIN APP ---
def main():
    authenticator = setup_auth()
    
    # 1. Login Gate (with Container Clearing)
    # We use st.empty() to hold the login widget so we can wipe it away immediately
    login_container = st.empty()
    
    with login_container:
        authenticator.login('main')
    
    if st.session_state["authentication_status"] is not True:
        # If failure or no input, stop here. 
        # The login form is visible in `login_container`.
        st.stop()
    else:
        # If success, clear the login form instantly
        login_container.empty()

    # --- SECURE AREA ---
    
    # 2. Background Auto-Launch (Smart Launcher)
    df = load_data(limit=500)
    if df.empty and not is_port_open(4840):
         subprocess.Popen([sys.executable, SERVER_SCRIPT], cwd=os.getcwd())
         subprocess.Popen([sys.executable, INGEST_SCRIPT], cwd=os.getcwd())
         time.sleep(2)
         st.rerun()

    # 3. Sidebar Navigation (Collapsible)
    with st.sidebar:
        st.header("Gaia Enterprise")
        st.markdown(f"**User:** {st.session_state['name']}")
        st.markdown(f"**Role:** Administrator")
        st.divider()

        selected = option_menu(
            "Navigation", 
            ["Fleet Overview", "Asset Diagnostics", "System Topology"], 
            icons=['grid-fill', 'activity', 'diagram-3-fill'], 
            menu_icon="cast", 
            default_index=0, 
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#666", "font-size": "16px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#e9ecef", "font-weight": "500"},
                "nav-link-selected": {"background-color": "#003478", "font-weight": "400"},
            }
        )
        
        st.divider()
        if st.button("Log out"):
            authenticator.logout('main')
    
    # 4. View Routing
    if selected == "Fleet Overview":
        view_fleet_overview(df)
    elif selected == "Asset Diagnostics":
        view_asset_diagnostics(df)
    elif selected == "System Topology":
        view_system_topology(df)
        
    # 5. Footer
    st.markdown("---")
    st.caption(f"Authenticated as: **{st.session_state['name']}** | Gaia Predictive v6.0")

if __name__ == "__main__":
    main()
