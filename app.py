#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Earth Intelligence - Open-World Multi-Disaster Prediction App
Module 1: Infrastructure Initialization, Premium UI/UX, and Global Control Panel
"""

# ==========================================
# 1. ABSOLUTE LIBRARY IMPORTS
# ==========================================
import streamlit as st
import ee
import geemap.foliumap as geemap
import datetime
import pandas as pd
import numpy as np
import requests
import json
from streamlit_folium import st_folium
from google.oauth2 import service_account

# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Earth Intelligence | Multi-Disaster AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 3. PREMIUM CORPORATE DARK-THEME CSS
# ==========================================
PREMIUM_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b, #0f172a 70%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling with Glow */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid #00f6ff33;
        box-shadow: 2px 0 15px rgba(0, 246, 255, 0.1);
    }
    
    /* Input Elements (Text Box, Dropdowns) */
    .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: #00f6ff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px;
        box-shadow: inset 0 0 5px rgba(59, 130, 246, 0.5);
        transition: all 0.3s ease-in-out;
    }
    .stTextInput>div>div>input:focus {
        border: 1px solid #00f6ff !important;
        box-shadow: 0 0 15px rgba(0, 246, 255, 0.6), inset 0 0 8px rgba(0, 246, 255, 0.4);
    }
    
    /* Selectbox Styling */
    div[data-baseweb="select"] > div {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        color: #e2e8f0;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.2);
    }
    
    /* Map Container Glowing Border */
    iframe {
        border: 2px solid #00f6ff;
        border-radius: 12px;
        box-shadow: 0 0 25px rgba(0, 246, 255, 0.25), inset 0 0 15px rgba(0, 246, 255, 0.15);
        background-color: #000000;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid #10b981 !important;
        color: #10b981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ==========================================
# 4. ROBUST GOOGLE EARTH ENGINE INIT
# ==========================================
@st.cache_resource
def initialize_earth_engine():
    """Initializes Google Earth Engine using Streamlit Secrets with error handling."""
    try:
        if not ee.data._credentials:
            ee_creds = st.secrets["EARTH_ENGINE_CREDENTIALS"]
            credentials = service_account.Credentials.from_service_account_info(ee_creds)
            ee.Initialize(credentials)
            return True
    except KeyError:
        st.error("🚨 Configuration Error: 'EARTH_ENGINE_CREDENTIALS' not found in st.secrets.")
        return False
    except Exception as e:
        st.error(f"🚨 Earth Engine Initialization Failed: {str(e)}")
        return False
    return True

ee_initialized = initialize_earth_engine()

# ==========================================
# 5. DISASTER SELECTOR SIDEBAR
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Google_Earth_icon.svg/2048px-Google_Earth_icon.svg.png", width=60)
st.sidebar.title("Earth Intelligence")
st.sidebar.markdown("---")

disaster_mode = st.sidebar.selectbox(
    "🛰️ Select Disaster Analysis Mode",
    [
        "Flood Detection (NDWI)",
        "Drought Assessment (NDVI)",
        "Wildfire Monitoring (NBR)",
        "Cyclone Track Analysis (Vapor/Thermal bands)"
    ],
    index=0
)

# ==========================================
# 6. REAL-TIME DATE STREAMER
# ==========================================
st.sidebar.markdown("### ⏱️ Temporal Parameters")
current_date = datetime.datetime.today()
start_date = current_date - datetime.timedelta(days=90)

st.sidebar.info(
    f"**Live Data Stream (Last 90 Days):**\n\n"
    f"🟢 **Start:** {start_date.strftime('%Y-%m-%d')}\n\n"
    f"🔴 **End:** {current_date.strftime('%Y-%m-%d')}"
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Google Earth Engine & AI")

# ==========================================
# 7. GLOBAL OPEN-WORLD CONTROL PANEL
# ==========================================
st.title("🌐 Global Open-World Control Panel")
st.markdown("Search for any region globally and draw a polygon to extract real-time AI analytics.")

# Search Box Logic using Nominatim API for accurate lat/lon resolution
search_query = st.text_input("🔍 Global Search (City, Province, or Country)", placeholder="e.g., California, USA or Tokyo, Japan")

# Default Global Center
map_center = [20.0, 0.0]
map_zoom = 2

if search_query:
    try:
        headers = {'User-Agent': 'EarthIntelligenceApp/1.0'}
        url = f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=1"
        response = requests.get(url, headers=headers).json()
        if response:
            map_center = [float(response[0]['lat']), float(response[0]['lon'])]
            map_zoom = 10
            st.success(f"📍 Location locked: {response[0]['display_name']}")
        else:
            st.warning("⚠️ Location not found. Showing global view.")
    except Exception as e:
        st.error(f"Geocoding Error: {e}")

# Render Interactive Geemap
if ee_initialized:
    # Initialize Map with drawing tools enabled
    m = geemap.Map(
        center=map_center, 
        zoom=map_zoom, 
        plugin_Draw=True,
        Draw_export=False,
        locate_control=True,
        plugin_LatLngPopup=False
    )
    
    # Add Premium Basemap (Google Hybrid)
    m.add_basemap("HYBRID")
    
    # Map rendering with Streamlit Folium wrapper to capture events
    st.markdown("### 🗺️ Intelligence Map")
    map_data = st_folium(
        m, 
        width=1400, 
        height=650, 
        returned_objects=["last_active_drawing", "last_clicked"]
    )
    
    # Event Listener / Placeholder for captured ROI (Region of Interest)
    st.markdown("### 🎯 Telemetry & Geometry Capture")
    
    roi_geometry = None
    if map_data and map_data.get("last_active_drawing"):
        drawing_data = map_data["last_active_drawing"]
        geom_type = drawing_data["geometry"]["type"]
        coords = drawing_data["geometry"]["coordinates"]
        
        # Convert drawn geojson to Earth Engine Geometry
        if geom_type == "Polygon":
            roi_geometry = ee.Geometry.Polygon(coords)
            st.success(f"✅ Polygon ROI Captured successfully! Coordinates mapped for {disaster_mode}.")
        elif geom_type == "Point":
            roi_geometry = ee.Geometry.Point(coords)
            st.success(f"✅ Point ROI Captured successfully! Coordinates mapped for {disaster_mode}.")
            
        with st.expander("View Raw GeoJSON Coordinates"):
            st.json(drawing_data)
            
    elif map_data and map_data.get("last_clicked"):
        click_data = map_data["last_clicked"]
        st.info(f"🖱️ Map clicked at Latitude: {click_data['lat']:.4f}, Longitude: {click_data['lng']:.4f}. Draw a polygon to initiate deep analysis.")
    else:
        st.info("✋ Waiting for user interaction. Please draw a bounding box/polygon on the map to proceed.")

else:
    st.warning("Map visualization disabled pending Earth Engine authentication.")

# --- End of Module 1 ---
