#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Earth Intelligence - Open-World Multi-Disaster Prediction & Monitoring Application
Enterprise-grade Streamlit application integrating Google Earth Engine for live satellite analysis.
"""

# ==========================================
# 1. ABSOLUTE LIBRARY IMPORTS
# ==========================================
import json
import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium

# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION & DARK THEME CSS
# ==========================================
st.set_page_config(
    page_title="Earth Intelligence | Multi-Disaster AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

PREMIUM_DARK_CSS = """
<style>
    /* Global Base Styling */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b, #0f172a 70%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(0, 246, 255, 0.2);
        box-shadow: 2px 0 15px rgba(0, 246, 255, 0.05);
    }
    
    /* Input Control Components */
    .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: #00f6ff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px;
        box-shadow: inset 0 0 5px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stTextInput>div>div>input:focus {
        border: 1px solid #00f6ff !important;
        box-shadow: 0 0 12px rgba(0, 246, 255, 0.5);
    }
    
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px;
        color: #e2e8f0 !important;
    }
    
    /* Map Frame Aesthetics */
    iframe {
        border: 2px solid #00f6ff !important;
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(0, 246, 255, 0.2) !important;
    }
    
    /* Metrics Header Customizations */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
</style>
"""
st.markdown(PREMIUM_DARK_CSS, unsafe_allow_html=True)

# ==========================================
# 3. BULLETPROOF GEE INITIALIZATION
# ==========================================
@st.cache_resource
def initialize_earth_engine():
    """Safely parses Streamlit secrets and initializes Earth Engine with ServiceAccountCredentials."""
    try:
        if "EARTH_ENGINE_CREDENTIALS" in st.secrets:
            creds_raw = st.secrets["EARTH_ENGINE_CREDENTIALS"]
            if isinstance(creds_raw, str):
                creds_dict = json.loads(creds_raw, strict=False)
            else:
                creds_dict = dict(creds_raw)
            
            creds = ee.ServiceAccountCredentials(
                creds_dict['client_email'],
                key_data=json.dumps(creds_dict)
            )
            ee.Initialize(credentials=creds)
            return True, "Service Account initialized."
        else:
            ee.Initialize()
            return True, "Standard OAuth initialized."
    except Exception as e:
        return False, str(e)

ee_initialized, ee_status_msg = initialize_earth_engine()

# ==========================================
# 4. SATELLITE PROCESSING & CLOUD MASKING PIPELINES
# ==========================================
def mask_s2_clouds(image):
    """Masks clouds and cirrus in Sentinel-2 images using QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000.0)

def get_safe_s2_collection(roi, start_date, end_date):
    """Retrieves cloud-masked Sentinel-2 HARMONIZED collection with fallback range."""
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    )
    
    # Fallback to broader range if initial query returns no images
    if collection.size().getInfo() == 0:
        start_dt = datetime.datetime.strptime(str(start_date)[:10], '%Y-%m-%d')
        fallback_start = (start_dt - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(fallback_start, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        )
        
    return collection.map(mask_s2_clouds)

def detect_floods(roi, start_date, end_date):
    """Calculates NDWI = (Green - NIR) / (Green + NIR) using Sentinel-2 B3 and B8."""
    try:
        s2_col = get_safe_s2_collection(roi, start_date, end_date)
        s2_img = s2_col.median().clip(roi)
        ndwi = s2_img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        water_mask = ndwi.gt(0.0)
        water_layer = ndwi.updateMask(water_mask)
        
        area_img = ee.Image.pixelArea().mask(water_mask).divide(1e6)
        stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=20, maxPixels=1e9
        )
        water_sqkm = float(stats.get('area', 0).getInfo() or 0.0)
        
        vis_params = {
            'min': 0.0,
            'max': 0.6,
            'palette': ['#003366', '#00ffff']
        }
        metrics = {
            'disaster_type': 'Flood Detection (NDWI)',
            'affected_area_sqkm': round(water_sqkm, 2),
            'sensor_used': 'Sentinel-2 MSI (B3, B8)',
            'threshold': 'NDWI > 0.0'
        }
        return water_layer, vis_params, metrics
    except Exception as e:
        return ee.Image.constant(0).clip(roi), {'min': 0, 'max': 1}, {'affected_area_sqkm': 0.0, 'error': str(e)}

def assess_drought(roi, start_date, end_date):
    """Calculates NDVI = (NIR - Red) / (NIR + Red) using Sentinel-2 B8 and B4."""
    try:
        s2_col = get_safe_s2_collection(roi, start_date, end_date)
        s2_img = s2_col.median().clip(roi)
        ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Drought Severity Categorization: 1: Severe (<0.2), 2: Moderate (0.2-0.4), 3: Healthy (>0.4)
        drought_classified = (
            ee.Image(1).where(ndvi.gte(0.2).And(ndvi.lt(0.4)), 2)
            .where(ndvi.gte(0.4), 3)
            .clip(roi)
        )
        
        drought_area_img = ee.Image.pixelArea().mask(drought_classified.eq(1)).divide(1e6)
        stats = drought_area_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=30, maxPixels=1e9
        )
        severe_drought_sqkm = float(stats.get('area', 0).getInfo() or 0.0)
        
        vis_params = {
            'min': 1,
            'max': 3,
            'palette': ['#ff0033', '#ffcc00', '#00ff66'] # Red (Severe), Yellow (Mod), Green (Healthy)
        }
        metrics = {
            'disaster_type': 'Drought Assessment (NDVI)',
            'affected_area_sqkm': round(severe_drought_sqkm, 2),
            'sensor_used': 'Sentinel-2 MSI (B4, B8)',
            'threshold': 'NDVI < 0.2 (Severe)'
        }
        return drought_classified, vis_params, metrics
    except Exception as e:
        return ee.Image.constant(0).clip(roi), {'min': 0, 'max': 1}, {'affected_area_sqkm': 0.0, 'error': str(e)}

def monitor_wildfires(roi, start_date, end_date):
    """Calculates NBR = (NIR - SWIR2) / (NIR + SWIR2) using Sentinel-2 B8 and B12."""
    try:
        s2_col = get_safe_s2_collection(roi, start_date, end_date)
        s2_img = s2_col.median().clip(roi)
        nbr = s2_img.normalizedDifference(['B8', 'B12']).rename('NBR')
        
        burn_mask = nbr.lt(0.1)
        burn_layer = nbr.updateMask(burn_mask)
        
        area_img = ee.Image.pixelArea().mask(burn_mask).divide(1e6)
        stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=20, maxPixels=1e9
        )
        burned_sqkm = float(stats.get('area', 0).getInfo() or 0.0)
        
        vis_params = {
            'min': -0.6,
            'max': 0.1,
            'palette': ['#1a1a1a', '#ff6600', '#ff0000'] # Black -> Orange -> Red
        }
        metrics = {
            'disaster_type': 'Wildfire Monitoring (NBR)',
            'affected_area_sqkm': round(burned_sqkm, 2),
            'sensor_used': 'Sentinel-2 MSI (B8, B12)',
            'threshold': 'NBR < 0.1'
        }
        return burn_layer, vis_params, metrics
    except Exception as e:
        return ee.Image.constant(0).clip(roi), {'min': 0, 'max': 1}, {'affected_area_sqkm': 0.0, 'error': str(e)}

def analyze_cyclone(roi, start_date, end_date):
    """Analyzes storm surface impacts using Sentinel-1 SAR backscatter VV anomalies."""
    try:
        s1_col = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        )
        
        if s1_col.size().getInfo() > 0:
            sar_img = s1_col.select('VV').median().clip(roi)
            roughness_mask = sar_img.lt(-15)
            
            area_img = ee.Image.pixelArea().mask(roughness_mask).divide(1e6)
            stats = area_img.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=roi, scale=100, maxPixels=1e9
            )
            affected_sqkm = float(stats.get('area', 0).getInfo() or 0.0)
            
            vis_params = {
                'min': -25.0,
                'max': 0.0,
                'palette': ['#000033', '#0000ff', '#00ffff', '#ffff00', '#ff0000']
            }
            sensor_used = 'Sentinel-1 SAR IW (VV)'
            layer = sar_img
        else:
            # Fallback to Sentinel-2 Moisture Proxy (SWIR/NIR)
            s2_col = get_safe_s2_collection(roi, start_date, end_date)
            layer = s2_col.median().clip(roi).select(['B11', 'B8', 'B4'])
            vis_params = {'min': 0.0, 'max': 0.8, 'gamma': 1.2}
            affected_sqkm = 0.0
            sensor_used = 'Sentinel-2 MSI SWIR Moisture Proxy'

        metrics = {
            'disaster_type': 'Cyclone Track Analysis',
            'affected_area_sqkm': round(affected_sqkm, 2),
            'sensor_used': sensor_used,
            'threshold': 'Radar Anomaly < -15dB'
        }
        return layer, vis_params, metrics
    except Exception as e:
        return ee.Image.constant(0).clip(roi), {'min': 0, 'max': 1}, {'affected_area_sqkm': 0.0, 'error': str(e)}

# ==========================================
# 5. SIDEBAR CONTROLS & LIVE DATE STREAMER
# ==========================================
st.sidebar.title("🌍 Earth Intelligence")
st.sidebar.markdown("---")

disaster_mode = st.sidebar.selectbox(
    "🛰️ Select Disaster Pipeline",
    [
        "Flood Detection (NDWI)",
        "Drought Assessment (NDVI)",
        "Wildfire Monitoring (NBR)",
        "Cyclone Track Analysis"
    ]
)

current_date = datetime.date.today()
start_date = current_date - datetime.timedelta(days=90)

ee_start_str = start_date.strftime('%Y-%m-%d')
ee_end_str = current_date.strftime('%Y-%m-%d')

st.sidebar.markdown("### ⏱️ Temporal Window")
st.sidebar.info(
    f"**Live Dynamic Feed (90 Days):**\n\n"
    f"🟢 **Start:** {ee_start_str}\n\n"
    f"🔴 **End:** {ee_end_str}"
)

st.sidebar.markdown("---")
if ee_initialized:
    st.sidebar.success("🟢 Google Earth Engine: Connected")
else:
    st.sidebar.error(f"🔴 GEE Error: {ee_status_msg}")

# ==========================================
# 6. DYNAMIC GEOLOCATION & MAP CONTROL PANEL
# ==========================================
st.title("🌐 Open-World Multi-Disaster Prediction Engine")
st.markdown("Type any location globally or draw a custom Region of Interest (ROI) on the map to trigger live analysis.")

search_query = st.text_input("🔍 Global Location Search", placeholder="e.g., California, USA or Tokyo, Japan")

map_center = [20.0, 0.0]
map_zoom = 2

if search_query:
    try:
        headers = {'User-Agent': 'EarthIntelligenceApp/1.0'}
        url = f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=1"
        res = requests.get(url, headers=headers).json()
        if res:
            map_center = [float(res[0]['lat']), float(res[0]['lon'])]
            map_zoom = 10
            st.success(f"📍 Target locked: {res[0]['display_name']}")
        else:
            st.warning("⚠️ Location not found. Maintaining default global view.")
    except Exception as ge:
        st.error(f"Geocoding error: {ge}")

# Initialize Geemap canvas
m = geemap.Map(
    center=map_center,
    zoom=map_zoom,
    plugin_Draw=True,
    Draw_export=False,
    locate_control=True
)
m.add_basemap("HYBRID")

# ==========================================
# 7. EVENT DRIVEN LOGIC CONTROLLER & RENDERING
# ==========================================
# Render the drawing map canvas
map_data = st_folium(
    m,
    width=1400,
    height=550,
    key="earth_intelligence_map",
    returned_objects=["last_active_drawing"]
)

# Robust ROI Resolution
roi = None
if map_data and map_data.get("last_active_drawing"):
    drawing_geom = map_data["last_active_drawing"]["geometry"]
    geom_type = drawing_geom["type"]
    coords = drawing_geom["coordinates"]
    
    if geom_type == "Polygon":
        roi = ee.Geometry.Polygon(coords)
        st.info("🎯 Custom Drawn Polygon ROI active.")
    elif geom_type == "Point":
        roi = ee.Geometry.Point(coords).buffer(10000)
        st.info("🎯 Point ROI active (10km buffer applied).")
else:
    # Default 0.5-degree bounding box around the current map center
    lat_c, lon_c = map_center[0], map_center[1]
    roi = ee.Geometry.BBox(lon_c - 0.25, lat_c - 0.25, lon_c + 0.25, lat_c + 0.25)
    st.caption("ℹ️ *Default 0.5° bounding box active around target location.*")

# Executive Pipeline Execution
ee_layer = None
vis_params = {}
metrics_data = {}

if ee_initialized and roi:
    with st.spinner(f"🛰️ Processing Satellite Feeds for {disaster_mode}..."):
        if disaster_mode == "Flood Detection (NDWI)":
            ee_layer, vis_params, metrics_data = detect_floods(roi, ee_start_str, ee_end_str)
        elif disaster_mode == "Drought Assessment (NDVI)":
            ee_layer, vis_params, metrics_data = assess_drought(roi, ee_start_str, ee_end_str)
        elif disaster_mode == "Wildfire Monitoring (NBR)":
            ee_layer, vis_params, metrics_data = monitor_wildfires(roi, ee_start_str, ee_end_str)
        elif disaster_mode == "Cyclone Track Analysis":
            ee_layer, vis_params, metrics_data = analyze_cyclone(roi, ee_start_str, ee_end_str)

# Map Overlay for Analysis Layer
if ee_initialized and ee_layer is not None:
    analysis_map = geemap.Map(center=map_center, zoom=map_zoom)
    analysis_map.add_basemap("HYBRID")
    analysis_map.addLayer(ee_layer, vis_params, f"Analysis: {disaster_mode}")
    
    # Highlight ROI boundary
    roi_outline = ee.FeatureCollection([ee.Feature(roi)]).style(color='00f6ff', fillColor='00f6ff11', width=2)
    analysis_map.addLayer(roi_outline, {}, "ROI Boundary")
    
    st.markdown("### 🗺️ Analyzed Satellite Visualization")
    st_folium(analysis_map, width=1400, height=550, key="processed_output_map")

# ==========================================
# 8. METRIC INFOCARDS & REPORT EXPORT
# ==========================================
st.markdown("### 📊 Real-Time Analytics & Telemetry")

col1, col2, col3, col4 = st.columns(4)

affected_area = metrics_data.get('affected_area_sqkm', 0.0)
sensor_used = metrics_data.get('sensor_used', 'Sentinel Spacecraft')
threshold = metrics_data.get('threshold', 'Adaptive Index')

def create_card(label, value, subtext, border_color="#00f6ff"):
    return f"""
    <div style="
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid {border_color}55;
        border-top: 3px solid {border_color};
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;">
        <p style="color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin:0;">{label}</p>
        <h2 style="color: #ffffff; font-size: 1.7rem; font-weight: 800; margin: 6px 0;">{value}</h2>
        <p style="color: #64748b; font-size: 0.75rem; margin:0;">{subtext}</p>
    </div>
    """

with col1:
    status_color = "#ff0055" if affected_area > 50 else "#00f6ff"
    st.markdown(create_card("Impacted Area", f"{affected_area:,.2f} km²", f"Threshold: {threshold}", status_color), unsafe_allow_html=True)

with col2:
    st.markdown(create_card("Active Pipeline", disaster_mode.split()[0], f"Mode: {disaster_mode}", "#3b82f6"), unsafe_allow_html=True)

with col3:
    st.markdown(create_card("Satellite Ingest", sensor_used.split()[0], f"Sensor: {sensor_used}", "#8b5cf6"), unsafe_allow_html=True)

with col4:
    st.markdown(create_card("System Health", "ONLINE", f"Window: Past 90 Days", "#10b981"), unsafe_allow_html=True)

# Export Data Section
with st.expander("📋 Export Intelligence Report"):
    df_report = pd.DataFrame([{
        "Timestamp (UTC)": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Disaster Category": disaster_mode,
        "Impacted Area (sq km)": affected_area,
        "Spacecraft / Sensor": sensor_used,
        "Applied Threshold": threshold,
        "Start Date": ee_start_str,
        "End Date": ee_end_str
    }])
    st.dataframe(df_report, use_container_width=True)
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Executive CSV Report",
        data=csv,
        file_name=f"EarthIntelligence_Report_{ee_end_str}.csv",
        mime="text/csv"
    )
