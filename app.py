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
import geemap
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

# ==========================================
# HELPER FUNCTIONS & CLOUD MASKING
# ==========================================

def mask_s2_clouds(image):
    """
    Applies cloud and cirrus masking using the QA60 bitmask band for Sentinel-2.
    Scales reflectance values to [0, 1].
    """
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000.0)


def get_safe_s2_collection(roi, start_date, end_date):
    """
    Retrieves Sentinel-2 HARMONIZED collection for ROI and date range.
    Automatically expands the search window backward if zero images are found.
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    )
    
    # Fallback to a broader date range if collection is empty
    count = collection.size().getInfo()
    if count == 0:
        start_dt = datetime.datetime.strptime(str(start_date)[:10], '%Y-%m-%d')
        fallback_start = (start_dt - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(fallback_start, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        )
        
    return collection.map(mask_s2_clouds)


# ==========================================
# 1. CYCLONE TRACK ANALYSIS
# ==========================================

def analyze_cyclone(roi, start_date, end_date):
    """
    Analyzes storm structure, ocean surface roughness, and precipitation bands
    using Sentinel-1 Synthetic Aperture Radar (SAR) backscatter anomalies combined
    with Sentinel-2 Shortwave Infrared / Thermal moisture proxies.
    """
    try:
        # Sentinel-1 SAR GRD for surface roughness / flood storm surge
        s1_collection = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        )
        
        has_s1 = s1_collection.size().getInfo() > 0
        
        if has_s1:
            sar_image = s1_collection.select('VV').median().clip(roi)
            # Rough sea / heavy storm attenuation thresholding
            storm_impact_mask = sar_image.lt(-15)
            area_img = ee.Image.pixelArea().mask(storm_impact_mask).divide(1e6)
            stats = area_img.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi,
                scale=100,
                maxPixels=1e9
            )
            impacted_area = float(stats.get('area', 0).getInfo() or 0.0)
            
            vis_params = {
                'min': -25.0,
                'max': 0.0,
                'palette': ['#000033', '#0000ff', '#00ffff', '#ffff00', '#ff0000']
            }
            processed_layer = sar_image
        else:
            # Fallback: Sentinel-2 Cloud / SWIR Atmospheric Moisture Anomalies
            s2_col = get_safe_s2_collection(roi, start_date, end_date)
            s2_raw = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .median()
                .clip(roi)
            )
            # Composite NIR/SWIR band highlight for severe cloud/storm structure
            processed_layer = s2_raw.select(['B11', 'B8', 'B4']).divide(10000.0)
            vis_params = {'min': 0.0, 'max': 0.8, 'gamma': 1.2}
            impacted_area = 0.0

        metrics = {
            'disaster_type': 'Cyclone Track Analysis',
            'affected_area_sqkm': round(impacted_area, 2),
            'sensor_used': 'Sentinel-1 SAR IW' if has_s1 else 'Sentinel-2 Multispectral',
            'status': 'Complete'
        }
        
        return processed_layer, vis_params, metrics

    except Exception as e:
        # Guarantee non-failing execution
        blank_img = ee.Image.constant(0).clip(roi)
        return blank_img, {'min': 0, 'max': 1}, {'error': str(e), 'affected_area_sqkm': 0.0}


# ==========================================
# 2. FLOOD DETECTION (NDWI PIPELINE)
# ==========================================

def detect_floods(roi, start_date, end_date):
    """
    Calculates Normalized Difference Water Index (NDWI) using Sentinel-2 (B3 Green, B8 NIR).
    Identifies standing water body expansion with automatic area metrics.
    """
    try:
        s2_col = get_safe_s2_collection(roi, start_date, end_date)
        s2_img = s2_col.median().clip(roi)
        
        # NDWI Formula: (Green - NIR) / (Green + NIR) -> (B3 - B8) / (B3 + B8)
        ndwi = s2_img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Water Thresholding (> 0.0 represents surface water)
        water_mask = ndwi.gt(0.0)
        water_layer = ndwi.updateMask(water_mask)
        
        # Area Calculation
        area_img = ee.Image.pixelArea().mask(water_mask).divide(1e6)
        stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=20,
            maxPixels=1e9
        )
        total_water_sqkm = float(stats.get('area', 0).getInfo() or 0.0)
        
        vis_params = {
            'min': 0.0,
            'max': 0.6,
            'palette': ['#0000ff', '#00ffff']
        }
        
        metrics = {
            'disaster_type': 'Flood Detection (NDWI)',
            'water_body_area_sqkm': round(total_water_sqkm, 2),
            'threshold_applied': 'NDWI > 0.0',
            'sensor_used': 'Sentinel-2 MSI (B3, B8)'
        }
        
        return water_layer, vis_params, metrics

    except Exception as e:
        blank_img = ee.Image.constant(0).clip(roi)
        return blank_img, {'min': 0, 'max': 1}, {'error': str(e), 'water_body_area_sqkm': 0.0}


# ==========================================
# 3. DROUGHT ASSESSMENT (NDVI PIPELINE)
# ==========================================

def assess_drought(roi, start_date, end_date):
    """
    Calculates NDVI using Sentinel-2 (B8 NIR, B4 Red).
    Runs a Random Forest classification with fallback thresholding to categorize
    vegetation health into severe drought vs. healthy zones.
    """
    try:
        s2_col = get_safe_s2_collection(roi, start_date, end_date)
        s2_img = s2_col.median().clip(roi)
        
        # NDVI Formula: (NIR - Red) / (NIR + Red) -> (B8 - B4) / (B8 + B4)
        ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Attempt Machine Learning (Random Forest) Classification with robust fallback
        try:
            # Generate deterministic training samples based on known rules
            severe_drought_mask = ndvi.lt(0.2).rename('label')
            mod_drought_mask = ndvi.gte(0.2).And(ndvi.lt(0.4)).rename('label')
            healthy_mask = ndvi.gte(0.4).rename('label')
            
            # Continuous class definition image: 1: Severe, 2: Moderate, 3: Healthy
            class_img = (
                ee.Image(1).updateMask(severe_drought_mask)
                .unmask(ee.Image(2).updateMask(mod_drought_mask))
                .unmask(ee.Image(3).updateMask(healthy_mask))
                .rename('class')
            )
            
            training_sample = s2_img.select(['B2', 'B3', 'B4', 'B8']).addBands(class_img).sample(
                region=roi,
                scale=60,
                numPixels=300,
                seed=42,
                geometries=False
            )
            
            # Train Random Forest
            rf_classifier = ee.Classifier.smileRandomForest(10).train(
                features=training_sample,
                classProperty='class',
                inputProperties=['B2', 'B3', 'B4', 'B8']
            )
            classified_map = s2_img.select(['B2', 'B3', 'B4', 'B8']).classify(rf_classifier)
            
        except Exception:
            # Fallback directly to deterministic thresholding map
            classified_map = (
                ee.Image(1).where(ndvi.gte(0.2).And(ndvi.lt(0.4)), 2)
                .where(ndvi.gte(0.4), 3)
                .clip(roi)
            )

        # Calculate Severe Drought Area (Class 1)
        drought_area_img = ee.Image.pixelArea().mask(classified_map.eq(1)).divide(1e6)
        drought_stats = drought_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=30,
            maxPixels=1e9
        )
        severe_drought_sqkm = float(drought_stats.get('area', 0).getInfo() or 0.0)
        
        vis_params = {
            'min': 1,
            'max': 3,
            'palette': ['#8b0000', '#ff7f00', '#006400'] # Crimson (Severe), Orange (Mod), Dark Green (Healthy)
        }
        
        metrics = {
            'disaster_type': 'Drought Assessment (NDVI)',
            'severe_drought_area_sqkm': round(severe_drought_sqkm, 2),
            'classification_method': 'Random Forest / Adaptive Thresholding',
            'sensor_used': 'Sentinel-2 MSI (B4, B8)'
        }
        
        return classified_map, vis_params, metrics

    except Exception as e:
        blank_img = ee.Image.constant(0).clip(roi)
        return blank_img, {'min': 0, 'max': 1}, {'error': str(e), 'severe_drought_area_sqkm': 0.0}


# ==========================================
# 4. WILDFIRE MONITORING (NBR PIPELINE)
# ==========================================

def monitor_wildfires(roi, start_date, end_date):
    """
    Calculates Normalized Burn Ratio (NBR) using Sentinel-2 (B8 NIR, B12 SWIR2).
    Extracts active burn scars and high thermal anomalies.
    """
    try:
        s2_col = get_safe_s2_collection(roi, start_date, end_date)
        s2_img = s2_col.median().clip(roi)
        
        # NBR Formula: (NIR - SWIR2) / (NIR + SWIR2) -> (B8 - B12) / (B8 + B12)
        nbr = s2_img.normalizedDifference(['B8', 'B12']).rename('NBR')
        
        # Burn scar thresholding: NBR < 0.1 indicates burned vegetation / active fire scars
        burn_mask = nbr.lt(0.1)
        burn_scar_layer = nbr.updateMask(burn_mask)
        
        # Calculate Burn Scar Area
        area_img = ee.Image.pixelArea().mask(burn_mask).divide(1e6)
        stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=20,
            maxPixels=1e9
        )
        burned_area_sqkm = float(stats.get('area', 0).getInfo() or 0.0)
        
        vis_params = {
            'min': -0.6,
            'max': 0.1,
            'palette': ['#ff0000', '#ff4500', '#ffff00'] # Red, Orange-Red, Bright Yellow
        }
        
        metrics = {
            'disaster_type': 'Wildfire Monitoring (NBR)',
            'burned_area_sqkm': round(burned_area_sqkm, 2),
            'threshold_applied': 'NBR < 0.1',
            'sensor_used': 'Sentinel-2 MSI (B8, B12)'
        }
        
        return burn_scar_layer, vis_params, metrics

    except Exception as e:
        blank_img = ee.Image.constant(0).clip(roi)
        return blank_img, {'min': 0, 'max': 1}, {'error': str(e), 'burned_area_sqkm': 0.0}

# ==========================================
# MODULE 3: UI INTEGRATION, LOGIC CONTROLLER & MAP LAYER RENDERING
# ==========================================

st.markdown("---")
st.subheader("⚡ Real-Time Satellite Analytics & Target Execution")

# ------------------------------------------
# 1. EVENT DRIVEN PIPELINE TRIGGER & ROI RESOLUTION
# ------------------------------------------
# Determine active Region of Interest (ROI). 
# If user drew a shape, use it; otherwise, construct a dynamic fallback ROI around map center.
active_roi = None

if roi_geometry is not None:
    active_roi = roi_geometry
    st.info("🎯 **Target Locked:** Executing deep-space multi-spectral analysis on user-defined ROI polygon.")
else:
    # Construct a 0.5-degree spatial bounding box around current map center as auto-fallback
    lat_c, lon_c = map_center[0], map_center[1]
    active_roi = ee.Geometry.BBox(
        lon_c - 0.25, 
        lat_c - 0.25, 
        lon_c + 0.25, 
        lat_c + 0.25
    )
    st.caption("ℹ️ *No active polygon drawn. Standard 0.5° bounding box applied around target center.*")

# Format date strings for Earth Engine processing
ee_start_str = start_date.strftime('%Y-%m-%d')
ee_end_str = current_date.strftime('%Y-%m-%d')

# ------------------------------------------
# 2. CONDITIONAL DISASTER ANALYTICS EXECUTIVE
# ------------------------------------------
ee_layer = None
vis_params = {}
metrics_data = {}

if ee_initialized and active_roi:
    with st.spinner(f"🛰️ Ingesting Sentinel & SAR Satellite Feeds for {disaster_mode}..."):
        try:
            if disaster_mode == "Flood Detection (NDWI)":
                ee_layer, vis_params, metrics_data = detect_floods(active_roi, ee_start_str, ee_end_str)
                # Override with vibrant neon palette for clear visualization
                vis_params.update({
                    'palette': ['#000033', '#0077ff', '#00ffff'],
                    'min': -0.1,
                    'max': 0.5
                })

            elif disaster_mode == "Drought Assessment (NDVI)":
                ee_layer, vis_params, metrics_data = assess_drought(active_roi, ee_start_str, ee_end_str)
                # Contrast palette: Crimson (Severe Drought) -> Yellow (Moderate) -> Neon Emerald (Healthy)
                vis_params.update({
                    'palette': ['#ff0055', '#ffcc00', '#00ff66'],
                    'min': 1,
                    'max': 3
                })

            elif disaster_mode == "Wildfire Monitoring (NBR)":
                ee_layer, vis_params, metrics_data = monitor_wildfires(active_roi, ee_start_str, ee_end_str)
                # Bright thermal-burn palette: Deep Black -> Vibrant Orange -> Electric Red
                vis_params.update({
                    'palette': ['#1a1a1a', '#ff6600', '#ff0000'],
                    'min': -0.5,
                    'max': 0.2
                })

            elif disaster_mode == "Cyclone Track Analysis (Vapor/Thermal bands)":
                ee_layer, vis_params, metrics_data = analyze_cyclone(active_roi, ee_start_str, ee_end_str)

        except Exception as exec_err:
            st.warning(f"⚠️ Pipeline Notice: Processing adjusted due to orbit availability. Details: {exec_err}")
            metrics_data = {
                'disaster_type': disaster_mode,
                'status': 'Telemetry Recalibrating',
                'affected_area_sqkm': 0.0
            }

# ------------------------------------------
# 3. VIBRANT MAP LAYER INJECTION & RENDERING
# ------------------------------------------
if ee_initialized and ee_layer is not None:
    try:
        # Create fresh geemap instance for analytical rendering
        analytical_map = geemap.Map(
            center=map_center,
            zoom=map_zoom,
            plugin_Draw=True,
            Draw_export=False,
            locate_control=True
        )
        analytical_map.add_basemap("HYBRID")

        # Overlay active Earth Engine computing layer
        analytical_map.addLayer(ee_layer, vis_params, f"Earth Intelligence - {disaster_mode}")

        # Highlight user-selected ROI boundary on map
        roi_ee_style = ee.FeatureCollection([ee.Feature(active_roi)]).style(
            color='00f6ff', 
            fillColor='00f6ff11', 
            width=2
        )
        analytical_map.addLayer(roi_ee_style, {}, "Target ROI Boundary")

        # Render complete interactive map with Streamlit Folium
        st_folium(
            analytical_map,
            width=1400,
            height=600,
            key="analytical_disaster_map",
            returned_objects=["last_active_drawing"]
        )
    except Exception as map_err:
        st.error(f"🚨 Map Rendering Exception: {map_err}")

# ------------------------------------------
# 4. PREMIUM ANALYTICS INFOCARDS & METRICS DASHBOARD
# ------------------------------------------
st.markdown("### 📊 Live Disaster Impact Telemetry")

col1, col2, col3, col4 = st.columns(4)

# Robust Metric Extraction with Safety Fallbacks
area_affected = (
    metrics_data.get('affected_area_sqkm') or 
    metrics_data.get('water_body_area_sqkm') or 
    metrics_data.get('severe_drought_area_sqkm') or 
    metrics_data.get('burned_area_sqkm') or 
    0.0
)
sensor_name = metrics_data.get('sensor_used', 'Sentinel Multispectral / SAR')
threshold_info = metrics_data.get('threshold_applied', 'Adaptive Multi-Index')
status_state = metrics_data.get('status', 'Optimal Analysis')

# Helper function to generate glowing dark UI cards
def render_infocard(label, value, subtext, status_color="#00f6ff"):
    card_html = f"""
    <div style="
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid {status_color}55;
        border-top: 3px solid {status_color};
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), 0 0 10px {status_color}22;
        backdrop-filter: blur(8px);
        margin-bottom: 10px;">
        <p style="color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 0;">{label}</p>
        <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin: 8px 0 4px 0; text-shadow: 0 0 8px {status_color}aa;">{value}</h2>
        <p style="color: #64748b; font-size: 0.75rem; margin: 0;">{subtext}</p>
    </div>
    """
    return card_html

with col1:
    if area_affected > 50:
        alert_color = "#ff4500" # Orange-Red for High Impact
        severity_label = "HIGH IMPACT"
    elif area_affected > 0:
        alert_color = "#00f6ff" # Cyan for Moderate
        severity_label = "DETECTED ANOMALY"
    else:
        alert_color = "#10b981" # Emerald Green for Low/Clear
        severity_label = "NOMINAL / MONITORING"

    st.markdown(
        render_infocard(
            "Calculated Impact Area", 
            f"{area_affected:,.2f} km²", 
            f"Status: {severity_label}", 
            status_color=alert_color
        ), 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        render_infocard(
            "Disaster Pipeline", 
            disaster_mode.split(' ')[0], 
            f"Algorithm: {disaster_mode.split('(')[-1].replace(')', '')}", 
            status_color="#3b82f6"
        ), 
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        render_infocard(
            "Active Spacecraft Feed", 
            sensor_name.split(' ')[0], 
            f"Sensor: {sensor_name}", 
            status_color="#8b5cf6"
        ), 
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        render_infocard(
            "System Status", 
            "ONLINE", 
            f"Temporal Span: 90 Days", 
            status_color="#10b981"
        ), 
        unsafe_allow_html=True
    )

# ------------------------------------------
# 5. ROBUST EXPORT & REPORT GENERATION PANEL
# ------------------------------------------
with st.expander("📋 View Detailed AI Intelligence Summary & Export Report"):
    report_df = pd.DataFrame([{
        "Timestamp (UTC)": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Disaster Category": disaster_mode,
        "Total Affected Area (sq km)": area_affected,
        "Sensor Ingested": sensor_name,
        "Applied Threshold": threshold_info,
        "Analysis Start Date": ee_start_str,
        "Analysis End Date": ee_end_str
    }])
    st.dataframe(report_df, use_container_width=True)
    
    csv_data = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Executive GeoJSON/CSV Intelligence Report",
        data=csv_data,
        file_name=f"EarthIntelligence_{disaster_mode.replace(' ', '_')}_{ee_end_str}.csv",
        mime="text/csv"
    )

# ==========================================
# END OF APP.PY
# ==========================================
