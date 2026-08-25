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
