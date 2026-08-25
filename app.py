"""
===============================================================================
PRODUCTION-READY EARTH ENGINE & SCIKIT-LEARN DROUGHT/FLOOD PREDICTIVE PIPELINE
===============================================================================
Architecture Overview:
----------------------
1. Ingestion & Preprocessing: Fetches harmonized Sentinel-2 Surface Reflectance 
   data, applies QA60 cloud masking, and clips to a target Region of Interest (ROI).
2. Index Extraction: Computes Normalized Difference Vegetation Index (NDVI) and 
   Normalized Difference Water Index (NDWI) directly within Earth Engine.
3. Feature Engineering & Sampling: Generates stratified ground-truth samples 
   using threshold-based heuristics to create training data ('Flood', 'Drought', 'Normal').
4. ML Engine: Trains an optimized scikit-learn Random Forest Classifier on remote
   sensing spectral signatures.
5. EE Classification Integration: Synchronizes trained features into server-side 
   ee.Classifier for scalable inference across millions of raster pixels.
6. Analytics & Map Generation: Synthesizes class distribution stats, metrics, 
   and interactive Map objects using geemap.
"""

from typing import Dict, Any, Tuple
import logging
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import ee
import geemap

# Configure logging standard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("EarthEngineAIPipeline")

# Global Configuration Parameters
CLASS_MAPPING = {"Drought": 0, "Normal": 1, "Flood": 2}
CLASS_LABELS = {0: "Drought", 1: "Normal", 2: "Flood"}
PALETTE = ["#d7191c", "#fdae61", "#2b83ba"]  # Red = Drought, Yellow = Normal, Blue = Flood


# =============================================================================
# 1. EARTH ENGINE INITIALIZATION & PREPROCESSING
# =============================================================================

def initialize_earth_engine(project_id: str) -> None:
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Successfully initialized Earth Engine project: {project_id}")
    except Exception as exc:
        logger.error(f"Failed to initialize Earth Engine: {exc}")
        raise RuntimeError("EE Initialization Error") from exc


def mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds and cirrus using Sentinel-2 QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000.0)


def compute_spectral_indices(image: ee.Image) -> ee.Image:
    """
    Computes NDVI and NDWI indices from Sentinel-2 surface reflectance bands.
    B8 = NIR, B4 = Red, B3 = Green
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, ndwi])


def get_sentinel2_composite(
    roi: ee.Geometry, 
    start_date: str, 
    end_date: str
) -> ee.Image:
    """Retrieves, masks, and computes median composite over ROI for specified dates."""
    logger.info(f"Fetching Sentinel-2 SR Harmonized imagery from {start_date} to {end_date}...")
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_sentinel2_clouds)
    )
    
    composite = collection.map(compute_spectral_indices).median().clip(roi)
    return composite


# =============================================================================
# 2. FEATURE EXTRACTION & TRAINING DATA GENERATION
# =============================================================================

def generate_ground_truth_samples(
    composite: ee.Image, 
    roi: ee.Geometry, 
    num_samples: int = 2000
) -> ee.FeatureCollection:
    """
    Applies synthetic labeling based on spectral thresholds to construct ground-truth 
    training points across the Region of Interest.
    """
    ndvi = composite.select('NDVI')
    ndwi = composite.select('NDWI')
    
    # Define heuristic label logic:
    # Flood: High Water Index (NDWI > 0.1)
    # Drought: Low Vegetation Index (NDVI < 0.2) AND Low Water Index (NDWI < -0.1)
    # Normal: Remaining conditions
    label_image = ee.Image(CLASS_MAPPING["Normal"]) \
        .where(ndwi.gt(0.1), CLASS_MAPPING["Flood"]) \
        .where(ndvi.lt(0.2).And(ndwi.lt(-0.1)), CLASS_MAPPING["Drought"]) \
        .rename('label')

    sample_image = composite.select(['NDVI', 'NDWI']).addBands(label_image)
    
    samples = sample_image.stratifiedSample(
        numPoints=num_samples,
        classBand='label',
        region=roi,
        scale=30,
        geometries=True
    )
    return samples


def extract_features_to_dataframe(samples: ee.FeatureCollection) -> pd.DataFrame:
    """Extracts features from Earth Engine FeatureCollection to a local Pandas DataFrame."""
    logger.info("Downloading stratified samples to local memory...")
    samples_dict = samples.getInfo()['features']
    
    data = []
    for feat in samples_dict:
        props = feat['properties']
        data.append({
            'NDVI': props.get('NDVI'),
            'NDWI': props.get('NDWI'),
            'label': props.get('label')
        })
        
    df = pd.DataFrame(data).dropna()
    return df


# =============================================================================
# 3. MACHINE LEARNING MODEL DEVELOPMENT
# =============================================================================

def train_random_forest(
    df: pd.DataFrame
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Trains a Scikit-Learn Random Forest Classifier and computes performance metrics."""
    X = df[['NDVI', 'NDWI']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, target_names=[CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())], output_dict=True
    )
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    logger.info(f"Random Forest successfully trained. Validation Accuracy: {accuracy:.4f}")
    return clf, metrics


# =============================================================================
# 4. EARTH ENGINE CLASSIFICATION & INFERENCE
# =============================================================================

def apply_native_ee_classification(
    composite: ee.Image, 
    training_samples: ee.FeatureCollection
) -> ee.Image:
    """
    Trains a native Earth Engine Random Forest model using the sampled points 
    to perform server-side inference directly over the full raster scene.
    """
    classifier = ee.Classifier.smileRandomForest(100).train(
        features=training_samples,
        classProperty='label',
        inputProperties=['NDVI', 'NDWI']
    )
    classified_image = composite.select(['NDVI', 'NDWI']).classify(classifier)
    return classified_image


# =============================================================================
# 5. VISUALIZATION & DATA SUMMARY EXPORT
# =============================================================================

def generate_interactive_map(
    roi: ee.Geometry, 
    composite: ee.Image, 
    classified: ee.Image
) -> geemap.Map:
    """Generates a high-fidelity visual prediction map using geemap."""
    Map = geemap.Map()
    Map.centerObject(roi, 9)
    
    # Layer 1: True Color Composite
    Map.addLayer(
        composite, 
        {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3}, 
        'Sentinel-2 RGB'
    )
    
    # Layer 2: Predictions
    viz_params = {
        'min': 0,
        'max': 2,
        'palette': PALETTE
    }
    Map.addLayer(classified, viz_params, 'Drought/Flood Predictions')
    
    # Legend
    legend_dict = {
        'Drought': PALETTE[0],
        'Normal': PALETTE[1],
        'Flood': PALETTE[2]
    }
    Map.add_legend(title="Hazard Classification", legend_dict=legend_dict)
    
    return Map


def export_data_summary(
    classified: ee.Image, 
    roi: ee.Geometry, 
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates areal statistics for predictions and synthesizes execution metrics."""
    logger.info("Computing spatial area calculations across target region...")
    area_image = ee.Image.pixelArea().addBands(classified)
    
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(
            groupField=1,
            groupName='class'
        ),
        geometry=roi,
        scale=30,
        maxPixels=1e10
    ).getInfo()
    
    class_areas_sqkm = {}
    for item in stats.get('groups', []):
        class_idx = int(item['class'])
        label = CLASS_LABELS.get(class_idx, "Unknown")
        area_sqkm = item['sum'] / 1e6  # Convert square meters to Sq Kilometers
        class_areas_sqkm[label] = round(area_sqkm, 2)
        
    summary = {
        'model_performance': metrics,
        'spatial_distribution_sqkm': class_areas_sqkm
    }
    return summary


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run_pipeline(
    project_id: str, 
    roi: ee.Geometry, 
    start_date: str, 
    end_date: str
) -> Tuple[geemap.Map, Dict[str, Any]]:
    """Executes the full Drought and Flood Predictive AI pipeline end-to-end."""
    initialize_earth_engine(project_id=project_id)
    
    # Step 1: Preprocessing & Composite Generation
    composite = get_sentinel2_composite(roi, start_date, end_date)
    
    # Step 2: Sampling
    samples = generate_ground_truth_samples(composite, roi, num_samples=1500)
    
    # Step 3: Local ML Training
    df_samples = extract_features_to_dataframe(samples)
    rf_model, metrics = train_random_forest(df_samples)
    
    # Step 4: Server-Side EE Raster Classification
    classified_map = apply_native_ee_classification(composite, samples)
    
    # Step 5: Render Maps & Export Summaries
    map_object = generate_interactive_map(roi, composite, classified_map)
    summary_data = export_data_summary(classified_map, roi, metrics)
    
    return map_object, summary_data


if __name__ == "__main__":
    # Example Execution Parameters
    GOOGLE_PROJECT_ID = "your-gcp-project-id"  # Replace with valid GCP Project ID
    
    # Target ROI: Region vulnerable to seasonal flooding and dry spells
    TARGET_ROI = ee.Geometry.Polygon([
        [[13.80, 12.80], [13.80, 13.50], [14.80, 13.50], [14.80, 12.80]]
    ])
    
    START_PERIOD = "2023-06-01"
    END_PERIOD = "2023-10-31"
    
    try:
        prediction_map, summary = run_pipeline(
            project_id=GOOGLE_PROJECT_ID,
            roi=TARGET_ROI,
            start_date=START_PERIOD,
            end_date=END_PERIOD
        )
        
        logger.info("Pipeline executed successfully.")
        print("\n--- DATA SUMMARY STRUCT ---")
        print(json.dumps(summary, indent=2))
        
        # Export map to HTML for local rendering:
        # prediction_map.to_html("drought_flood_prediction.html")
        
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
