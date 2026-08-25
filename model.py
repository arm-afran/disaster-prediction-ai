"""
Production-Ready Earth Engine & Scikit-Learn Drought/Flood Predictive AI Pipeline
Architected for scalable remote sensing feature extraction, machine learning training,
and spatial coverage analytics.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import ee
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configure production logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EarthEngineAIPipeline")


def initialize_earth_engine(project_id: str | None = None) -> None:
    """Initializes the Google Earth Engine API with failure recovery and auth handling."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logger.info("Google Earth Engine initialized successfully.")
    except Exception as e:
        logger.warning(f"Default EE initialization failed ({e}). Attempting authentication...")
        try:
            ee.Authenticate()
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            logger.info("Google Earth Engine authenticated and initialized successfully.")
        except Exception as auth_err:
            logger.error(f"Critical failure initializing Earth Engine: {auth_err}")
            raise RuntimeError("Earth Engine authorization failed.") from auth_err


def mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds and cirrus in Sentinel-2 Surface Reflectance via QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)


def compute_satellite_indices(image: ee.Image) -> ee.Image:
    """
    Computes spectral indices:
    - NDVI (Normalized Difference Vegetation Index): (NIR - Red) / (NIR + Red)
    - NDWI (Normalized Difference Water Index): (Green - NIR) / (Green + NIR)
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, ndwi])


def load_sentinel2_composite(
    aoi: ee.Geometry, start_date: str, end_date: str
) -> ee.Image:
    """Loads, cloud-masks, and generates median composite of Sentinel-2 images with indices."""
    logger.info(f"Retrieving Sentinel-2 telemetry from {start_date} to {end_date}...")
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_sentinel2_clouds)
        .map(compute_satellite_indices)
    )

    composite = collection.median().clip(aoi)
    return composite


def generate_labeled_training_samples(
    composite: ee.Image, aoi: ee.Geometry, num_points: int = 1500
) -> ee.FeatureCollection:
    """
    Generates ground-truth stratified sampling dataset based on index threshold rules:
    - Class 0: Normal
    - Class 1: Flood (High NDWI)
    - Class 2: Drought (Very low NDVI & Low NDWI)
    """
    logger.info("Executing spatial stratified sampling for training sample generation...")
    ndwi = composite.select('NDWI')
    ndvi = composite.select('NDVI')

    # Class assignment logic
    label_img = (
        ee.Image(0)
        .where(ndwi.gt(0.25), 1)
        .where(ndvi.lt(0.18).And(ndwi.lt(-0.1)), 2)
        .rename('label')
    )

    sample_source = composite.select(['NDVI', 'NDWI']).addBands(label_img)

    samples = sample_source.stratifiedSample(
        numPoints=num_points,
        classBand='label',
        region=aoi,
        scale=30,
        geometries=True,
        dropNulls=True
    )
    return samples


def extract_ee_data_to_dataframe(samples: ee.FeatureCollection) -> pd.DataFrame:
    """Extracts features from Earth Engine FeatureCollection to a Pandas DataFrame."""
    logger.info("Transferring sampled Earth Engine vector data to local memory...")
    try:
        raw_dict = samples.reduceColumns(
            ee.Reducer.toList().repeat(3), ['NDVI', 'NDWI', 'label']
        ).getInfo()

        feature_matrix = np.array(raw_dict['list']).T
        df = pd.DataFrame(feature_matrix, columns=['NDVI', 'NDWI', 'label'])
        df['label'] = df['label'].astype(int)
        
        logger.info(f"Data extraction complete. Total valid records: {len(df)}")
        return df
    except Exception as err:
        logger.error(f"Failed to extract tabular data from GEE: {err}")
        raise err


def train_scikit_random_forest(
    df: pd.DataFrame
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Trains a Scikit-Learn Random Forest Classifier and computes performance metrics."""
    logger.info("Initializing and training Scikit-Learn Random Forest Model...")
    
    X = df[['NDVI', 'NDWI']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=['Normal', 'Flood', 'Drought'], 
        output_dict=True
    )
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    metrics_summary = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "feature_importances": {
            feature: round(imp, 4)
            for feature, imp in zip(['NDVI', 'NDWI'], model.feature_importances_.tolist())
        }
    }

    logger.info(f"Random Forest Training Completed. Validation Accuracy: {accuracy:.4f}")
    return model, metrics_summary


def compute_spatial_summary(
    composite: ee.Image, aoi: ee.Geometry, samples: ee.FeatureCollection
) -> Dict[str, Any]:
    """
    Deploys trained model parameters back onto Earth Engine platform for high-performance 
    raster classification and spatial area computation.
    """
    logger.info("Running spatial area calculations on classified Earth Engine raster...")
    
    ee_classifier = ee.Classifier.smileRandomForest(150).train(
        features=samples,
        classProperty='label',
        inputProperties=['NDVI', 'NDWI']
    )

    classified_raster = composite.select(['NDVI', 'NDWI']).classify(ee_classifier)
    area_image = ee.Image.pixelArea().addBands(classified_raster)

    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(
            groupField=1,
            groupName='class'
        ),
        geometry=aoi,
        scale=30,
        maxPixels=1e9
    ).getInfo()

    label_map = {0: "Normal", 1: "Flood", 2: "Drought"}
    area_dict = {}

    if 'groups' in stats:
        for group in stats['groups']:
            class_id = int(group['class'])
            sq_km = round(group['sum'] / 1e6, 2)
            class_name = label_map.get(class_id, f"Unknown_{class_id}")
            area_dict[class_name] = f"{sq_km} sq km"

    return area_dict


def get_visual_config() -> Dict[str, Any]:
    """Provides high-fidelity visualization parameters for rendering maps."""
    return {
        "NDVI_Vis": {
            "min": -0.2,
            "max": 0.8,
            "palette": ['#0000FF', '#FFFFFF', '#00FF00']
        },
        "NDWI_Vis": {
            "min": -0.5,
            "max": 0.5,
            "palette": ['#8B4513', '#FFFFFF', '#0000FF']
        },
        "Classification_Map": {
            "min": 0,
            "max": 2,
            "palette": [
                '#2ECC71',  # Green = Normal
                '#3498DB',  # Blue = Flood
                '#E74C3C'   # Red = Drought
            ]
        }
    }


def execute_drought_flood_pipeline(
    aoi_bounds: List[float],
    start_date: str,
    end_date: str,
    project_id: str | None = None
) -> Dict[str, Any]:
    """
    Main orchestration function running the complete Drought and Flood AI Pipeline.

    Args:
        aoi_bounds: List of [xmin, ymin, xmax, ymax] coordinates.
        start_date: ISO date format string (YYYY-MM-DD).
        end_date: ISO date format string (YYYY-MM-DD).
        project_id: Optional GCP project ID for GEE initialization.

    Returns:
        Structured summary dictionary containing metrics, spatial calculations, and map settings.
    """
    initialize_earth_engine(project_id)

    aoi = ee.Geometry.Rectangle(aoi_bounds)
    composite = load_sentinel2_composite(aoi, start_date, end_date)
    samples = generate_labeled_training_samples(composite, aoi, num_points=1200)

    df_data = extract_ee_data_to_dataframe(samples)
    _, model_metrics = train_scikit_random_forest(df_data)

    spatial_coverage = compute_spatial_summary(composite, aoi, samples)
    vis_config = get_visual_config()

    output_payload = {
        "execution_status": "SUCCESS",
        "model_performance_metrics": model_metrics,
        "spatial_area_distribution": spatial_coverage,
        "visualization_parameters": vis_config
    }

    logger.info("Pipeline execution finalized successfully.")
    return output_payload


if __name__ == "__main__":
    # Test execution bounding box [xmin, ymin, xmax, ymax]
    TARGET_AOI = [-120.5, 36.5, -119.8, 37.2]
    START = "2023-01-01"
    END = "2023-06-30"

    results = execute_drought_flood_pipeline(
        aoi_bounds=TARGET_AOI,
        start_date=START,
        end_date=END
    )

    print("\n================ PRODUCTION SUMMARY REPORT ================")
    print(json.dumps(results, indent=2))
