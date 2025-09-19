#!/usr/bin/env python3

"""
Script to extract ground truth detections within 200 metres of shoreline.
Based on the SARFish reference code in ../reference/SARFish_metric.py

This script analyzes the ground truth detection data and uses the shoreline
distance information that's already computed in the CSV labels to filter
detections within 200m of shore.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

def load_ground_truth_data(csv_path: str) -> pd.DataFrame:
    """Load ground truth detection data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} detections from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()

def filter_detections_by_shore_distance(df: pd.DataFrame,
                                      distance_threshold_m: float = 200.0,
                                      shoreline_type: str = "xView3_shoreline") -> pd.DataFrame:
    """
    Filter detections based on distance from shore.

    Args:
        df: DataFrame with ground truth detections
        distance_threshold_m: Distance threshold in meters (default: 200m)
        shoreline_type: Type of shoreline data to use ("xView3_shoreline" or "global_shoreline_vector")

    Returns:
        Filtered DataFrame with detections within distance_threshold_m of shore
    """

    # Convert distance threshold from meters to kilometers (CSV data is in km)
    distance_threshold_km = distance_threshold_m / 1000.0

    # Determine the column name based on shoreline type
    if shoreline_type == "xView3_shoreline":
        distance_column = "xView3_shoreline_distance_from_shore_km"
    elif shoreline_type == "global_shoreline_vector":
        distance_column = "global_shoreline_vector_distance_from_shore_km"
    else:
        raise ValueError(f"Unknown shoreline_type: {shoreline_type}")

    # Check if the distance column exists
    if distance_column not in df.columns:
        print(f"Warning: Column '{distance_column}' not found in data")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Filter detections within the distance threshold
    shore_detections = df[df[distance_column] <= distance_threshold_km].copy()

    print(f"Found {len(shore_detections)} detections within {distance_threshold_m}m of shoreline")
    print(f"  (using {shoreline_type} distance data)")

    return shore_detections

def analyze_shore_detections(df: pd.DataFrame, distance_threshold_m: float = 200.0) -> Dict:
    """Analyze the filtered shore detections and return summary statistics."""

    if len(df) == 0:
        return {"total": 0}

    analysis = {
        "total": len(df),
        "distance_threshold_m": distance_threshold_m
    }

    # Confidence level breakdown
    if 'detect_scene_confidence' in df.columns:
        conf_counts = df['detect_scene_confidence'].value_counts()
        analysis['confidence_breakdown'] = conf_counts.to_dict()

    # Vessel type breakdown
    if 'is_vessel' in df.columns:
        vessel_counts = df['is_vessel'].value_counts()
        analysis['vessel_breakdown'] = vessel_counts.to_dict()

        # If we have fishing vessel info
        if 'is_fishing' in df.columns:
            fishing_counts = df[df['is_vessel'] == True]['is_fishing'].value_counts()
            analysis['fishing_breakdown'] = fishing_counts.to_dict()

    # Scene breakdown
    if 'scene_id' in df.columns:
        scene_counts = df['scene_id'].value_counts()
        analysis['scenes_with_shore_detections'] = len(scene_counts)
        analysis['detections_per_scene'] = scene_counts.to_dict()

    return analysis

def save_results(shore_detections: pd.DataFrame, output_path: str, analysis: Dict):
    """Save the filtered detections and analysis to files."""

    if len(shore_detections) > 0:
        # Save the filtered detections
        shore_detections.to_csv(output_path, index=False)
        print(f"Saved {len(shore_detections)} shore detections to {output_path}")

        # Save analysis summary
        analysis_path = output_path.replace('.csv', '_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write(f"Shore Detection Analysis (within {analysis['distance_threshold_m']}m)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total detections near shore: {analysis['total']}\n\n")

            if 'confidence_breakdown' in analysis:
                f.write("Confidence Level Breakdown:\n")
                for conf, count in analysis['confidence_breakdown'].items():
                    f.write(f"  {conf}: {count}\n")
                f.write("\n")

            if 'vessel_breakdown' in analysis:
                f.write("Vessel Type Breakdown:\n")
                for vessel_type, count in analysis['vessel_breakdown'].items():
                    f.write(f"  Is Vessel {vessel_type}: {count}\n")
                f.write("\n")

            if 'fishing_breakdown' in analysis:
                f.write("Fishing Vessel Breakdown (among vessels):\n")
                for fishing_type, count in analysis['fishing_breakdown'].items():
                    f.write(f"  Is Fishing {fishing_type}: {count}\n")
                f.write("\n")

            if 'scenes_with_shore_detections' in analysis:
                f.write(f"Number of scenes with shore detections: {analysis['scenes_with_shore_detections']}\n\n")

                f.write("Detections per scene:\n")
                for scene, count in analysis['detections_per_scene'].items():
                    f.write(f"  {scene}: {count}\n")

        print(f"Saved analysis summary to {analysis_path}")
    else:
        print("No detections found within the specified distance threshold")

def main():
    parser = argparse.ArgumentParser(
        description="Filter ground truth detections within specified distance of shoreline"
    )
    parser.add_argument("--csv-path", required=True,
                       help="Path to ground truth CSV file (e.g., SLC_validation.csv)")
    parser.add_argument("--distance", type=float, default=200.0,
                       help="Distance threshold in meters (default: 200m)")
    parser.add_argument("--shoreline-type", choices=["xView3_shoreline", "global_shoreline_vector"],
                       default="xView3_shoreline",
                       help="Type of shoreline data to use")
    parser.add_argument("--output",
                       help="Output CSV path (default: based on input filename)")

    args = parser.parse_args()

    # Load ground truth data
    ground_truth = load_ground_truth_data(args.csv_path)
    if len(ground_truth) == 0:
        print("No data loaded, exiting")
        return

    # Filter detections by shore distance
    shore_detections = filter_detections_by_shore_distance(
        ground_truth,
        distance_threshold_m=args.distance,
        shoreline_type=args.shoreline_type
    )

    # Analyze the results
    analysis = analyze_shore_detections(shore_detections, args.distance)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.csv_path)
        output_path = input_path.parent / f"{input_path.stem}_shore_{int(args.distance)}m.csv"

    # Save results
    save_results(shore_detections, str(output_path), analysis)

    # Print summary
    print(f"\nSummary:")
    print(f"  Input detections: {len(ground_truth)}")
    print(f"  Shore detections (≤{args.distance}m): {len(shore_detections)}")
    print(f"  Percentage: {100*len(shore_detections)/len(ground_truth):.1f}%")

if __name__ == "__main__":
    main()