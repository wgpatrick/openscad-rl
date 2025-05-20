#!/usr/bin/env python3
"""
Inspect Alignment Script

This script loads two STL files (a source and a reference), aligns them
using the logic from reward_function.py, and visualizes the result.
It helps in debugging and understanding the alignment process.
"""

import argparse
import os
import sys
import logging
import numpy as np # For seeding

# Ensure the script can find reward_function.py if it's in the same directory
# or a directory in PYTHONPATH
try:
    from reward import _align_point_clouds, load_config, setup_logger, CONFIG, ConfigError
except ImportError:
    # If reward_function is in the same directory, try adding it to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        from reward import _align_point_clouds, load_config, setup_logger, CONFIG, ConfigError
    except ImportError as e:
        print(f"Error: Could not import from reward.py. Make sure it's in the same directory or your PYTHONPATH. Details: {e}")
        sys.exit(1)

try:
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
except ImportError:
    print("Error: Open3D library not found. Please ensure it's installed (e.g., pip install open3d).")
    sys.exit(1)
except AttributeError: # Handle cases where set_verbosity_level might not be available on older o3d
    pass


logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Visualize alignment of two STL files.")
    parser.add_argument("source_stl_path", type=str, help="Path to the source STL file (the one to be aligned).")
    parser.add_argument("reference_stl_path", type=str, help="Path to the reference STL file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file (default: config.yaml).")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper()))

    try:
        # Load configuration
        # The global CONFIG in reward_function will be populated by load_config
        loaded_cfg = load_config(args.config)
        if not loaded_cfg: # load_config raises error or returns None on failure
             logger.error(f"Failed to load configuration from {args.config}")
             return # Exit if config loading failed

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return

    # Validate input STL file paths
    if not os.path.exists(args.source_stl_path):
        logger.error(f"Source STL file not found: {args.source_stl_path}")
        return
    if not os.path.exists(args.reference_stl_path):
        logger.error(f"Reference STL file not found: {args.reference_stl_path}")
        return

    # Extract necessary parameters from config, with defaults if not present
    similarity_config = CONFIG.get('reward_thresholds', {})
    num_points = CONFIG.get('similarity_point_cloud_samples', 50000) # Match CadEval/reward_function
    voxel_size_fpfh_p = similarity_config.get('voxel_size_for_fpfh_mm', 0.5) # Example, ensure in config
    seed = CONFIG.get('similarity_random_seed', 42)

    logger.info(f"Using random seed: {seed} for alignment.")
    logger.info(f"Number of points for sampling: {num_points}")
    logger.info(f"Voxel size for FPFH: {voxel_size_fpfh_p}")


    # Set seeds for reproducibility - note: _align_point_clouds itself also sets seeds internally
    if seed is not None:
        np.random.seed(seed)
        o3d.utility.random.seed(seed)
        # Python's random module is also seeded in reward_function's check_shape_similarity
        # which _align_point_clouds is part of. If using _align_point_clouds directly,
        # consider if `random.seed(seed)` is needed here or if _align_point_clouds handles it.
        # For now, assume _align_point_clouds handles its internal Python random seeding if necessary.


    logger.info(f"Aligning source STL '{args.source_stl_path}' to reference STL '{args.reference_stl_path}'...")

    # Call the alignment function
    # _align_point_clouds expects paths, num_points, voxel_size_fpfh, seed
    aligned_source_pcd, reference_pcd, icp_fitness, details = _align_point_clouds(
        generated_stl_path=args.source_stl_path,
        reference_stl_path=args.reference_stl_path,
        num_points_for_sampling=num_points,
        voxel_size_fpfh=voxel_size_fpfh_p, # Pass the FPFH voxel size
        seed=seed
    )

    if aligned_source_pcd is None or reference_pcd is None:
        logger.error(f"Point cloud alignment failed. Details: {details}")
        return

    logger.info(f"Alignment successful. ICP Fitness: {icp_fitness:.4f}")
    logger.info(details)


    # Visualize the point clouds
    # Color the point clouds for better differentiation
    aligned_source_pcd.paint_uniform_color([1, 0.706, 0])  # Orange
    reference_pcd.paint_uniform_color([0, 0.651, 0.929])    # Blue

    logger.info("Displaying aligned source (Orange) and reference (Blue) point clouds...")
    o3d.visualization.draw_geometries([aligned_source_pcd, reference_pcd],
                                      window_name="Aligned Point Clouds",
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False) # Normals can clutter the view

    logger.info("Visualization window closed.")

if __name__ == "__main__":
    main() 