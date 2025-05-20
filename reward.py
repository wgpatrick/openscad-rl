#!/usr/bin/env python3
"""
Reward Function for OpenSCAD-based Reinforcement Learning

This script defines a reward function that:
1. Takes OpenSCAD code as input.
2. Renders it to an STL file using the OpenSCAD CLI.
3. Performs geometric analysis (volume, Chamfer distance, Hausdorff distance)
   by comparing the generated STL to a reference STL.
4. Returns a reward of 1 if all steps are successful and metrics are within
   thresholds defined in config.yaml, otherwise returns 0.
"""

import os
import sys
import subprocess
import time
import json
import glob
import logging
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import yaml # For config loading
import re # Moved import re to the top
import argparse # Add argparse import
import random # Add this import at the top of the file

# Attempt to import geometry libraries
try:
    import trimesh
except ImportError:
    print("Error: Trimesh library not found. Please install using: pip install trimesh[easy]")
    sys.exit(1)

try:
    import open3d as o3d
    # Suppress Open3D informational messages if possible
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
except ImportError:
    print("Error: Open3D library not found. Please ensure it's installed in your environment (e.g., pip install open3d).")
    sys.exit(1)
except AttributeError: # Handle cases where set_verbosity_level might not be available on older o3d
    pass


# --- Global Variables & Configuration ---
CONFIG: Dict[str, Any] = {}
TASKS_DATA: List[Dict[str, str]] = [] # Cache for tasks.json
DEFAULT_LOG_LEVEL = logging.INFO
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class RewardFunctionError(Exception):
    """Base exception for errors in the reward function."""
    pass

class RenderError(RewardFunctionError):
    """Exception raised for errors during OpenSCAD rendering."""
    pass

class GeometryCheckError(RewardFunctionError):
    """Exception raised for errors during geometry checks."""
    pass

class ConfigError(RewardFunctionError):
    """Exception raised for configuration errors."""
    pass


# --- Logger Setup ---
def setup_logger(name: str, level: Union[int, str] = DEFAULT_LOG_LEVEL, log_file: Optional[str] = None) -> logging.Logger:
    """Sets up a logger instance."""
    local_logger = logging.getLogger(name)
    local_logger.setLevel(level)
    local_logger.propagate = False # To avoid duplicate messages if root logger is configured

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout) # Changed from sys.stderr to sys.stdout
    ch.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in local_logger.handlers):
        local_logger.addHandler(ch)

    # File Handler (Optional)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a') # Append mode
        fh.setFormatter(formatter)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in local_logger.handlers):
            local_logger.addHandler(fh)
    return local_logger

# --- Configuration Loader ---
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    global CONFIG
    abs_config_path_found = None # Initialize for use in error messages
    try:
        # Enhanced path finding:
        # 1. If config_path is absolute, use it.
        # 2. If relative, try CWD first (common for running scripts/tests from project root).
        # 3. Then try relative to this script file (reward.py).
        
        possible_paths = []
        if os.path.isabs(config_path):
            possible_paths.append(config_path)
        else:
            possible_paths.append(os.path.abspath(config_path)) # Relative to CWD
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths.append(os.path.join(script_dir, config_path)) # Relative to script

        for p in possible_paths:
            if os.path.exists(p):
                abs_config_path_found = p
                logger.debug(f"Attempting to load configuration from: {p}")
                break
        
        if not abs_config_path_found:
            # Fallback for the original logic if the above doesn't find it, just in case.
            # This part might be redundant now but kept for safety / if user relies on old behavior.
            script_dir_fallback = os.path.dirname(os.path.abspath(__file__))
            original_attempt_path = os.path.join(script_dir_fallback, config_path)
            if os.path.exists(original_attempt_path):
                abs_config_path_found = original_attempt_path
            elif os.path.exists(os.path.abspath(config_path)): # CWD as last direct resort
                 abs_config_path_found = os.path.abspath(config_path)

        if not abs_config_path_found:
            raise ConfigError(f"Configuration file '{config_path}' not found. Searched paths based on CWD and script location: {possible_paths}")

        with open(abs_config_path_found, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        if not loaded_data: # Check if loaded_data is None or empty (e.g. empty file)
            raise ConfigError(f"Configuration file {abs_config_path_found} is empty or invalid YAML.")

        CONFIG.clear()  # Clear the existing global CONFIG dictionary
        CONFIG.update(loaded_data)  # Update it in-place with the new data

        logger.info(f"Successfully loaded configuration into global CONFIG from: {abs_config_path_found}")

        # Resolve reference_stl_file path relative to the config file's directory if it's relative
        # This part is less relevant for tests using _calculate_reward_core directly with an absolute path,
        # but good for the main script or geometric_reward if it were to use a global reference.
        if 'reference_stl_file' in CONFIG and not os.path.isabs(CONFIG['reference_stl_file']):
            config_dir = os.path.dirname(abs_config_path_found) # Use the found config path
            CONFIG['reference_stl_file'] = os.path.normpath(os.path.join(config_dir, CONFIG['reference_stl_file']))
            logger.info(f"Resolved reference_stl_file to: {CONFIG['reference_stl_file']}")

        # Basic validation for required keys (can be expanded)
        if 'openscad' not in CONFIG or 'executable_path' not in CONFIG['openscad']:
            raise ConfigError("Missing 'openscad.executable_path' in configuration.")
        if 'reward_thresholds' not in CONFIG:
            raise ConfigError("Missing 'reward_thresholds' section in configuration.")
        
        return CONFIG
        
    except FileNotFoundError: # Should be caught by explicit checks, but as a safeguard
        # This specific exception might not be hit if ConfigError is raised first.
        raise ConfigError(f"Configuration file not found (FileNotFound): {config_path}")
    except yaml.YAMLError as e:
        path_for_error = abs_config_path_found if abs_config_path_found else config_path
        raise ConfigError(f"Error parsing YAML configuration file {path_for_error}: {e}")
    except ConfigError as e: # Re-raise ConfigErrors to ensure they are not caught by the generic Exception
        raise
    except Exception as e:
        path_for_error = abs_config_path_found if abs_config_path_found else config_path
        # Catch other potential errors, e.g., if CONFIG.update fails due to loaded_data not being a dict.
        raise ConfigError(f"An unexpected error occurred while loading/processing configuration from {path_for_error}: {e}")


# --- Task Data Loader ---
def load_tasks_data(tasks_json_path: str = "tasks/tasks.json") -> List[Dict[str, str]]:
    """Loads task definitions from a JSON file."""
    global TASKS_DATA
    if TASKS_DATA: # Return cached version if already loaded
        return TASKS_DATA
    
    try:
        # Try to find tasks.json relative to this script file first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_tasks_path = os.path.join(script_dir, tasks_json_path)

        if not os.path.exists(abs_tasks_path):
            # If not found, try relative to current working directory
            abs_tasks_path = os.path.abspath(tasks_json_path)
            if not os.path.exists(abs_tasks_path):
                logger.error(f"Tasks JSON file not found at {tasks_json_path} (tried relative to script and CWD).")
                return [] # Return empty list, geometric_reward will handle this

        with open(abs_tasks_path, 'r') as f:
            TASKS_DATA = json.load(f)
        logger.info(f"Successfully loaded tasks data from: {abs_tasks_path}")
        return TASKS_DATA
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing tasks JSON file {abs_tasks_path}: {e}")
        TASKS_DATA = [] # Reset on error
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading tasks data: {e}")
        TASKS_DATA = [] # Reset on error
        return []

# --- Placeholder for OpenSCAD Rendering Logic (to be adapted from render_scad.py) ---
def validate_openscad_config(openscad_config: Dict[str, Any]) -> str:
    """Validates OpenSCAD configuration (executable path and version)."""
    logger.info("Validating OpenSCAD configuration...")
    executable_path = openscad_config.get('executable_path')
    if not executable_path:
        raise ConfigError("Missing required configuration key: 'openscad.executable_path'")

    minimum_version_str = openscad_config.get('minimum_version', '2021.01')

    # Simplified check: just ensure the executable path exists
    if not os.path.exists(executable_path) and not shutil.which(executable_path):
         # Try finding in common locations if it's just 'openscad' or 'openscad-nightly'
        common_paths_to_check = []
        if sys.platform == "darwin": # macOS
            common_paths_to_check.extend([
                "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
                "/usr/local/bin/openscad"
            ])
        elif sys.platform.startswith("linux"): # Linux
            common_paths_to_check.extend([
                "/usr/bin/openscad",
                "/snap/bin/openscad"
            ])
        elif sys.platform == "win32": # Windows
            # On Windows, 'openscad' might be in PATH, or in Program Files
            # shutil.which should handle PATH. Explicit checks are harder.
            pass

        found_common = False
        if not shutil.which(executable_path): # If not in PATH
            for p_to_check in common_paths_to_check:
                if os.path.exists(p_to_check):
                    logger.warning(f"OpenSCAD executable not found at '{executable_path}', but found at '{p_to_check}'. Updating config for this session.")
                    openscad_config['executable_path'] = p_to_check # Update in-memory config
                    executable_path = p_to_check
                    found_common = True
                    break
            if not found_common:
                 raise RenderError(f"OpenSCAD executable not found at specified path: {executable_path}, and not found in PATH or common locations.")
        else: # Found in PATH by shutil.which
            executable_path = shutil.which(executable_path)
            openscad_config['executable_path'] = executable_path # Store the full path
            logger.info(f"OpenSCAD executable found in PATH: {executable_path}")


    logger.info(f"OpenSCAD executable path set to: {executable_path}")

    try:
        process = subprocess.run(
            [executable_path, "--version"],
            capture_output=True, text=True, timeout=10, check=True
        )
        version_output = process.stderr.strip() or process.stdout.strip()
        # Example: "OpenSCAD version 2023.08.18" or "OpenSCAD version 2024.01.31.ai12345"
        match = re.search(r"(\d{4}\.\d{2}(?:\.\d{2})?(?:\.\w+)?)", version_output)
        if not match:
            raise RenderError(f"Could not parse OpenSCAD version from output: {version_output}")

        version_str_full = match.group(1)
        # For comparison, use only major.minor (e.g., 2021.01 from 2021.01.12 or 2021.01.dev)
        version_str_compare_parts = version_str_full.split('.')[:2]
        version_str_compare = ".".join(version_str_compare_parts)

        logger.info(f"Detected OpenSCAD version: {version_str_full} (comparing as {version_str_compare})")

        # Simple string comparison works for YYYY.MM format if consistent
        if version_str_compare < minimum_version_str:
            raise RenderError(
                f"Detected OpenSCAD version {version_str_full} is below minimum required {minimum_version_str}."
            )
        logger.info(f"OpenSCAD version meets minimum requirement ({minimum_version_str}).")
        return version_str_full
    except FileNotFoundError:
        raise RenderError(f"OpenSCAD executable check failed (not found?): {executable_path}")
    except subprocess.TimeoutExpired:
        raise RenderError("Checking OpenSCAD version timed out.")
    except subprocess.CalledProcessError as e:
        raise RenderError(f"Checking OpenSCAD version failed (Exit Code {e.returncode}). Stderr: {e.stderr}")
    except Exception as e:
        raise RenderError(f"Unexpected error during OpenSCAD version check: {e}")


def _build_openscad_command(
    scad_path: str,
    stl_path: str,
    openscad_config: Dict[str, Any]
) -> List[str]:
    """Builds the OpenSCAD command list."""
    executable_path = openscad_config.get('executable_path')
    if not executable_path:
         raise ConfigError("Missing 'openscad.executable_path' in configuration for building command.")

    export_format = openscad_config.get('export_format', 'asciistl').lower()
    if export_format not in ['asciistl', 'stl', 'binarystl']:
         logger.warning(f"Unsupported openscad.export_format '{export_format}'. Defaulting to 'asciistl'.")
         export_format = 'asciistl'
    if export_format == 'binarystl': # OpenSCAD CLI uses 'stl' for binary
        export_format = 'stl'

    command = [
        executable_path,
        "-o", stl_path,
        scad_path,
        "--export-format", export_format,
    ]
    # Add other options from config if needed, e.g., --camera, --viewall, features, etc.
    # For the reward function, a simple render is likely sufficient.
    return command


def render_openscad_code_to_stl(
    openscad_code: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Renders OpenSCAD code string to an STL file.

    Args:
        openscad_code: String containing the OpenSCAD code.
        output_dir: Directory to save the temporary .scad and output .stl file.
        config: The application configuration dictionary.

    Returns:
        Absolute path to the generated .stl file if successful, None otherwise.
    """
    openscad_cfg = config.get('openscad', {})
    scad_filename = "temp_design.scad"
    stl_filename = "temp_design.stl"
    
    scad_path = os.path.join(output_dir, scad_filename)
    stl_path = os.path.join(output_dir, stl_filename)

    try:
        # Write the OpenSCAD code to a temporary .scad file
        with open(scad_path, "w") as f:
            f.write(openscad_code)
        logger.debug(f"Temporary SCAD file written to: {scad_path}")

        command = _build_openscad_command(scad_path, stl_path, openscad_cfg)
        logger.info(f"Executing OpenSCAD command: {' '.join(command)}")

        timeout_seconds = openscad_cfg.get('render_timeout_seconds', 60)
        start_time = time.time()

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False # We'll check returncode manually
        )
        duration = time.time() - start_time

        if process.returncode == 0:
            if os.path.exists(stl_path) and os.path.getsize(stl_path) > 0:
                logger.info(f"OpenSCAD render successful for temp_design.scad in {duration:.2f}s. STL: {stl_path}")
                return stl_path
            else:
                # Check if there's an error message from OpenSCAD even with exit code 0
                err_msg = process.stderr.strip() if process.stderr else "STL file not created or is empty despite exit code 0."
                # Filter common OpenSCAD info/warnings from being critical errors if they don't prevent output
                if "Deprecation warning" in err_msg or "Thrown from" in err_msg: # Common warnings
                    if os.path.exists(stl_path) and os.path.getsize(stl_path) > 0:
                        logger.warning(f"OpenSCAD render for temp_design.scad completed with warnings but STL was generated ({duration:.2f}s). Stderr: {err_msg}")
                        return stl_path
                raise RenderError(f"OpenSCAD completed (code 0) but output STL missing or empty. Stderr: {err_msg}")
        else:
            raise RenderError(f"OpenSCAD failed (Code {process.returncode}). Duration: {duration:.2f}s. Stderr: {process.stderr.strip()}")

    except subprocess.TimeoutExpired:
        logger.error(f"OpenSCAD render timed out after {timeout_seconds} seconds for temp_design.scad.")
        raise RenderError(f"OpenSCAD render timed out after {timeout_seconds}s.")
    except RenderError as e: # Re-raise known render errors
        logger.error(f"RenderError: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during OpenSCAD rendering: {e}", exc_info=True)
        raise RenderError(f"Unexpected rendering error: {e}")
    finally:
        # Clean up the temporary .scad file, .stl is cleaned up by the caller or in main reward func
        # if os.path.exists(scad_path):
        #     try:
        #         os.remove(scad_path)
        #         logger.debug(f"Cleaned up temporary SCAD file: {scad_path}")
        #     except OSError as e:
        #         logger.warning(f"Could not remove temporary SCAD file {scad_path}: {e}")
        pass # SCAD file is cleaned up with the temp directory

# --- Placeholder for Geometry Checking Logic (to be adapted from geometry_check.py) ---

def _clean_mesh_for_checks(mesh: o3d.geometry.TriangleMesh, mesh_name: str) -> Optional[o3d.geometry.TriangleMesh]:
    """Applies cleaning steps to an Open3D mesh. Returns cleaned mesh or None if it becomes invalid."""
    try:
        if not mesh.has_triangles():
            logger.debug(f"Mesh {mesh_name} has no triangles before cleaning.")
            return None
        initial_vertices = len(mesh.vertices)
        initial_triangles = len(mesh.triangles)
        
        mesh.merge_close_vertices(0.0001)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        
        final_vertices = len(mesh.vertices)
        final_triangles = len(mesh.triangles)
        
        if not mesh.has_triangles():
            logger.warning(f"Mesh {mesh_name} became empty after cleaning operations.")
            return None
            
        if initial_vertices != final_vertices or initial_triangles != final_triangles:
            logger.debug(f"Cleaned mesh {mesh_name}: Vertices {initial_vertices}->{final_vertices}, Triangles {initial_triangles}->{final_triangles}")
        return mesh
    except Exception as e:
        logger.warning(f"Error during mesh cleaning for {mesh_name}: {e}")
        return mesh # Return original on error, or None if it was already bad


def check_volume_similarity(
    generated_stl_path: str,
    reference_stl_path: str,
    threshold_percent: float
) -> Tuple[bool, Optional[str]]:
    """
    Compares the volume of the generated mesh against the reference mesh.
    Returns (passed_check, error_message_or_none)
    """
    try:
        gen_mesh_trimesh = trimesh.load_mesh(generated_stl_path, force='mesh')
        ref_mesh_trimesh = trimesh.load_mesh(reference_stl_path, force='mesh')

        if not gen_mesh_trimesh or not hasattr(gen_mesh_trimesh, 'is_watertight'):
            return False, f"Generated mesh {os.path.basename(generated_stl_path)} could not be loaded by trimesh or is invalid."
        if not ref_mesh_trimesh or not hasattr(ref_mesh_trimesh, 'is_watertight'):
            return False, f"Reference mesh {os.path.basename(reference_stl_path)} could not be loaded by trimesh or is invalid."

        if not gen_mesh_trimesh.is_watertight:
            logger.warning(f"Generated mesh {os.path.basename(generated_stl_path)} is not watertight. Volume calculation may be inaccurate.")
            # Depending on strictness, could return False here. For now, proceed but log.
            # return False, "Generated mesh is not watertight."
        if not ref_mesh_trimesh.is_watertight:
            logger.warning(f"Reference mesh {os.path.basename(reference_stl_path)} is not watertight. Volume calculation may be inaccurate.")
            # return False, "Reference mesh is not watertight."


        gen_vol = gen_mesh_trimesh.volume
        ref_vol = ref_mesh_trimesh.volume
        
        logger.info(f"Volume Check: Generated Vol={gen_vol:.4f} mm^3, Reference Vol={ref_vol:.4f} mm^3")

        if ref_vol == 0: # Avoid division by zero
            if gen_vol == 0:
                logger.info("Volume check passed: Both reference and generated volumes are zero.")
                return True, None
            else:
                msg = f"Reference volume is zero, but generated volume is {gen_vol:.4f} mm^3."
                logger.warning(f"Volume check failed: {msg}")
                return False, msg
        
        volume_diff_percent = (abs(gen_vol - ref_vol) / abs(ref_vol)) * 100

        if volume_diff_percent <= threshold_percent:
            logger.info(f"Volume check passed: Difference = {volume_diff_percent:.2f}% (Threshold <= {threshold_percent:.2f}%)")
            return True, None
        else:
            msg = f"Volume difference {volume_diff_percent:.2f}% exceeds threshold {threshold_percent:.2f}%."
            logger.warning(f"Volume check failed: {msg}")
            return False, msg

    except trimesh.TrimeshError as e: # Catch specific trimesh errors if any
        logger.error(f"Trimesh error during volume check: {e}", exc_info=True)
        return False, f"Trimesh error during volume check: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during volume check: {e}", exc_info=True)
        return False, f"Unexpected error during volume check: {e}"


def _preprocess_pcd_for_similarity(pcd: o3d.geometry.PointCloud, voxel_size: float) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Any]]:
    """Downsamples, estimates normals, computes FPFH features for a point cloud."""
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if not pcd_down.has_points():
            logger.warning("PCD downsampling resulted in zero points.")
            return None, None
        
        # Estimate normals
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=35))
        if not pcd_down.has_normals():
            logger.warning("PCD normal estimation failed after downsampling.")
            # Proceed to FPFH if desired, or return None, None
        
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        if fpfh is None or not hasattr(fpfh, 'data') or fpfh.data.shape[1] == 0:
            logger.warning("FPFH computation failed or produced empty features.")
            return pcd_down, None # Return downsampled cloud even if features fail
            
        return pcd_down, fpfh
    except Exception as e:
        logger.error(f"Error during PCD preprocessing for similarity: {e}", exc_info=True)
        return None, None


def _align_point_clouds(
    generated_stl_path: str,
    reference_stl_path: str,
    num_points_for_sampling: int,
    voxel_size_fpfh: float,
    seed: Optional[int]
) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[o3d.geometry.PointCloud], Optional[float], str]:
    """
    Loads two STL files, samples point clouds, and aligns them using RANSAC + ICP.
    Returns (reference_pcd, generated_pcd_aligned, icp_fitness, error_message_or_none).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            o3d.utility.random.seed(seed)
        except AttributeError:
            logger.debug("o3d.utility.random.seed not available. Relying on numpy/random seeds.")
        except Exception as e_seed:
            logger.warning(f"Error setting Open3D seed: {e_seed}")

    try:
        gen_mesh_o3d = o3d.io.read_triangle_mesh(generated_stl_path)
        ref_mesh_o3d = o3d.io.read_triangle_mesh(reference_stl_path)

        if not gen_mesh_o3d.has_triangles():
            return None, None, None, "Generated STL for alignment is empty or invalid."
        if not ref_mesh_o3d.has_triangles():
            return None, None, None, "Reference STL for alignment is empty or invalid."

        if not gen_mesh_o3d.has_vertex_normals(): gen_mesh_o3d.compute_vertex_normals()
        if not ref_mesh_o3d.has_vertex_normals(): ref_mesh_o3d.compute_vertex_normals()
        
        gen_pcd_orig = gen_mesh_o3d.sample_points_uniformly(number_of_points=num_points_for_sampling)
        ref_pcd_orig = ref_mesh_o3d.sample_points_uniformly(number_of_points=num_points_for_sampling)

        if not gen_pcd_orig.has_points() or not ref_pcd_orig.has_points():
            return ref_pcd_orig, gen_pcd_orig, None, "Point cloud sampling resulted in zero points."

        gen_pcd_down, fpfh_gen = _preprocess_pcd_for_similarity(gen_pcd_orig, voxel_size_fpfh)
        ref_pcd_down, fpfh_ref = _preprocess_pcd_for_similarity(ref_pcd_orig, voxel_size_fpfh)

        if not all([gen_pcd_down, fpfh_gen, ref_pcd_down, fpfh_ref]):
            return ref_pcd_orig, gen_pcd_orig, None, "PCD preprocessing for RANSAC failed."

        ransac_distance_thresh = voxel_size_fpfh * 1.5
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=gen_pcd_down, target=ref_pcd_down,
            source_feature=fpfh_gen, target_feature=fpfh_ref,
            mutual_filter=True,
            max_correspondence_distance=ransac_distance_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4, checkers=[],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        T_ransac = ransac_result.transformation
        logger.debug(f"RANSAC Global Registration Fitness: {ransac_result.fitness:.4f}, Inlier RMSE: {ransac_result.inlier_rmse:.4f}")

        gen_pcd_globally_aligned = o3d.geometry.PointCloud(gen_pcd_orig) # Use a copy
        gen_pcd_globally_aligned.transform(T_ransac)

        icp_refinement_threshold = 1.5
        icp_result = o3d.pipelines.registration.registration_icp(
            source=gen_pcd_globally_aligned, target=ref_pcd_orig,
            max_correspondence_distance=icp_refinement_threshold,
            init=np.identity(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        final_fitness = icp_result.fitness
        logger.info(f"ICP Refinement Fitness: {final_fitness:.4f}, Inlier RMSE: {icp_result.inlier_rmse:.4f}")
        
        gen_pcd_final_aligned = o3d.geometry.PointCloud(gen_pcd_orig) # Use a fresh copy of original generated pcd
        gen_pcd_final_aligned.transform(icp_result.transformation @ T_ransac) # Apply combined transform

        return ref_pcd_orig, gen_pcd_final_aligned, final_fitness, None

    except Exception as e:
        logger.error(f"Unexpected error during point cloud alignment: {e}", exc_info=True)
        return None, None, None, f"Unexpected error during alignment: {e}"


def check_shape_similarity(
    generated_stl_path: str,
    reference_stl_path: str,
    thresholds: Dict[str, float],
    num_points_for_sampling: int = 20000,
    seed: Optional[int] = 42
) -> Tuple[bool, Optional[str]]:
    """
    Performs shape similarity checks (Chamfer, Hausdorff) using point cloud comparison
    after RANSAC + ICP alignment. Closely matches CadEval/scripts/geometry_check.py parameters.
    Returns (passed_all_checks, error_message_or_none_if_passed)
    """
    VOXEL_SIZE_FPFH = 5.0

    ref_pcd, gen_pcd_aligned, final_fitness, align_error_msg = _align_point_clouds(
        generated_stl_path,
        reference_stl_path,
        num_points_for_sampling,
        VOXEL_SIZE_FPFH,
        seed
    )

    if align_error_msg:
        logger.error(f"Shape similarity check failed due to alignment error: {align_error_msg}")
        return False, align_error_msg
    
    if not ref_pcd or not gen_pcd_aligned or not ref_pcd.has_points() or not gen_pcd_aligned.has_points():
        return False, "Alignment resulted in invalid or empty point clouds."

    if final_fitness is None: # Should be caught by align_error_msg but as a safeguard
        return False, "Alignment process did not return a fitness score."

    dist_gen_to_ref = np.asarray(gen_pcd_aligned.compute_point_cloud_distance(ref_pcd))
    dist_ref_to_gen = np.asarray(ref_pcd.compute_point_cloud_distance(gen_pcd_aligned))
    
    if dist_gen_to_ref.size == 0 or dist_ref_to_gen.size == 0:
        return False, "Distance calculation for Chamfer resulted in empty arrays."
        
    chamfer_distance = (np.mean(dist_gen_to_ref) + np.mean(dist_ref_to_gen)) / 2.0
    logger.info(f"Chamfer Distance: {chamfer_distance:.4f} mm")

    all_distances = np.concatenate((dist_gen_to_ref, dist_ref_to_gen))
    if all_distances.size == 0:
            return False, "Distance calculation for Hausdorff resulted in empty arrays."
    hausdorff_95p = np.percentile(all_distances, 95)
    logger.info(f"Hausdorff Distance (95th Percentile): {hausdorff_95p:.4f} mm")

    chamfer_ok = chamfer_distance <= thresholds['chamfer_mm']
    hausdorff_ok = hausdorff_95p <= thresholds['hausdorff_mm']
    fitness_ok = final_fitness >= thresholds['icp_fitness']

    error_messages = []
    if not chamfer_ok:
        error_messages.append(f"Chamfer distance {chamfer_distance:.4f}mm > threshold {thresholds['chamfer_mm']:.4f}mm")
    if not hausdorff_ok:
        error_messages.append(f"Hausdorff 95p distance {hausdorff_95p:.4f}mm > threshold {thresholds['hausdorff_mm']:.4f}mm")
    if not fitness_ok:
        error_messages.append(f"ICP fitness {final_fitness:.4f} < threshold {thresholds['icp_fitness']:.4f}")
    
    if not error_messages:
        logger.info("Shape similarity checks (Chamfer, Hausdorff, ICP Fitness) PASSED.")
        return True, None
    else:
        full_error_msg = "Shape similarity FAILED: " + "; ".join(error_messages)
        logger.warning(full_error_msg)
        return False, full_error_msg

# --- Main Reward Function (Core Logic) ---
def _calculate_reward_core(
    openscad_code: str,
    specific_reference_stl_path: str, # Expects an absolute or resolvable path
    global_config: Dict[str, Any],
    debug_mode: bool = False
) -> int:
    """
    Calculates the reward for a given OpenSCAD code string against a specific reference STL.
    This is the core logic, expecting all paths and configs to be resolved.
    """
    temp_dir = "" # Initialize to ensure it's available in finally block
    try:
        # Ensure global_config has essential keys (OpenSCAD path, thresholds)
        if 'openscad' not in global_config or 'executable_path' not in global_config['openscad']:
            raise ConfigError("OpenSCAD configuration missing in global_config for _calculate_reward_core")
        if 'reward_thresholds' not in global_config:
            raise ConfigError("reward_thresholds missing in global_config for _calculate_reward_core")

        # Create a temporary directory for .scad and .stl files
        temp_dir = tempfile.mkdtemp(prefix="reward_eval_")
        logger.debug(f"Created temporary directory for core evaluation: {temp_dir}")

        # 1. Render OpenSCAD to STL
        logger.debug("Core Step 1: Rendering OpenSCAD code to STL...")
        generated_stl_path = render_openscad_code_to_stl(openscad_code, temp_dir, global_config)
        if not generated_stl_path:
            logger.warning("Core Reward = 0 (Rendering failed or produced no STL)")
            return 0
        logger.debug(f"Core Rendering successful: {generated_stl_path}")

        # 2. Perform Geometric Checks
        logger.debug("Core Step 2: Performing geometric checks...")
        if not os.path.exists(specific_reference_stl_path):
            logger.error(f"Specific Reference STL file not found at path: {specific_reference_stl_path}. Cannot perform geometric checks.")
            logger.warning("Core Reward = 0 (Specific Reference STL missing)")
            return 0
        
        thresholds_cfg = global_config.get('reward_thresholds', {})

        # Check 2a: Volume Similarity
        vol_threshold = thresholds_cfg.get('volume_threshold_percent', 5.0)
        logger.debug(f"Core Checking Volume Similarity (Threshold: {vol_threshold}%)...")
        volume_passed, vol_msg = check_volume_similarity(generated_stl_path, specific_reference_stl_path, vol_threshold)
        if not volume_passed:
            logger.warning(f"Core Volume check failed: {vol_msg}. Reward = 0.")
            return 0
        logger.debug("Core Volume check PASSED.")

        # Check 2b: Shape Similarity (Chamfer, Hausdorff, ICP Fitness)
        shape_thresholds = {
            'chamfer_mm': thresholds_cfg.get('chamfer_threshold_mm', 1.0),
            'hausdorff_mm': thresholds_cfg.get('hausdorff_threshold_mm', 1.0),
            'icp_fitness': thresholds_cfg.get('icp_fitness_threshold', 0.95)
        }
        num_samples = thresholds_cfg.get('similarity_point_cloud_samples', 50000) # updated default
        shape_similarity_seed = thresholds_cfg.get('similarity_random_seed', 42) # standard seed name
        voxel_size_fpfh = thresholds_cfg.get('voxel_size_for_fpfh_mm', 0.5) # From config for _align_point_clouds

        logger.debug(f"Core Checking Shape Similarity (Chamfer <= {shape_thresholds['chamfer_mm']}mm, "
                    f"Hausdorff <= {shape_thresholds['hausdorff_mm']}mm, "
                    f"ICP Fitness >= {shape_thresholds['icp_fitness']}) with seed {shape_similarity_seed}...")
        
        # Call the alignment and similarity check function directly if it's combined
        # Or call _align_point_clouds then the distance computations.
        # Assuming check_shape_similarity now uses _align_point_clouds which pulls voxel_size_fpfh from config
        shape_passed, shape_msg = check_shape_similarity(
            generated_stl_path, 
            specific_reference_stl_path, 
            shape_thresholds, 
            num_samples,
            seed=shape_similarity_seed
            # voxel_size_fpfh is now read by _align_point_clouds from global_config['reward_thresholds'] or defaults
        )
        if not shape_passed:
            logger.warning(f"Core Shape similarity check failed: {shape_msg}. Reward = 0.")
            return 0
        logger.debug("Core Shape similarity check PASSED.")

        logger.info("All core checks passed. Reward = 1.")
        return 1

    except RenderError as e:
        logger.warning(f"Core Reward = 0 due to RenderError: {e}")
        return 0
    except GeometryCheckError as e: 
        logger.warning(f"Core Reward = 0 due to GeometryCheckError: {e}")
        return 0
    except ConfigError as e:
        logger.error(f"Core Reward calculation failed due to Configuration Error: {e}. Reward = 0.")
        return 0
    except Exception as e:
        logger.error(f"An unexpected error occurred in _calculate_reward_core: {e}", exc_info=True)
        logger.warning("Core Reward = 0 due to unexpected error.")
        return 0
    finally:
        if temp_dir and os.path.exists(temp_dir):
            if debug_mode:
                logger.info(f"Debug mode: Temporary directory preserved at {temp_dir}")
            else:
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory for core eval: {temp_dir}")
                except OSError as e:
                    logger.error(f"Error cleaning up temporary directory {temp_dir} for core eval: {e}")

# --- New Geometric Reward Function for Trainer ---
def geometric_reward(openscad_code: str, task_id: str, debug_mode: bool = False) -> float:
    """
    Calculates the geometric reward for a given OpenSCAD code and task ID.
    This is the entry point for the GRPOTrainer.
    Returns a float (0.0 or 1.0).
    """
    global CONFIG
    # 1. Load global configuration (OpenSCAD path, default thresholds, etc.)
    # This ensures CONFIG is populated if not already done by a standalone run.
    if not CONFIG:
        try:
            # Use default config.yaml path relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_config_path = os.path.join(script_dir, "config.yaml")
            if not os.path.exists(default_config_path):
                 # Fallback to cwd if not found next to script (e.g. when called from other scripts)
                default_config_path = "config.yaml"
            
            load_config(default_config_path) 
            validate_openscad_config(CONFIG.get('openscad', {}))
        except ConfigError as e:
            logger.error(f"geometric_reward: Failed to load/validate base configuration: {e}")
            return 0.0 # Cannot proceed without base config
        except Exception as e_cfg:
            logger.error(f"geometric_reward: Unexpected error loading/validating base configuration: {e_cfg}")
            return 0.0

    # 2. Load task definitions to find the reference STL for the task_id
    tasks = load_tasks_data()
    if not tasks:
        logger.error(f"geometric_reward: No tasks loaded from tasks.json. Cannot find reference for task_id '{task_id}'.")
        return 0.0

    specific_reference_stl_path = None
    task_found = False
    for task_info in tasks:
        if task_info.get("id") == task_id:
            task_found = True
            specific_reference_stl_path = task_info.get("ref_stl")
            break
    
    if not task_found:
        logger.error(f"geometric_reward: Task ID '{task_id}' not found in tasks.json.")
        return 0.0
    if not specific_reference_stl_path:
        logger.error(f"geometric_reward: 'ref_stl' not defined for task ID '{task_id}' in tasks.json.")
        return 0.0

    # Ensure the reference STL path is absolute or resolvable from CWD or script dir
    # If paths in tasks.json are relative to the project root, this should work if script is run from root.
    # For robustness, one might want to resolve it relative to tasks.json path or make them absolute.
    # For now, assume path is okay or relative to where the script is run / where tasks.json is found.
    # We can make this more robust if needed, e.g. by resolving rel to tasks.json dir.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_json_dir = os.path.dirname(os.path.join(script_dir, "tasks/tasks.json")) # dir of default tasks.json
    
    if not os.path.isabs(specific_reference_stl_path):
        # Try resolving relative to tasks.json's directory
        resolved_path = os.path.normpath(os.path.join(tasks_json_dir, specific_reference_stl_path))
        if os.path.exists(resolved_path):
            specific_reference_stl_path = resolved_path
        else:
            # Try resolving relative to current working directory as a fallback
            resolved_path_cwd = os.path.abspath(specific_reference_stl_path)
            if os.path.exists(resolved_path_cwd):
                 specific_reference_stl_path = resolved_path_cwd
            else:
                logger.error(f"geometric_reward: Reference STL '{specific_reference_stl_path}' for task '{task_id}' not found (tried relative to tasks.json dir and CWD). Original path from json: {task_info.get('ref_stl')}")
                return 0.0
    
    logger.info(f"geometric_reward: Using reference STL '{specific_reference_stl_path}' for task '{task_id}'.")

    # 3. Call the core reward calculation logic
    reward_value = _calculate_reward_core(
        openscad_code=openscad_code,
        specific_reference_stl_path=specific_reference_stl_path,
        global_config=CONFIG, # Pass the globally loaded and validated config
        debug_mode=debug_mode
    )
    return float(reward_value)


# --- Modified Main Execution / Example Usage ---
if __name__ == "__main__":
    # Setup logger for standalone execution. geometric_reward will use this logger if called.
    main_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "reward_test.log") # New log name
    logger = setup_logger(__name__, level=logging.DEBUG, log_file=main_log_file)
    
    o3d_logger = logging.getLogger("Open3D")
    o3d_logger.setLevel(logging.WARNING) # Reduce Open3D verbosity for tests unless errors
    # If you want Open3D logs to also go to the file:
    # if main_log_file:
    #     fh = logging.FileHandler(main_log_file, mode='a')
    #     fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    #     if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(main_log_file) for h in o3d_logger.handlers):
    #          o3d_logger.addHandler(fh)
    
    logger.info("--- Reward Script Started (Standalone Test) ---")

    parser = argparse.ArgumentParser(description="Evaluate OpenSCAD code using geometric_reward or _calculate_reward_core.")
    parser.add_argument("--code", type=str, help="A string of OpenSCAD code to evaluate.")
    parser.add_argument("--file", type=str, help="Path to an OpenSCAD (.scad) file to evaluate.")
    parser.add_argument("--task_id", type=str, help="Task ID (from tasks/tasks.json) to use for reference STL lookup with geometric_reward.")
    parser.add_argument("--reference_stl", type=str, help="Direct path to a specific reference STL (to test _calculate_reward_core directly).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the base configuration YAML file (default: config.yaml).")
    parser.add_argument("--tasks_json", type=str, default="tasks/tasks.json", help="Path to the tasks JSON file (default: tasks/tasks.json).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (preserves temporary files).")

    args = parser.parse_args()

    try:
        # Load base config first - this populates global CONFIG and validates OpenSCAD
        cfg = load_config(args.config)
        validate_openscad_config(cfg.get('openscad', {}))

        # Load tasks data (needed by geometric_reward, and good to have for context)
        load_tasks_data()

        openscad_code_to_evaluate = None
        source_description = ""

        if args.code:
            openscad_code_to_evaluate = args.code
            source_description = "from command line --code argument"
        elif args.file:
            try:
                with open(args.file, 'r') as f:
                    openscad_code_to_evaluate = f.read()
                source_description = f"from file: {args.file}"
            except FileNotFoundError:
                logger.error(f"SCAD file not found: {args.file}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error reading SCAD file {args.file}: {e}")
                sys.exit(1)
        
        if openscad_code_to_evaluate is not None:
            reward = -1.0 # Default to -1 to indicate not run or error
            if args.task_id:
                logger.info(f"Evaluating SCAD code {source_description} using geometric_reward for task_id: '{args.task_id}'")
                logger.info(f"SCAD Code:\n{openscad_code_to_evaluate}")
                reward = geometric_reward(openscad_code_to_evaluate, args.task_id, debug_mode=args.debug)
                logger.info(f"Evaluation via geometric_reward for task '{args.task_id}': Reward = {reward}")
            elif args.reference_stl:
                logger.info(f"Evaluating SCAD code {source_description} using _calculate_reward_core with reference_stl: '{args.reference_stl}'")
                logger.info(f"SCAD Code:\n{openscad_code_to_evaluate}")
                if not os.path.exists(args.reference_stl):
                    logger.error(f"Provided --reference_stl '{args.reference_stl}' not found.")
                    sys.exit(1)
                reward = _calculate_reward_core(openscad_code_to_evaluate, args.reference_stl, cfg, debug_mode=args.debug)
                logger.info(f"Evaluation via _calculate_reward_core: Reward = {reward}")
            else:
                # Fallback: Use reference_stl_file from config.yaml if it exists (legacy behavior for simple test)
                legacy_ref_stl = cfg.get('reference_stl_file')
                if legacy_ref_stl and os.path.exists(legacy_ref_stl):
                    logger.warning(f"No --task_id or --reference_stl provided. Falling back to use 'reference_stl_file' from {args.config} ({legacy_ref_stl}) for testing _calculate_reward_core.")
                    logger.info(f"Evaluating SCAD code {source_description} using _calculate_reward_core with reference_stl from config: '{legacy_ref_stl}'")
                    logger.info(f"SCAD Code:\n{openscad_code_to_evaluate}")
                    reward = _calculate_reward_core(openscad_code_to_evaluate, legacy_ref_stl, cfg, debug_mode=args.debug)
                    logger.info(f"Evaluation via _calculate_reward_core (legacy config ref): Reward = {reward}")
                else:
                    logger.error("No --task_id provided for geometric_reward, and no valid --reference_stl for _calculate_reward_core. Cannot evaluate.")
                    logger.error(f"Please provide --task_id, or --reference_stl. Legacy 'reference_stl_file' from config was '{legacy_ref_stl}'.")
                    sys.exit(1)
            print(f"Reward: {reward}") # Print final reward for easy capture

        else:
            logger.info("No SCAD code or file provided. Running internal test suite (if defined and configured).")
            # --- Example Test Cases (can be expanded or moved to a dedicated test script) ---
            example_scad_good = "cube(10, center=true);"
            # Assuming task1_cube in tasks.json points to a 10x10x10 cube STL.
            default_test_task_id = "task1_cube" 
            tasks_for_test = load_tasks_data()
            ref_stl_for_good_test = None
            for t_info in tasks_for_test:
                if t_info.get("id") == default_test_task_id:
                    ref_stl_for_good_test = t_info.get("ref_stl")
                    if ref_stl_for_good_test and not os.path.isabs(ref_stl_for_good_test):
                         script_dir = os.path.dirname(os.path.abspath(__file__))
                         tasks_json_dir = os.path.dirname(os.path.join(script_dir, args.tasks_json))
                         ref_stl_for_good_test = os.path.normpath(os.path.join(tasks_json_dir, ref_stl_for_good_test))
                    break
            
            if not ref_stl_for_good_test or not os.path.exists(ref_stl_for_good_test):
                logger.warning(f"Cannot run default 'Good Cube' test. Reference STL for task '{default_test_task_id}' not found or configured in {args.tasks_json}. Looked for: {ref_stl_for_good_test}")
                # Attempt to create a DUMMY reference if task1.stl for 'cube(10)' is the target
                # This logic might need to be smarter or removed if tasks.json is authoritative
                if default_test_task_id == "task1_cube" and (not ref_stl_for_good_test or "task1.stl" in ref_stl_for_good_test) :
                    dummy_ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", "stls")
                    dummy_ref_path = os.path.join(dummy_ref_dir, "task1.stl") # Matches default task1.stl name
                    if not os.path.exists(dummy_ref_path):
                        logger.warning(f"Attempting to create a DUMMY reference cube at {dummy_ref_path} for testing purposes.")
                        os.makedirs(dummy_ref_dir, exist_ok=True)
                        try:
                            dummy_scad_code_for_ref = "cube(10, center=true);"
                            # Use _calculate_reward_core's render part, but need to be careful with temp dirs
                            # For simplicity, just call render_openscad_code_to_stl directly to a known place
                            dummy_render_temp_dir = tempfile.mkdtemp(prefix="dummy_ref_render_")
                            dummy_stl_path_rendered = render_openscad_code_to_stl(dummy_scad_code_for_ref, dummy_render_temp_dir, cfg)
                            if dummy_stl_path_rendered:
                                shutil.move(dummy_stl_path_rendered, dummy_ref_path)
                                logger.info(f"Dummy reference STL created: {dummy_ref_path}.")
                                ref_stl_for_good_test = dummy_ref_path
                            shutil.rmtree(dummy_render_temp_dir)
                        except Exception as e_dummy:
                            logger.error(f"Could not create dummy reference STL: {e_dummy}")
                    else:
                         ref_stl_for_good_test = dummy_ref_path # It exists, use it.

            test_cases = []
            if ref_stl_for_good_test and os.path.exists(ref_stl_for_good_test):
                 test_cases.append({"name": f"Good Cube (vs {default_test_task_id})", "code": example_scad_good, "task_id": default_test_task_id, "expected_reward_approx": 1.0})
            
            # Add more test cases. These will use geometric_reward if task_id is provided.
            test_cases.extend([
                {"name": "Slightly Off Cube (vs task1_cube)", "code": "cube([10.5, 10, 9.8], center=true);", "task_id": default_test_task_id, "expected_reward_approx": "0.0 or 1.0"},
                {"name": "Different Shape (Sphere vs task1_cube)", "code": "sphere(r=6);", "task_id": default_test_task_id, "expected_reward_approx": 0.0},
                {"name": "Bad Syntax (Render Failure Test)", "code": "cube(10, center=true", "task_id": default_test_task_id, "expected_reward_approx": 0.0}, # task_id ensures it tries to find a ref, but render will fail
            ])

            if not test_cases:
                logger.info("No test cases to run (likely due to missing reference STL for default tests).")
            else:
                logger.info(f"Using reference STL from task '{default_test_task_id}': {ref_stl_for_good_test if ref_stl_for_good_test else 'Not Found'}")

            for test in test_cases:
                logger.info(f"\n--- Running Internal Test Case: {test['name']} ---")
                logger.debug(f"OpenSCAD Code:\n{test['code']}")
                
                current_reward = -1.0
                if test.get("task_id"):
                    current_reward = geometric_reward(test['code'], test['task_id'], debug_mode=args.debug)
                # Add elif for direct _calculate_reward_core if needed for some tests
                else: # Should not happen with current test_cases structure
                    logger.warning(f"Test case '{test['name']}' missing task_id, cannot run with geometric_reward.")
                    continue
                                
                logger.info(f"Test '{test['name']}': Reward = {current_reward} (Expected: {test['expected_reward_approx']})")
                if str(current_reward) != str(test['expected_reward_approx']) and test['expected_reward_approx'] != "0.0 or 1.0":
                    logger.warning(f"Potential mismatch for test '{test['name']}'!")
            
            logger.info("\n--- Standalone Test Run Finished ---")

    except ConfigError as e:
        logger.critical(f"Configuration error during standalone test: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during standalone test: {e}", exc_info=True)
        sys.exit(1) 