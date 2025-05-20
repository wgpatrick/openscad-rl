import pytest
import os
import sys
import logging

# Add project root to sys.path to allow importing reward
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the core calculation function and config loaders from reward.py
from reward import _calculate_reward_core, load_config, validate_openscad_config, CONFIG, setup_logger as reward_setup_logger

# Configure logging for tests
# Use the logger from reward.py or set up a new one for tests
# logger = logging.getLogger("test_reward")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# It's often better to let the main reward script manage its logger, and tests can use it or their own.
# For simplicity, let's ensure the reward logger is initialized if tests run standalone.

# Get the main logger from reward.py and set its level for testing
logger = reward_setup_logger("reward_test_runner", level=logging.DEBUG) # Or logging.INFO


# Define SCAD files, their reference STLs, and expected rewards
# Ensure `scad_file` paths are relative to `tests/scad_test_files/`
# Ensure `reference_stl_path` paths are relative to `tests/reference_stls_test_only/`
TEST_CASES = [
    {
        "name": "task1_rep1_fail", 
        "scad_file_name": "task1_google_gemini-2.5-pro-preview-03-25_default_rep1.scad", 
        "reference_stl_name": "task1.stl", # Updated name
        "expected_reward": 0.0
    },
    {
        "name": "task1_rep2_pass", 
        "scad_file_name": "task1_google_gemini-2.5-pro-preview-03-25_default_rep2.scad", 
        "reference_stl_name": "task1.stl", # Updated name
        "expected_reward": 1.0
    },
    {
        "name": "task1_rep3_pass", 
        "scad_file_name": "task1_google_gemini-2.5-pro-preview-03-25_default_rep3.scad", 
        "reference_stl_name": "task1.stl", # Updated name
        "expected_reward": 1.0
    },
    {
        "name": "task1_rep4_fail", 
        "scad_file_name": "task1_google_gemini-2.5-pro-preview-03-25_default_rep4.scad", 
        "reference_stl_name": "task1.stl", # Updated name
        "expected_reward": 0.0
    },
    {
        "name": "task12_rep1_fail",
        "scad_file_name": "task12_openai_o3-2025-04-16_default_rep1.scad",
        "reference_stl_name": "task12.stl",
        "expected_reward": 0.0
    },
    {
        "name": "task12_rep2_pass",
        "scad_file_name": "task12_openai_o3-2025-04-16_default_rep2.scad",
        "reference_stl_name": "task12.stl",
        "expected_reward": 1.0
    },
    {
        "name": "task20_rep1_pass",
        "scad_file_name": "task20_openai_o3-mini-2025-01-31_default_rep1.scad",
        "reference_stl_name": "task20.stl",
        "expected_reward": 1.0
    },
    {
        "name": "task20_rep2_fail",
        "scad_file_name": "task20_openai_o3-mini-2025-01-31_default_rep2.scad",
        "reference_stl_name": "task20.stl",
        "expected_reward": 0.0
    },
    {
        "name": "task21_rep1_pass",
        "scad_file_name": "task21_openai_o3-mini-2025-01-31_default_rep1.scad",
        "reference_stl_name": "task21.stl",
        "expected_reward": 1.0
    },
    {
        "name": "task21_rep2_fail",
        "scad_file_name": "task21_openai_o3-mini-2025-01-31_default_rep2.scad",
        "reference_stl_name": "task21.stl",
        "expected_reward": 0.0
    }
    # Add more test cases here as they are provided
]

# Global setup for loading config once
@pytest.fixture(scope="session", autouse=True)
def setup_global_config_for_tests(): # Renamed to avoid potential clash if reward.py had a similar name
    logger.info("Setting up global config for tests...")
    try:
        # config.yaml should be found by load_config relative to reward.py or CWD
        # Running pytest from project root should make CWD work.
        config_path = "config.yaml" 
        load_config(config_path) # Populates global CONFIG
        if not CONFIG:
            raise RuntimeError("Test setup: Global CONFIG not populated after load_config.")
        
        openscad_cfg = CONFIG.get('openscad')
        if not openscad_cfg:
             raise RuntimeError("Test setup: 'openscad' section missing in loaded config.")
        validate_openscad_config(openscad_cfg)
        logger.info("Global config loaded and OpenSCAD validated successfully for tests.")

    except Exception as e:
        logger.error(f"Critical error during global test setup: {e}", exc_info=True)
        pytest.fail(f"Global test setup failed: {e}", pytrace=False)


def get_test_file_path(base_dir_name: str, file_name: str) -> str:
    """Constructs an absolute path to a test file within a subdirectory of tests/."""
    # __file__ is tests/test_reward.py. os.path.dirname(__file__) is tests/
    return os.path.join(os.path.dirname(__file__), base_dir_name, file_name)

def load_scad_code_from_test_file(scad_file_name: str) -> str:
    """Loads OpenSCAD code from a file in tests/scad_test_files/."""
    full_path = get_test_file_path("scad_test_files", scad_file_name)
    try:
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"SCAD file not found: {full_path}")
        with open(full_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        pytest.fail(f"Test SCAD file not found: {full_path}. Ensure it's in 'tests/scad_test_files/'.", pytrace=False)
        return "" 
    except Exception as e:
        pytest.fail(f"Error reading SCAD file {full_path}: {e}", pytrace=False)
        return ""

# Parameterize tests to run for each case and multiple repetitions
@pytest.mark.parametrize("test_case", TEST_CASES)
@pytest.mark.parametrize("repetition", range(1, 4)) # Run each test 3 times
def test_reward_calculation(test_case, repetition): # Renamed test function
    """
    Tests the _calculate_reward_core function with various SCAD codes and specific reference STLs.
    Each test is run multiple times to check for consistency.
    """
    test_name = test_case['name']
    scad_file_name = test_case["scad_file_name"]
    reference_stl_name = test_case["reference_stl_name"]
    expected_reward = test_case["expected_reward"]

    logger.info(f"Running test: {test_name}, Repetition: {repetition}/{3}")
    
    scad_code_to_test = load_scad_code_from_test_file(scad_file_name)
    if not scad_code_to_test: 
        pytest.fail(f"SCAD code is empty for {test_name} from file {scad_file_name}. Check file content and path.")

    logger.debug(f"SCAD Code for {test_name}:\n{scad_code_to_test}")

    # Get the absolute path to the test-specific reference STL
    specific_reference_stl_path = get_test_file_path("reference_stls_test_only", reference_stl_name)
    if not os.path.exists(specific_reference_stl_path):
        pytest.fail(f"Reference STL file for test '{test_name}' not found: {specific_reference_stl_path}", pytrace=False)

    # Call the _calculate_reward_core function
    # debug_mode=True will leave temp files behind.
    reward_value = _calculate_reward_core(
        openscad_code=scad_code_to_test,
        specific_reference_stl_path=specific_reference_stl_path,
        global_config=CONFIG, # CONFIG is populated by the setup_global_config_for_tests fixture
        debug_mode=False  # Set to True if you want to inspect temp files for a failing test
    )

    logger.info(f"Test: {test_name} (Rep {repetition}) - Expected: {expected_reward}, Got: {float(reward_value)}")
    
    assert float(reward_value) == expected_reward, \
        f"Test '{test_name}' (Rep {repetition}) failed. Expected reward {expected_reward}, but got {float(reward_value)}."

# Example of how to run:
# Ensure pytest is installed: pip install pytest
# From the project root directory (openscad-RL):
# pytest tests/test_reward.py -v -s

if __name__ == "__main__":
    logger.info("Running tests directly is not fully supported. Please use pytest.")
    print("To run tests, navigate to the project root directory and execute:")
    print("pytest tests/test_reward.py -v -s") 