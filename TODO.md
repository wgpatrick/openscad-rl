### Project To-Do List

**Phase 1: Core Reward Function & Project Setup**
*   [x] Initialize Git repository and push to GitHub.
*   [x] Create initial `README.md` and Python-specific `.gitignore`.
*   [x] Develop `reward.py` (formerly `reward_function.py`).
    *   [x] Implement OpenSCAD rendering to STL.
    *   [x] Implement geometric checks (volume, Chamfer, Hausdorff, ICP fitness).
    *   [x] Load configuration from `config.yaml`.
    *   [x] Address non-determinism in geometric checks (seeding, RANSAC/ICP params).
    *   [x] Refactor for `geometric_reward(openscad_code, task_id)` interface.
    *   [x] Ensure global `CONFIG` is updated in-place by `load_config` for testability.
*   [x] Create `inspect_alignment.py` for visualizing point cloud alignment.
*   [x] Set up Conda environment with `environment.yml`.
*   [x] Restructure project directory:
    *   [x] `reward.py`
    *   [x] `tasks/` (with `tasks.json` and `tasks/stls/`)
    *   [x] `configs/` (with `single_task.yaml`)
    *   [x] `train_grpo.py` (placeholder created)
    *   [x] `eval_all_tasks.py` (placeholder created)
    *   [x] `tests/` directory (with subdirs for test files).
*   [x] Initial work on `train_grpo.py`:
    *   [x] CLI parsing.
    *   [x] Dataset wrapper for OpenSCAD tasks.
    *   [x] Model loading (Qwen3-8B, 8-bit quantization, LoRA).
    *   [x] WandB initialization.
    *   [x] Basic GRPOTrainer setup.
    *   [x] Reward function wrapper for `geometric_reward`.
*   [ ] Update `config.yaml` with refined/finalized paths and parameters, especially for `train_grpo.py` and any default test configurations if ever needed.
*   [ ] Update `tasks/tasks.json` with a diverse set of tasks for actual training/evaluation.

**Phase 2: Reward Function Unit Testing & Verification**
*   [x] Create comprehensive unit tests for `reward.py` in `tests/test_reward.py`.
    *   [x] Set up test structure to use `_calculate_reward_core` with test-specific SCAD files and reference STLs.
    *   [x] Created `tests/scad_test_files/` for input SCADs.
    *   [x] Created `tests/reference_stls_test_only/` for test-specific reference STLs.
    *   [x] Identified and implemented 10 diverse SCAD files for testing (covering expected correct, incorrect cases for 3 distinct tasks).
    *   [x] For each test SCAD file:
        *   [x] Defined the expected reward (0 or 1).
        *   [x] Implemented test cases (3 repetitions each) asserting the output of `_calculate_reward_core`.
        *   [x] Ensured reference STLs for these tests are placed in `tests/reference_stls_test_only/` and referenced directly by path.
    *   [x] Ensured tests cover different failure points (render failure implicitly tested if SCAD is bad, volume/shape mismatch explicitly through pass/fail cases).
    *   [x] Installed `pytest` in the Conda environment and successfully ran all (30) tests.
*   [ ] (Optional) Test command-line interface of `reward.py` more formally (though manual testing seems to have covered its use).
*   [ ] (Optional) Add specific tests for edge cases like empty SCAD code, non-existent reference STLs (though pytest would fail), SCAD files that produce empty STLs, etc., if not implicitly covered.

**Phase 3: GRPO Training Script Refinement (`train_grpo.py`)**
*   [ ] Finalize `train_grpo.py` implementation.
*   [ ] Refine model loading and configuration (e.g., paths, LoRA settings from `config.yaml` or training script args).
*   [ ] Ensure robust task loading from `tasks/tasks.json` and iteration for training.
*   [ ] Implement the core GRPO (Generative Reward Policy Optimization) loop in detail:
    *   [ ] Policy forward pass (generate SCAD code from prompt/task description in `tasks.json`).
    *   [ ] Correct extraction of generated SCAD code from model output (handle potential model verbosity, code block extraction).
    *   [ ] Reward calculation using `geometric_reward(generated_scad_code, task_id)` from `reward.py`.
    *   [ ] Policy update based on rewards (ensure gradients flow correctly through the policy network).
*   [ ] Enhance logging with Weights & Biases (W&B):
    *   [ ] Log rewards (per step, per epoch, distributions).
    *   [ ] Log generated SCAD code samples (especially successful ones or interesting failures).
    *   [ ] Log model outputs/completions.
    *   [ ] Log training progress (loss, learning rate, etc.).
    *   [ ] Log full configuration used for the run.
*   [ ] Implement configuration management for training runs:
    *   [ ] Allow `train_grpo.py` to use different configuration files from `configs/` (e.g., `single_task_training_config.yaml`, `multi_task_training_config.yaml`) for different training setups (learning rates, epochs, model params etc.).
*   [ ] Implement checkpointing (saving model weights periodically and at the end of training) and model saving (best model based on validation, if applicable).
*   [ ] Add basic error handling and recovery (e.g., gracefully handle OpenSCAD render errors during training batch, skip problematic task if necessary).

**Phase 4: GPU Instance Setup & Training Execution**
*   [ ] Set up GPU instance (e.g., Lambda Labs, Vast.ai, or cloud provider).
*   [ ] Perform software installations on the GPU instance:
    *   [ ] Install OpenSCAD (ensure version compatibility from `config.yaml`).
        *   [ ] Verify OpenSCAD command-line functionality with a simple test render.
    *   [ ] Install Conda and create the Python environment from `environment.yml`.
        *   [ ] Verify Python package installations (e.g., `trimesh`, `open3d`, `torch`, `transformers`, `peft`, `wandb`, `pyyaml`, `pytest`).
    *   [ ] Clone the `openscad-RL` Git repository.
    *   [ ] Configure SSH access (if not already done).
*   [ ] Copy local test files (`tests/scad_test_files/`, `tests/reference_stls_test_only/`) to the GPU instance.
*   [ ] Run the `reward.py` unit tests (`pytest tests/test_reward.py -v -s`) on the GPU instance to ensure environment consistency and that `reward.py` functions correctly there.
*   [ ] Run a small, short training job with `train_grpo.py` on a single task to test the end-to-end pipeline on the GPU.
*   [ ] Conduct full training runs with multiple tasks and for sufficient epochs.

**Phase 5: Evaluation & Iteration**
*   [ ] Develop `eval_all_tasks.py` for evaluating a trained model:
    *   [ ] Load a trained model checkpoint.
    *   [ ] Iterate through tasks in `tasks/tasks.json`.
    *   [ ] Generate SCAD code for each task using the model.
    *   [ ] Calculate reward using `geometric_reward`.
    *   [ ] Report aggregate metrics (e.g., success rate per task, overall success rate, average reward).
    *   [ ] Save generated SCADs and rendered STLs for successful/failed evaluations for manual review.
*   [ ] Analyze results and iterate on model architecture, prompts, reward function parameters, or training data/curriculum.

**Miscellaneous/Ongoing:**
*   [ ] Maintain and update documentation (`README.md`, comments in code).
*   [ ] Regularly commit and push changes to GitHub.
*   [ ] Consider adding more sophisticated geometric analysis or alternative reward shaping if initial results are suboptimal.
*   [ ] (Optional) Explore more complex OpenSCAD features or a wider variety of tasks. 