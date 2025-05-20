#!/usr/bin/env python
# train_grpo.py
# Launch with: accelerate launch train_grpo.py --config configs/multi_task.yaml

import argparse, json, os, random
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import reward                                              # <- your reward.py
import wandb

# ---------- 1. CLI / YAML parsing ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="YAML with training hyper-params")
    return p.parse_args()

# ---------- 2. Dataset wrapper ----------
class OpenScadTaskDataset(Dataset):
    """
    Each item is {'prompt': str, 'task_id': str}
    task_id lets reward function load the right reference STL.
    """
    def __init__(self, task_json_path):
        # Ensure task_json_path is an absolute path or resolvable
        # For simplicity, assuming it's relative to the script or a known location.
        # If train_grpo.py is at the root, "tasks/tasks.json" works.
        with open(task_json_path) as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        item = self.tasks[idx]
        return {"prompt": item["prompt"], "task_id": item["id"]}

# ---------- 3. Main ----------
def main():
    args = parse_args()
    # Note: The TRL GRPOConfig.from_yaml expects the YAML to be a valid GRPOConfig dump.
    # You might need to structure your YAMLs in configs/ accordingly.
    # For now, we load it, but some parameters below are hardcoded for clarity from your example.
    cfg = GRPOConfig.from_yaml(args.config)

    # 3a. Tokenizer & base model (8-bit) -----------------------
    # It's good practice to make model names configurable via the YAML (cfg.model_name)
    model_name = getattr(cfg, 'model_name', "qwen/Qwen3-8B") # Default if not in YAML
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # Or torch.bfloat16 for Ampere GPUs
        load_in_8bit=True,  # Requires bitsandbytes
        device_map="auto",  # Handles multi-GPU. For single GPU, can be specific e.g. "cuda:0"
        trust_remote_code=True
    )

    # 3b. Add LoRA adapters (PEFT) -----------------------------
    # These LoRA parameters could also be part of the GRPOConfig YAML
    lora_cfg = LoraConfig(
        r=getattr(cfg, 'lora_r', 16),
        lora_alpha=getattr(cfg, 'lora_alpha', 32),
        target_modules=getattr(cfg, 'lora_target_modules', ["q_proj","v_proj","k_proj"]), # check model's module names
        lora_dropout=getattr(cfg, 'lora_dropout', 0.05),
        bias=getattr(cfg, 'lora_bias', "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3c. Build dataset ---------------------------------------
    # The path to tasks.json could also be configurable
    train_ds = OpenScadTaskDataset(task_json_path="tasks/tasks.json")

    # 3d. WandB   --------------------------------------------
    wandb_project = getattr(cfg, 'wandb_project', "qwen3-openscad-rl-grpo")
    wandb.init(project=wandb_project, config=cfg.to_dict())

    # 3e. Reward fn wrapper expected by GRPO ------------------
    # This assumes reward.py has been adapted to have geometric_reward(code, task_id)
    # and that it handles its own configuration loading (e.g., for OpenSCAD path)
    # The GRPO trainer will pass full texts of completions, not just the generated part.
    # We might need to truncate prompts from completions if reward.geometric_reward expects only the generated code.
    def reward_function_for_trainer(completions: list[str], prompts: list[str], metas: list[dict]) -> list[float]:
        """
        completions: list[str]  – model outputs (prompt + generated text)
        prompts:     list[str]  - the original prompts
        metas:       list[dict] – contains 'task_id'
        """
        rewards_output = []
        for i, (code_full, p_text, meta) in enumerate(zip(completions, prompts, metas)):
            task_id = meta["task_id"]
            # Extract only the generated part of the code
            # This assumes the prompt is a prefix of the completion.
            if code_full.startswith(p_text):
                generated_code = code_full[len(p_text):]
            else:
                # Fallback or error if prompt is not a prefix (should not happen with standard generation)
                print(f"Warning: Prompt not a prefix of completion for task {task_id}. Using full completion.")
                generated_code = code_full
            
            print(f"\n--- Evaluating for Task: {task_id} ---")
            print(f"Prompt: {p_text}")
            print(f"Generated SCAD (for reward):\n{generated_code}")
            
            # Call your reward function
            # Ensure reward.geometric_reward is adapted to take task_id
            # and handle its config for OpenSCAD paths etc. internally.
            current_reward = reward.geometric_reward(generated_code, task_id)  # scalar float
            print(f"Reward: {current_reward}")
            rewards_output.append(float(current_reward))
        return rewards_output

    # 3f. Trainer --------------------------------------------
    # Ensure the GRPOTrainer `args` (which is `cfg`) has all necessary training arguments
    # like learning_rate, batch_size, gradient_accumulation_steps, etc.
    # These should be in your YAML file (e.g., configs/multi_task.yaml).
    
    # The `ref_model` for KL divergence. It can be None, or path to model, or name.
    # If model was loaded with device_map="auto", ref_model might also need careful handling
    # or to be loaded separately if not using the same base for PEFT.
    # For GRPOTrainer, ref_model is often the original pre-trained model before LoRA.
    # It can be set to None if you don't want to use a reference model for KL penalty.
    # If None, trainer will use self.model as ref_model, meaning KL will be 0.
    # For GRPO, ref_model should be the SFT model, if you have one, or the base pre-trained model.
    ref_model_name = getattr(cfg, 'ref_model_name', model_name) # Default to same base model

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_function_for_trainer, # Note: The API uses reward_fn, not reward_funcs
        args=cfg,                       # This is your GRPOConfig object from YAML
        train_dataset=train_ds,
        # ref_model_name=ref_model_name, # Pass the name for KL baseline if desired
                                        # GRPOTrainer might handle ref_model loading internally or expect it pre-loaded.
                                        # The example used "qwen/Qwen3-8B" directly.
                                        # For PEFT models, this usually means loading the base model separately without adapters.
                                        # Or, it can be None.
        # For simplicity and based on common TRL DPO/GRPO usage with PEFT:
        # Often, the ref_model is implicitly the base model before PEFT adapters are applied.
        # If an explicit ref_model is loaded, it shouldn't have the LoRA adapters.
        # TRL handles creating a non-trainable reference model from `model` if `ref_model` is None.
        # For GRPOTrainer, `ref_model` argument for the trainer explicitly loads another model.
        # If you want to use the base of the LoRA model as ref, usually it's handled internally if ref_model=None.
        # However, the user code specified ref_model="qwen/Qwen3-8B".
        # Let's stick to user's code for now and ensure it is loaded by trainer.
        ref_model_name=ref_model_name, # This should be the name of the model for the trainer to load as reference.
    )

    trainer.train()
    
    # Save model
    final_model_path = Path(cfg.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main() 