import os
import json
import torch
import numpy as np
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback



from gym_ms import Gravity_cleanup  # your environment

print("CUDA available:", torch.cuda.is_available())
print("Torch built with CUDA:", torch.version.cuda)

# === Load curriculum config ===
with open("curriculum_config.json", "r") as f:
    curriculum_stages = json.load(f)

# === Paths ===
log_dir = os.path.expanduser("~/sb3_logs/gravity_ppo")
checkpoint_dir = os.path.abspath("demo-ai-4-aocs-main/src/sb3_checkpoints")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(checkpoint_dir, "best_model.zip")
final_model_path = os.path.join(checkpoint_dir, "final_model.zip")


best_reward = -np.inf
previous_model = None

resume_from_best = True #============================================================================== set to true if you want to reload a previously trained model
if resume_from_best:
    print(f"  Found final model checkpoint at {final_model_path}")
    checkpoint_path = final_model_path
else:
    print("  Starting training from scratch.")
    checkpoint_path = None

# === Curriculum Training Loop ===
for stage in curriculum_stages:
    stage_index = stage["stage_index"]
    env_config = stage["env_config"]
    training_iters = stage["training_iters"]


    print(f"\n Stage {stage_index + 1} with reward weights: {env_config['reward_weights']}")

    if training_iters == 0:
        print(f" Skipping Stage {stage_index + 1} (0 training iterations)")
        continue

    def make_env():
        return Monitor(Gravity_cleanup(env_config))

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    # ===  separate evaluation environment ===
    eval_env = DummyVecEnv([lambda: Monitor(Gravity_cleanup(env_config))])
    eval_env = VecMonitor(eval_env)


    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    num_steps=2048
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=num_steps, # 1024 28672
        batch_size=512, # 256 4092 
        gae_lambda=0.99, #ok
        gamma=0.9999,  #ok
        n_epochs=5, #10?
        ent_coef=0.003, #intoduce entropy  schedulinbg? 0.003 (0.015 super run)
        learning_rate=1.0e-4, 
        clip_range=0.2,  #0.2
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu",  # or "cuda"
        policy_kwargs={
            "log_std_init": 0.0, #-1,5
            "lstm_hidden_size": 128
        }
    )
    eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=checkpoint_dir,  # where to save the best model
    log_path=log_dir,
    eval_freq=150000,                      # how often to evaluate
    deterministic=True,
    render=False,
    verbose=1
)
    model.set_logger(new_logger)

    if checkpoint_path:
        print(f"Restoring from checkpoint: {checkpoint_path}")
        model.set_parameters(checkpoint_path)

    if previous_model is not None:
        print(" Loading weights from previous stage")
        model.set_parameters(previous_model)

    total_steps = training_iters * num_steps
    print(f" Training for {total_steps:,} timesteps")

    model.learn(
        total_timesteps=total_steps,
        log_interval=1,
        reset_num_timesteps=True, #needs to be false to see old tensorboard ==============================================================
        callback=eval_callback
    )


    previous_model = model.get_parameters()
    model.save(final_model_path)
    print(f" Model saved to {final_model_path}")
    

    lstm_state = None
    done = False
    total_reward = 0
    obs=eval_env.reset()
    
    while not done: #sort of useless code atm, remmeber to remove 
        action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += float(reward)

    print(f" Final evaluation reward (stage {stage_index + 1}): {total_reward:.2f}")
   
    #total_delta_v = info.get("total_dv_used_kms", -1)
    #distance_to_saturn = info.get("final_distance_to_saturn_km", -1)

    print("\n Mission Summary:")
    total_delta_v = info[0].get("total_dv_used_kms", -1)
    final_distance = info[0].get("final_distance_to_saturn_km", -1)

    print(f"Total Δv Used: {total_delta_v:.3f} km/s")
    print(f"Final Distance to target: {final_distance:.3f} km")

