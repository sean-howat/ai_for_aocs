import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import webbrowser

#from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
#from gym_cleanup import Gravity_cleanup

from gym_ms import Gravity_cleanup

#  Load curriculum config 
with open("curriculum_config.json", "r") as f:
    all_stages = json.load(f)

final_env_config = all_stages[-1]["env_config"] # Use final stage config


print(" Using reward weights from final_env_config:")
for k, v in final_env_config.get("reward_weights", {}).items():
    print(f"  {k}: {v}")

# Load trained model 
#model_path = os.path.abspath("demo-ai-4-aocs-main/src/sb3_checkpoints/best_model.zip")
model_path = os.path.abspath("demo-ai-4-aocs-main/src/sb3_checkpoints/final_model.zip")

print(model_path)

model = RecurrentPPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

#  Create env 
env = Gravity_cleanup(final_env_config)
obs, _ = env.reset()

#  Logging 
trajectory = []
jd_times = []
delta_vs = []

reward_logs = {
    "alignment": [],
    "approach": [],
    "flyby": [],
    "dv_penalty": [],
    "closing_rate": [],
    "energy": [],
    "distance": [],
    "sun_penalty": [],
    "final_reward": [],
    "max_dv_exceeded": []
}

#  Run 1 episode
obs, _ = env.reset()
lstm_state = None

for step in range(1000):
    trajectory.append(obs)
    jd_times.append(env.epoch.jd)
    dv = getattr(env, "last_dv", np.zeros(3))
    delta_vs.append(np.array(dv).flatten())

    action, _ = model.predict(obs,state=lstm_state, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Log reward components
    for key in reward_logs:
        reward_logs[key].append(info.get(key, 0))

    if done or truncated:
        break

#  Clean up delta-v vectors 
cleaned_dvs = []
for i, dv in enumerate(delta_vs):
    dv_array = np.asarray(dv, dtype=np.float32).flatten()
    if dv_array.shape != (3,):
        print(f" delta_v at step {i} has shape {dv_array.shape}, replacing with zeros")
        dv_array = np.zeros(3, dtype=np.float32)
    cleaned_dvs.append(dv_array)

delta_vs_array = np.stack(cleaned_dvs)

#  Save data 
np.savez("trajectory_data_sb3.npz",
         observations=np.array(trajectory),
         jd_times=np.array(jd_times),
         num_frames=len(trajectory),
         dvs=delta_vs_array)

print(" Trajectory and rewards saved")

# === Plot weighted reward breakdown ===
plt.figure(figsize=(12, 8))
for key, values in reward_logs.items():
    weight = final_env_config["reward_weights"].get(key, 0.0)
    if weight != 0.0:
        weighted_values = [v * weight for v in values]
        plt.plot(weighted_values, label=f"{key} (weighted)")

plt.title("Weighted Reward Components Over Time (SB3)")
plt.xlabel("Step")
plt.ylabel("Weighted Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_breakdown_sb3.png")
webbrowser.open('file://' + os.path.realpath("reward_breakdown_sb3.png"))

# === Summary ===
print("\n Verifying reward weight mapping:")
for key in reward_logs:
    if key not in final_env_config["reward_weights"]:
        print(f" No weight defined for '{key}' — treated as 0.0")
    else:
        print(f" Using weight {final_env_config['reward_weights'][key]} for '{key}'")

weighted_summary = {
    "Total": {
        k: np.sum(v) * final_env_config["reward_weights"].get(k, 0.0)
        for k, v in reward_logs.items()
    },
    "Mean": {
        k: np.mean(v) * final_env_config["reward_weights"].get(k, 0.0)
        for k, v in reward_logs.items()
    }
}
weighted_df = pd.DataFrame(weighted_summary)
print("\n Weighted Reward Contribution Summary:")
print(weighted_df.round(3).to_string())

#   print out 
total_delta_v = info.get("total_dv_used_kms", -1)
distance_to_saturn = info.get("final_distance_to_saturn_km", -1)

print("\n Mission Summary:")
print(f"Total Δv Used: {total_delta_v:.3f} km/s")
print(f"Final Distance to target: {distance_to_saturn:.3f} km")
