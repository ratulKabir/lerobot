import gzip
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def load_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def extract_state(matrix, speed):
    pos = [matrix[0][3], matrix[1][3], matrix[2][3]]
    rot = np.array(matrix)[:3, :3]
    rpy = R.from_matrix(rot).as_euler('xyz', degrees=False).tolist()
    return pos + rpy + [speed]

def classify_task(future):
    """
    future: np.ndarray of shape [N, 7] → [pos_x, pos_y, pos_z, roll, pitch, yaw, speed]
    """
    # Use displacement and lateral shift for task prediction
    delta = future[-1]  # last future step
    dx, dy = delta[0], delta[1]
    speed = delta[6]

    lateral_shift = abs(dy)
    forward_shift = dx

    if speed < 0.5 and forward_shift < 0.5:
        return "stop"
    elif lateral_shift > 0.5:
        return "change lane to right" if dy > 0 else "change lane to left"
    elif speed - future[0][6] > 0.5:
        return "accelerate"
    elif future[0][6] - speed > 0.5:
        return "decelerate"
    else:
        return "keep speed"


def main():
    # === CONFIG ===
    fps = 10
    future_steps = 30
    repo_id = "rat45/carla"
    root_path = "/carla_lerobot/carla_dataset"
    meas_dir = Path("/carla_source/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48/measurements")
    image_dir = Path("/carla_source/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48/rgb")  # assumes *.png files like 00000.png

    # === FEATURES ===
    features = {
        "observation.images.top": {"dtype": "video", "shape": [3, 480, 640]},
        "action": {"dtype": "float32", "shape": (future_steps, 7)},
        "observation.state": {"dtype": "float32", "shape": (1, 7)},
    }

    # === INIT DATASET ===
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root_path,
        use_videos=True,
    )

    json_files = sorted(meas_dir.glob("*.json.gz"))
    n_frames = len(json_files)

    all_matrices = []
    all_speeds = []

    for f in json_files:
        data = load_json_gz(f)
        all_matrices.append(np.array(data["ego_matrix"]))
        all_speeds.append(data["speed"])

    for i in tqdm(range(n_frames)):
        # === State ===
        obs_state = extract_state(all_matrices[i], all_speeds[i])
        obs_state = torch.tensor(obs_state, dtype=torch.float32).reshape(1, 7)

        # === Action (future) ===
        T_i_inv = np.linalg.inv(all_matrices[i])
        future = []

        for j in range(i + 1, i + 1 + future_steps):
            if j >= n_frames:
                future.append([0.0]*7)
            else:
                T_rel = T_i_inv @ all_matrices[j]
                pos_rel = T_rel[:3, 3].tolist()
                rpy_rel = R.from_matrix(T_rel[:3, :3]).as_euler('xyz', degrees=False).tolist()
                speed_j = all_speeds[j]
                future.append(pos_rel + rpy_rel + [speed_j])

        future = torch.tensor(future, dtype=torch.float32)

        # === Image ===
        image_path = image_dir / f"{i:04d}.jpg"
        image = Image.open(image_path).convert("RGB").resize((640, 480))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # [3, H, W]

        # === Classify Task ===
        task = classify_task(future)
        
        # === Frame ===
        frame = {
            "observation.images.top": image,
            "action": future,
            "observation.state": obs_state,
        }

        dataset.add_frame(frame, task=task)

    # === SAVE EPISODE ===
    dataset.save_episode()
    print("✅ Saved one episode to:", root_path)

if __name__ == "__main__":
    main()
