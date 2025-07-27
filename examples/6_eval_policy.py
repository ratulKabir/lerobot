import torch
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# === Paths ===
RGB_DIR = Path("/carla_source/validation_1_scenario/routes_validation/random_weather_seed_2_balanced_150/Town13_Rep0_10_route0_01_11_13_24_48/rgb")

# === Intrinsics Computation from FOV ===
def get_camera_intrinsics(image_width, image_height, fov_deg):
    fx = fy = image_width / (2.0 * np.tan(fov_deg * np.pi / 360.0))
    cx = image_width / 2
    cy = image_height / 2
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K

# === Camera Pose in Ego Frame ===
def get_camera_to_ego_matrix(pos, rot_deg):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = R.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    return T

# === Project 3D Points to Image Space ===
def project_to_image(points_ego, ego_to_camera, intrinsics):
    # points_ego: Nx3 in ego frame at t=0
    N = points_ego.shape[0]
    homog = np.concatenate([points_ego, np.ones((N, 1))], axis=1).T  # 4xN
    points_cam = ego_to_camera @ homog  # 4xN
    points_cam = points_cam[:3, :]  # 3xN

    # Convention fix: from project_point()
    x = points_cam[0, :]  # forward (depth)
    y = points_cam[1, :]  # right
    z = points_cam[2, :]  # up

    in_front = x > 0.1
    x = x[in_front]
    y = y[in_front]
    z = z[in_front]

    remapped = np.stack([y, -z, x], axis=0)  # shape (3, N)
    pixels = intrinsics @ remapped
    pixels /= pixels[2, :]  # normalize
    return pixels[:2, :].T  # Nx2



# === Visualize Predicted and Ground Truth Actions ===
def veirfy_action(predicted_action, ground_truth_action, ego_to_camera, idx, intrinsics):
    img_path = RGB_DIR / f"{idx:04d}.jpg"
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Image not found: {img_path}")
        return

    pred_np = predicted_action.squeeze(0).cpu().numpy()[:, :3]
    gt_np = ground_truth_action.squeeze(0).cpu().numpy()[:, :3]

    pred_px = project_to_image(pred_np, ego_to_camera, intrinsics)
    gt_px = project_to_image(gt_np, ego_to_camera, intrinsics)

    for pt in gt_px:
        cv2.circle(image, tuple(pt.astype(int)), 4, (0, 255, 0), -1)  # Green = Ground Truth
    for pt in pred_px:
        cv2.circle(image, tuple(pt.astype(int)), 4, (0, 0, 255), -1)  # Red = Predicted

    save_path = f"outputs/eval/smol_vla_carla/jpgs/{idx:04d}.jpg"
    cv2.imwrite(str(save_path), image)
    print(f"Saved visualization: {save_path}")

# === Main Pipeline ===
def main():
    pretrained_policy_path = "outputs/train/2025-07-26/11-26-25_smolvla_local/checkpoints/last/pretrained_model"
    device = torch.device("cuda")

    policy = SmolVLAPolicy.from_pretrained(pretrained_policy_path)
    policy.eval().to(device)
    policy.reset()

    dataset = LeRobotDataset("rat45/carla", episodes=[0], root="/carla_lerobot/carla")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # === Camera Pose and Intrinsics ===
    camera_to_ego = get_camera_to_ego_matrix([-1.5, 0.0, 2.0], [0.0, 0.0, 0.0])
    ego_to_camera = np.linalg.inv(camera_to_ego)
    image_w, image_h = 1024, 512
    fov = 110
    intrinsics = get_camera_intrinsics(image_w, image_h, fov)

    # === Inference Loop ===
    index = 0
    for batch in dataloader:
        with torch.inference_mode():
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            _, actions = policy.select_action(batch)
            veirfy_action(actions, batch["action"], ego_to_camera, index, intrinsics)
            index += 1

if __name__ == "__main__":
    main()
