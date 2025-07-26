import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    pretrained_policy_path = "outputs/train/2025-07-26/11-26-25_smolvla_local/checkpoints/last/pretrained_model"  # Path to the pretrained policy, if any.

    # # Select your device
    device = torch.device("cuda")

    # We can now instantiate our policy with this config and the dataset stats.
    policy = SmolVLAPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    policy.to(device)
    policy.reset()

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset("rat45/carla", episodes=[0], root="/carla_lerobot/carla")

    # Then we create our optimizer and dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    for batch in dataloader:
        # Prepare observation for the policy running in Pytorch
        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            _, actions = policy.select_action(batch)
            print(f"predicted Action: {actions}")
            print("Ground Truth Action: ", batch["action"])
            print()

        # postprocess action

if __name__ == "__main__":
    main()
