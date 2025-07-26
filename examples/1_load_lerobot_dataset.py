# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pprint import pprint

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


repo_id = "lerobot/pusht"

delta_timestamps = {
    # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
    "observation.image": [-1, -0.5, -0.20, 0],
    # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
    "action": [t / 10 for t in range(64)],
}
# Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
# timestamp, you still get a valid timestamp.


carla_dataset = LeRobotDataset("rat45/carla", episodes=[0], root="/carla_lerobot/carla")
dataset_no_delta = LeRobotDataset(repo_id, episodes=[0, 1, 2])
dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=[0, 1, 2])

print(f"\n{dataset[0]['observation.image'].shape=}")  # (4, c, h, w)
print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

# Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
# PyTorch datasets.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=8,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader:
    print(f"{batch['observation.image'].shape=}")  # (32, 4, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
    print(f"{batch['action'].shape=}")  # (32, 64, c)
    break
