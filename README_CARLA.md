# Training CARLA Data Using Smolvla
This document describes how to train CARLA data using the smolvla policy.

## Installation

Follow the main [README.md](README.md) to install lerobot. Note: there are changes to the library in this repository.

## Dataset Generation

Use [examples/5_create_custom_dataset.py](examples/5_create_custom_dataset.py) to generate the dataset in lerobot format. You will need images and their respective measurements, such as DOFs. Note that dataset specific changes are necessary.

## Training

Run the following command to train:

```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=[REPO ID USED DURING DATA GEN]  \
    --batch_size=8 \
    --steps=20000
```

## Evaluation

Evaluate using [examples/6_eval_policy.py](examples/6_eval_policy.py).

## Results

Training and evaluation results are saved in the `outputs` folder.

See my result below:

<video src="outputs/eval/smol_vla_carla/output_video_corrected.mp4" controls width="600"></video>

## TODOs

- Improve task generation heuristics
- Increase data to make it a proper training
- Add augmentation to task description to force the model to learn the language relevance (add VQA and commentary same as [Simlingo](https://arxiv.org/abs/2503.09594))
- Augment data Using LLM. Similar to [SmolVLA](https://arxiv.org/abs/2506.01844).
- Create gym environment for proper evaluation
- And many more...