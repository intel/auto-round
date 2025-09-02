# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import torch

from tqdm import tqdm
from PIL import Image

from auto_round.diffusion.diffusion_dataset import get_diffusion_dataloader


def compute_clip(prompts, images, device: str = "cuda"):
    from torchmetrics.multimodal import CLIPScore
    clip_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    for prompt, img_path in tqdm(zip(prompts, images), desc=f"Computing CLIP score"):
        image_data = Image.open(img_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image_data)).permute(2, 0, 1)
        clip_model.update(image_tensor.to(torch.float32).to(device).unsqueeze(0), prompt)
    result = clip_model.compute().mean().item()
    return {"CLIP": result}


def compute_clip_iqa(prompts, images, device: str = "cuda"):
    from torchmetrics.multimodal import CLIPImageQualityAssessment
    clip_model = CLIPImageQualityAssessment(
        model_name_or_path="openai/clip-vit-large-patch14"
    ).to(device)
    for prompt, img_path in tqdm(zip(prompts, images), desc=f"Computing CLIP-IQA score"):
        image_data = Image.open(img_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image_data)).permute(2, 0, 1)
        clip_model.update(image_tensor.to(torch.float32).to(device).unsqueeze(0))
    result = clip_model.compute().mean().item()
    return {"CLIP-IQA": result}


def compute_image_reward_metrics(prompts, images, device = "cuda"):
    import ImageReward
    image_reward_model = ImageReward.load("ImageReward-v1.0", device=device)
    scores = []
    for prompt, img_path in tqdm(zip(prompts, images), desc="Computing image reward metrics"):
        score = image_reward_model.score(prompt, img_path)
        scores.append(score)

    return  {"ImageReward": np.mean(scores)}

metric_map = {
    "clip": compute_clip,
    "clip-iqa": compute_clip_iqa,
    "imagereward": compute_image_reward_metrics,
}

def diffusion_eval(
    pipe,
    prompt_file,
    metrics,
    image_save_dir,
    batch_size,
    gen_kwargs,
):
    dataloader, _, _ = get_diffusion_dataloader(prompt_file, nsamples=10, bs=batch_size)
    prompt_list = []
    image_list = []
    for image_ids, prompts in dataloader:
        prompt_list.extend(prompts)

        new_ids = []
        new_prompts = []
        for idx, image_id in enumerate(image_ids):
            image_id = image_id.item()
            image_list.append(os.path.join(image_save_dir, str(image_id) + ".png"))

            if os.path.exists(os.path.join(image_save_dir, str(image_id) + ".png")):
                continue
            new_ids.append(image_id)
            new_prompts.append(prompts[idx])

        if len(new_prompts) == 0:
            continue

        output =  pipe(prompt=new_prompts, **gen_kwargs)
        for idx, image_id in enumerate(new_ids):
            output.images[idx].save(os.path.join(image_save_dir, str(image_id) + ".png"))

    result = {}
    for metric in metrics:
        result.update(metric_map[metric](prompt_list, image_list, pipe.device))

    import tabulate
    print(tabulate.tabulate(result.items(), tablefmt="grid"))

