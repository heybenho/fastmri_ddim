"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import time
from collections import defaultdict
from pathlib import Path, WindowsPath

import numpy as np
import torch
from tqdm import tqdm

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.mri_data import fetch_dir

from ddim_module import DDIMModule

torch.set_float32_matmul_precision('medium')
torch.serialization.add_safe_globals([WindowsPath])
torch.multiprocessing.set_start_method('spawn', force=True)

def run_inference(challenge, checkpoint_path, data_path, output_path, device):
    start_time = time.perf_counter()

    model = DDIMModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model = model.to(device)

    data_transform = T.UnetDataTransform(which_challenge=challenge)
    dataset = SliceDataset(root=data_path, transform=data_transform, challenge=challenge)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)

    outputs = defaultdict(list)
    
    for batch in tqdm(dataloader, desc="Running inference"):
        image, _, mean, std, fname, slice_num, _ = batch
        
        with torch.no_grad():
            output = model.sample(image.to(device).unsqueeze(1), num_steps=1)
            
            # output normalization
            output = output.squeeze(1).cpu()
            mean = mean.unsqueeze(1).unsqueeze(2)
            std = std.unsqueeze(1).unsqueeze(2)
            output = (output * std + mean).cpu()
        
        outputs[fname[0]].append((slice_num[0].item(), output[0]))

    for fname in outputs:
        outputs[fname] = np.stack([out.numpy() for _, out in sorted(outputs[fname])])

    fastmri.save_reconstructions(outputs, output_path / "reconstructions")
    
    end_time = time.perf_counter()
    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    path_config = Path("ddim_config.yaml")

    challenge = "singlecoil"
    checkpoint_path = fetch_dir("checkpoint_path", path_config)
    data_path = fetch_dir("val_data_path", path_config)
    output_path = fetch_dir("output_path", path_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference(
        challenge,
        checkpoint_path,
        data_path,
        output_path,
        device,
    )