from fastmri.losses import SSIMLoss
import torch
import os
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

dataset_type = "ddim_trained"
# dataset_type = "unet_pretrained"     # unet_pretrained or ddim_trained

original_path = "../datasets/knee/singlecoil_val/"  # Path to original data

if dataset_type == "unet_pretrained":
    recon_path = "../outputs/knee/singlecoil_val_pretrained/reconstructions/"
else:
    recon_path = "../outputs/knee/singlecoil_val_ddim_1steps/reconstructions/"

original_files = sorted([f for f in os.listdir(original_path) if f.endswith('.h5')])
recon_files = sorted([f for f in os.listdir(recon_path) if f.endswith('.h5')])

print(len(original_files), len(recon_files))

ssim_loss = SSIMLoss()
ssim_scores = []
min_slices = 100
max_slices = 0

for recon_file in recon_files:
    if recon_file in original_files:

        with h5py.File(os.path.join(recon_path, recon_file), 'r') as f_recon:
            recon_data = f_recon['reconstruction'][()]
            
        with h5py.File(os.path.join(original_path, recon_file), 'r') as f_orig:
            orig_data = f_orig['reconstruction_rss'][()]

                
        if orig_data is not None and recon_data is not None:
            # tensors size (batch, slice, height, width)
            orig_tensor = torch.from_numpy(orig_data).unsqueeze(1)
            recon_tensor = torch.from_numpy(recon_data)

            if dataset_type == "ddim_trained":
                recon_tensor = recon_tensor.unsqueeze(1)

            # print(orig_tensor.shape, recon_tensor.shape)

            slices = orig_tensor.shape[0]
            min_slices = min(min_slices, slices)
            max_slices = max(max_slices, slices)
        
            data_range = max(torch.max(orig_tensor).item(), torch.max(recon_tensor).item()) - min(torch.min(orig_tensor).item(), torch.min(recon_tensor).item())
            # print("Data range:", data_range)
            data_range = torch.tensor([data_range], device=orig_tensor.device, dtype=orig_tensor.dtype)
            
            ssim = 1 - ssim_loss(orig_tensor, recon_tensor, data_range).item()
            ssim_scores.append((recon_file, ssim))
            
            print(f"File: {recon_file}, SSIM: {ssim:.4f}")

print("Min/max slices")
print(min_slices, max_slices)

# Print summary statistics
if ssim_scores:
    avg_ssim = np.mean([score for _, score in ssim_scores])
    print(f"\nAverage SSIM across {len(ssim_scores)} images: {avg_ssim:.4f}")