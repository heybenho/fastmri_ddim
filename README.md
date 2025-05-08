This is an implementation of a Denoising Diffusion Implicit Model (DDIM) for fast MRI reconstructions, similar to Meta AI's fastMRI efforts. DDIM is a diffusion-based model that trains like DDPM, but its inference uses much fewer timesteps, improving inference time significantly at little cost to quality. Compared to the standard U-Net model in fastMRI, DDIM reduced inference time by 89% with only a loss of 0.7% in SSIM score.

Meta AI fastMRI: https://github.com/facebookresearch/fastMRI
NYU fastMRI Dataset: https://fastmri.med.nyu.edu/
DDIM: https://arxiv.org/abs/2010.02502
