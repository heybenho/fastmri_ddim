from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from fastmri.models import Unet
from fastmri.pl_modules.mri_module import MriModule

class DDIMDiffusionModel(nn.Module):
    def __init__(self, net, beta, n_timesteps):
        super().__init__()
        self.net = net
        self.n_timesteps = n_timesteps

        self.register_buffer('beta', beta)
        alpha = 1 - beta
        self.register_buffer('alpha_bar', torch.cumprod(alpha, dim=0))
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - self.alpha_bar))
        
    def forward(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        x_t = (sqrt_alpha_bar_t * x_0) + (sqrt_one_minus_alpha_bar_t * noise)
        
        return x_t, noise

    def predict_noise(self, x_t, t):
        # Adding t as an additional channel
        t = t.expand(x_t.shape[0])
        t_normed = (t.float() / self.n_timesteps).view(-1, 1, 1, 1)
        t_chan = torch.ones_like(x_t[:,:1,:,:]) * t_normed
        x_with_t = torch.cat([x_t, t_chan], dim=1)

        return self.net(x_with_t)
    
    def sample(self, x_noisy, num_steps=50, device=None):

        if device is None:
            device = next(self.parameters()).device

        x_t = x_noisy.to(device)

        t_start = self.n_timesteps//2
        t_tensor = torch.tensor([t_start]).to(device)
        noise = torch.randn_like(x_t).to(device)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t_start].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_start].view(-1, 1, 1, 1)
        x_t = (sqrt_alpha_bar_t * x_noisy) + (sqrt_one_minus_alpha_bar_t * noise)
        timesteps = torch.linspace(t_start, 0, num_steps).long().to(device)

        # going backwards, so t_next = t-1
        # timesteps = torch.linspace(self.n_timesteps-1, 0, num_steps).long().to(device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
            alpha_bar_t_next = self.alpha_bar[t_next].view(-1, 1, 1, 1)
            
            noise_pred = self.predict_noise(x_t, t)
            
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            dir_xt = torch.sqrt(1 - alpha_bar_t_next) * noise_pred
            
            x_t = torch.sqrt(alpha_bar_t_next) * x0_pred + dir_xt
        
        return x_t

class DDIMModule(pl.LightningModule):
    """PyTorch Lightning module for DDIM training."""
    def __init__(
        self,
        num_cascades=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        n_timesteps=1000,
        lr=1e-3,
        weight_decay=0.0,
        beta1=0.999,
        beta2=0.999,
        lr_step_size=40,
        lr_gamma=0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = Unet(
            in_chans=2,         # upped to 2, additional channel for t
            out_chans=1,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )
        
        beta = torch.linspace(1e-4, 0.02, n_timesteps)
        
        self.diffusion = DDIMDiffusionModel(self.net, beta, n_timesteps)
        
    def forward(self, x):
        return self.diffusion.net(x)
        
    def training_step(self, batch, batch_idx):
        """
        Training/validation step:
        Sample random timesteps, add cumulative noise at that timestep, try to predict noise, calculate loss.
        """
        image, target, mean, std, fname, slice_num, _ = batch
        x = image.unsqueeze(1) if image.dim() == 3 else image
        batch_size = x.shape[0]
        
        t = torch.randint(0, self.hparams.n_timesteps, (batch_size,), device=self.device)
        x_t, noise = self.diffusion(x, t)
        predicted_noise = self.diffusion.predict_noise(x_t, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, _ = batch
        x = image.unsqueeze(1) if image.dim() == 3 else image
        batch_size = x.shape[0]
        
        t = torch.randint(0, self.hparams.n_timesteps, (batch_size,), device=self.device)
        x_t, noise = self.diffusion(x, t)
        predicted_noise = self.diffusion.predict_noise(x_t, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )
        return [optimizer], [scheduler]
    
    def sample(self, x_t=None, num_steps=50):
        return self.diffusion.sample(x_t, num_steps)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)
        
        parser.add_argument("--chans", default=1, type=int, help="Number of top-level U-Net filters.")
        parser.add_argument("--num_pool_layers", default=4, type=int, help="Number of U-Net pooling layers.")
        parser.add_argument("--n_timesteps", default=1000, type=int, help="Number of diffusion timesteps.")
        parser.add_argument("--drop_prob", default=0.0, type=float, help="U-Net dropout probability")
        parser.add_argument("--lr", default=0.001, type=float, help="RMSProp learning rate")
        parser.add_argument("--lr_step_size", default=40, type=int, help="Epoch at which to decrease step size")
        parser.add_argument("--lr_gamma", default=0.1, type=float, help="Amount to decrease step size")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Strength of weight decay regularization")

        return parser