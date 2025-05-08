import os
import yaml
import torch
from argparse import ArgumentParser

import optuna
from pytorch_lightning import Callback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.data.mri_data import fetch_dir
from fastmri.pl_modules import FastMriDataModule
from pathlib import Path, WindowsPath
from ddim_module import DDIMModule

import torch.multiprocessing
torch.set_float32_matmul_precision('medium')
torch.serialization.add_safe_globals([WindowsPath])
torch.multiprocessing.set_start_method('spawn', force=True)


def objective(trial, args):

    max_epochs = 10     # max epochs per trial
    min_delta = 0.001   # min change to be considered an improvement
    patience = 3        # num of consecutive non-improvement epochs for early stopping

    # Hyperparameters to tune
    chans = trial.suggest_categorical("chans", [8, 16, 32])
    num_pool_layers = trial.suggest_int("num_pool_layers", 3, 5)
    n_timesteps = trial.suggest_int("n_timesteps", 500, 1000, step=100)
    drop_prob = trial.suggest_float("drop_prob", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    lr_step_size = trial.suggest_int("lr_step_size", 2, 12, step=2)
    lr_gamma = trial.suggest_float("lr_gamma", 0.1, 0.9, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    

    mask_func = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    train_transform = UnetDataTransform(args.challenge, mask_func, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func, use_seed=True)
    test_transform = UnetDataTransform(args.challenge)

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=batch_size,
        num_workers=0,
    )
    
    model = DDIMModule(
        chans=chans,
        num_pool_layers=num_pool_layers,
        n_timesteps=n_timesteps,
        drop_prob=drop_prob,
        lr=lr,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        weight_decay=weight_decay,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode="min"
    )
    
    optuna_pruning = CustomPruningCallback(trial, monitor="val_loss")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,  # Reduced for faster trials
        accelerator="gpu" if args.gpus > 0 else "cpu",
        gpus=1 if args.gpus > 0 else None,
        callbacks=[early_stop_callback, optuna_pruning],
        deterministic=True,
        default_root_dir=args.default_root_dir / f"optuna_trial_{trial.number}",
        logger=pl.loggers.TensorBoardLogger(
            save_dir=args.default_root_dir / "optuna_logs",
            name=f"trial_{trial.number}"
        )
    )
    
    trainer.fit(model, datamodule=data_module)
    
    return trainer.callback_metrics["val_loss"].item()

class CustomPruningCallback(Callback):    
    def __init__(self, trial, monitor="val_loss", mode="min"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
    
    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        if self.monitor in metrics:
            current_score = metrics[self.monitor].item()
            self.trial.report(current_score, step=epoch)
            
            if self.trial.should_prune():
                message = f"Trial was pruned at epoch {epoch}."
                raise optuna.exceptions.TrialPruned(message)

def load_config_to_args(parser, config_path, section_name):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        if section_name in config:
            config = config[section_name]

            for key, value in config.items():
                parser.set_defaults(**{key: value})
                
        return parser
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found")
        return parser

def main():
    parser = ArgumentParser()

    path_config = Path("ddim_config.yaml")
    
    data_path = fetch_dir("data_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config)
    output_path = fetch_dir("output_path", path_config)
    
    # client and data_transform configs
    parser = load_config_to_args(parser, path_config, section_name="client")
    parser = load_config_to_args(parser, path_config, section_name="data_transform")

    # data module config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path=data_path, output_path=output_path, test_path=None)
    parser = load_config_to_args(parser, path_config, section_name="data_module")
    
    # Optuna-specific arguments
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    
    args = parser.parse_args()
    args.default_root_dir = Path(default_root_dir)
    
    os.makedirs(args.default_root_dir / "optuna_logs", exist_ok=True)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=4),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(args.default_root_dir / "best_params.txt", "w") as f:
        f.write(f"Best value: {trial.value}\n")
        f.write("Best hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()