import os
import torch
import yaml
from argparse import ArgumentParser
from pathlib import Path, WindowsPath

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.data.mri_data import fetch_dir

from fastmri.pl_modules import FastMriDataModule

from ddim_module import DDIMModule

torch.set_float32_matmul_precision('medium')
torch.serialization.add_safe_globals([WindowsPath])

def main(args):
    pl.seed_everything(4)

    mask_func = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)

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
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = DDIMModule(
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        n_timesteps=args.n_timesteps,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    logger_dir_name = f"../logs/knee/singlecoil/ddim_logs/steps{args.n_timesteps}_chans{args.chans}_pool{args.num_pool_layers}_lr{args.lr}/"
    os.makedirs(logger_dir_name, exist_ok=True)
    final_logger = TensorBoardLogger(
        save_dir="../logs/knee/singlecoil/ddim_logs",
        name=f"steps{args.n_timesteps}_chans{args.chans}_pool{args.num_pool_layers}_lr{args.lr}"
    )

    args.logger = final_logger

    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, datamodule=data_module)

def build_args():
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

    # module config
    parser = DDIMModule.add_model_specific_args(parser)
    parser = load_config_to_args(parser, path_config, section_name="model")

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(default_root_dir=default_root_dir)
    parser = load_config_to_args(parser, path_config, section_name="trainer")

    args = parser.parse_args()

    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
    ]

    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args

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

if __name__ == "__main__":
    args = build_args()
    main(args)