import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import classifier
import dataloader
import diffusion
import eval_utils
import utils

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
    "if_then_else", lambda condition, x, y: x if condition else y
)


def _load_from_checkpoint(config, tokenizer):
    if "hf" in config.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to("cuda")

    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config, logger=False
    ).to("cuda")


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def _train(config, logger, tokenizer, train_classifier=False):
    logger.info("Starting Training.")

    wandb_logger = None
    tb_logger = None
    # Initialize Logger
    if config.eval.use_tensorboard:
        tb_logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=config.checkpointing.save_dir, name="tensorboard_logs"
        )

    else:
        if config.get("wandb", None) is not None:
            wandb_logger = L.pytorch.loggers.WandbLogger(
                config=omegaconf.OmegaConf.to_object(config), **config.wandb
            )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    if not config.is_vision:
        _print_batch(train_ds, valid_ds, tokenizer)

    if train_classifier:
        # This param indicates classifier will be used for
        #   PPLM / NOS-style guidance
        #  (see: https://arxiv.org/abs/2305.20009).
        if getattr(config, "is_pplm_classifier", False):
            pretrained_model = _load_from_checkpoint(config, tokenizer)
            if (
                getattr(config.classifier_model, "use_encoder_ema", True)
                and pretrained_model.ema
            ):
                pretrained_model.load_ema_params()
            pretrained_backbone = pretrained_model.backbone
            # Remove the last layer for the classifier
            if hasattr(pretrained_backbone, "output_layer"):  # DiT
                delattr(pretrained_backbone, "output_layer")
            if hasattr(pretrained_backbone, "model.lm_head"):  # DiMamba
                delattr(pretrained_backbone, "model.lm_head")
            if getattr(config.classifier_model, "freeze_encoder", True):
                for param in pretrained_backbone.parameters():
                    param.requires_grad = False
        else:
            pretrained_backbone = None

        model = classifier.Classifier(
            config,
            tokenizer=valid_ds.tokenizer,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger if wandb_logger else tb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _ppl_eval(config, tokenizer):
    print(f"Evaluating perplexity on {config.data.valid}.")
    pretrained = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    pretrained.eval()
    if not config.eval.disable_ema:
        pretrained.load_ema_params()

    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    ppl = eval_utils.compute_ppl(pretrained, valid_ds)
    print(f"PPL: {ppl:0.3f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == "gen_ppl_eval":
        pass
    elif config.mode == "ppl_eval":
        pass
    elif "train" in config.mode:
        _train(config, logger, tokenizer, train_classifier="classifier" in config.mode)
    else:
        raise NotImplementedError(f"Mode {config.mode} not implemented.")


if __name__ == "__main__":
    main()
