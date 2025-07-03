import os
import re
import utils
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from pathlib import Path
from random import randint
import dataloader
import diffusion

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
    "if_then_else", lambda condition, x, y: x if condition else y
)


def _print_config(config: omegaconf.DictConfig, resolve: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
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


def create_inpainting_input(
    input_prior_1=None, input_prior_2=None, num_mask_tokens=10, tokenizer=None
):
    mask_tokens = [tokenizer.mask_token_id] * num_mask_tokens
    sep = [tokenizer.convert_tokens_to_ids("</s>")]

    if input_prior_1 and input_prior_2:
        input_ids = (
            sep
            + tokenizer.encode(input_prior_1, add_special_tokens=False)
            + mask_tokens
            + tokenizer.encode(input_prior_2, add_special_tokens=False)
            + sep
        )
    ### Inpaint at C-term
    elif input_prior_1:
        input_ids = (
            sep
            + tokenizer.encode(input_prior_1, add_special_tokens=False)
            + mask_tokens
            + sep
        )

    ### Inpaint at N-term
    elif input_prior_2:
        input_ids = (
            sep
            + mask_tokens
            + tokenizer.encode(input_prior_2, add_special_tokens=False)
            + sep
        )
    else:
        input_ids = sep + mask_tokens + sep

    return input_ids


def run_unconditional_sampling(config, logger, tokenizer, tb_logger):
    logger.info("Running unconditional sampling...")
    model = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config, logger=False
    )
    model.eval()

    sep_token = tokenizer.eos_token
    mask_token = tokenizer.mask_token
    assert mask_token is not None, "Tokenizer must define a mask token"

    seq_length_list = config.sample.uncond.seq_length
    num_samples = config.sample.uncond.num_samples

    if len(seq_length_list) == 1:
        sampled_lengths = [seq_length_list[0]] * num_samples
    else:
        sampled_lengths = [
            randint(min(seq_length_list), max(seq_length_list))
            for _ in range(num_samples)
        ]

    full_token_ids = []
    for slen in sampled_lengths:
        masked_str = f"{sep_token}{mask_token * slen}{sep_token}"
        # print(masked_str)
        input_ids = tokenizer.encode(masked_str, return_tensors="pt").squeeze(0)[1:-1]
        # BUG: </s> and <s> being added at start and end fucking weird
        # print(tokenizer.decode(input_ids))
        # print(input_ids)
        full_token_ids.append(input_ids)

    full_token_ids = torch.cat(full_token_ids, dim=0)
    chunks = full_token_ids.split(config.model.length, dim=0)

    if chunks[-1].size(0) < config.model.length:
        chunks = chunks[:-1]

    logger.info(f"Total chunks to sample from: {len(chunks)}")
    sampling_batch_size = config.sample.uncond.sampling_batch_size
    text_samples = []

    for i in range(0, len(chunks), sampling_batch_size):
        batch_chunks = chunks[i : i + sampling_batch_size]
        batch_tensor = torch.stack(batch_chunks).to(model.device)
        samples = model.sample_cond(batch_tensor)
        decoded = tokenizer.batch_decode(samples)
        text_samples.extend(decoded)

    if tb_logger is not None:
        tb_logger.experiment.add_text("Samples", "\n\n".join(text_samples))

    cleaned_text = re.sub(r"[\s,\[\]']+", " ", "\n".join(text_samples)).strip()
    sequences = [
        seq.strip() for seq in re.findall(r"</s>(.*?)</s>", cleaned_text) if seq.strip()
    ]

    out_path = Path(
        f"/home/dhruva/discrete-diffusion-guidance/samples/uncond_len_{'-'.join(map(str, seq_length_list))}_num_{num_samples}.fasta"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i, seq in enumerate(sequences):
            if len(seq) > min(sampled_lengths) - 10:
                f.write(f">seq_{i}\n{seq}\n")

    print("Unconditional samples:", sequences)


def run_inpainting_sampling(config, logger, tokenizer, tb_logger):
    logger.info("Running inpainting sampling...")
    model = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config, logger=False
    )
    model.eval()

    prior_1 = config.sample.inpaint.prior_1
    prior_2 = config.sample.inpaint.prior_2
    mask_len = config.sample.inpaint.mask_len
    num_samples = config.sample.inpaint.num_samples

    # sep_token = tokenizer.eos_token

    mask_token_id = tokenizer.mask_token_id
    assert tokenizer.mask_token is not None, "Tokenizer must define a mask token"

    batch = []
    for _ in range(num_samples):
        input_ids = create_inpainting_input(
            input_prior_1=prior_1,
            input_prior_2=prior_2,
            num_mask_tokens=mask_len,
            tokenizer=tokenizer,
        )
        if len(input_ids) < config.model.length:
            input_ids += [mask_token_id] * (config.model.length - len(input_ids))
        input_tensor = torch.tensor(input_ids)  # .to(model.device)
        batch.append(input_tensor)

    batch_tensor = torch.stack(batch)
    logger.info(f"Sampling {batch_tensor.size(0)} inpainted sequences.")

    samples = model.sample_cond(batch_tensor)
    text_samples = tokenizer.batch_decode(samples)

    if tb_logger is not None:
        tb_logger.experiment.add_text("Inpainted Samples", "\n\n".join(text_samples))

    cleaned_text = re.sub(r"[\s,\[\]']+", " ", "\n".join(text_samples)).strip()
    sequences = [
        seq.strip() for seq in re.findall(r"</s>(.*?)</s>", cleaned_text) if seq.strip()
    ]

    out_path = Path(f"samples/inpaint_len_{mask_len}_num_{num_samples}.fasta")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")

    print("Inpainted samples:", sequences)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: omegaconf.DictConfig) -> None:
    L.seed_everything(config.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    _print_config(config, resolve=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=config.checkpointing.save_dir, name="tensorboard_logs"
    )

    if config.sample.mode == "uncond":
        run_unconditional_sampling(config, logger, tokenizer, tb_logger)
    elif config.sample.mode == "inpaint":
        run_inpainting_sampling(config, logger, tokenizer, tb_logger)
    else:
        raise ValueError(f"Unsupported sampling mode: {config.sample.mode}")


if __name__ == "__main__":
    main()
