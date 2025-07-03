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


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: omegaconf.DictConfig) -> None:
    # Reproducibility
    L.seed_everything(config.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    _print_config(config, resolve=True)
    print(f"Checkpoint: {config.eval.checkpoint_path}")

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=config.checkpointing.save_dir, name="tensorboard_logs"
    )

    pretrained = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config, logger=False
    )
    pretrained.eval()

    mask_token = tokenizer.mask_token
    sep_token = tokenizer.eos_token
    assert mask_token is not None, "Tokenizer must define a mask token"

    # === Determine seq lengths ===
    seq_length_list = config.eval.seq_length
    # assert isinstance(seq_length_list, list), "`eval.seq_length` must be a list"

    if len(seq_length_list) == 1:
        sampled_lengths = [seq_length_list[0]] * config.eval.num_samples
    elif len(seq_length_list) >= 2:
        min_len, max_len = min(seq_length_list), max(seq_length_list)
        sampled_lengths = [
            randint(min_len, max_len) for _ in range(config.eval.num_samples)
        ]
    else:
        raise ValueError("config.eval.seq_length must have at least one element")

    # === Construct tokenized chunks ===

    full_token_ids = []

    for slen in sampled_lengths:
        masked_str = f"{sep_token}{mask_token * slen}{sep_token}"
        input_ids = tokenizer.encode(masked_str, return_tensors="pt").squeeze(0)
        full_token_ids.append(input_ids)

    # Concatenate all input IDs into one long tensor
    full_token_ids = torch.cat(full_token_ids, dim=0)

    # Split into chunks of model input length
    chunks = full_token_ids.split(config.model.length, dim=0)

    # Drop final chunk if it's shorter than model length
    if chunks[-1].size(0) < config.model.length:
        chunks = chunks[:-1]

    logger.info(f"Total chunks to sample from: {len(chunks)}")

    # === Sampling ===
    sampling_batch_size = config.eval.sampling_batch_size
    text_samples = []

    for i in range(0, len(chunks), sampling_batch_size):
        batch_chunks = chunks[i : i + sampling_batch_size]
        batch_tensor = torch.stack(batch_chunks).to(pretrained.device)
        logger.info(
            f"Sampling batch {i // sampling_batch_size + 1} with size {batch_tensor.size(0)}"
        )
        samples = pretrained.sample_cond(batch_tensor)
        decoded = tokenizer.batch_decode(samples)
        text_samples.extend(decoded)

    # === TensorBoard logging ===
    if tb_logger is not None:
        tb_logger.experiment.add_text("Samples", "\n\n".join(text_samples))

    raw_path = Path(
        f"/home/dhruva/discrete-diffusion-guidance/samples/RAW_len_{'-'.join(map(str, seq_length_list))}_num_samples_{config.eval.num_samples}.txt"
    )
    # === Save raw outputs ===
    with open(raw_path, "w") as f:
        for sample in text_samples:
            f.write(sample + "\n")

    # === Extract and clean sequences ===
    cleaned_text = re.sub(r"[\s,\[\]']+", " ", "\n".join(text_samples)).strip()
    sequences = [
        seq.strip() for seq in re.findall(r"</s>(.*?)</s>", cleaned_text) if seq.strip()
    ]

    # === Save to FASTA ===
    out_path = Path(
        f"/home/dhruva/discrete-diffusion-guidance/samples/len_{'-'.join(map(str, seq_length_list))}_num_samples_{config.eval.num_samples}.fasta"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i, seq in enumerate(sequences):
            if len(seq) > min(sampled_lengths) - 10:
                f.write(f">seq_{i}\n{seq}\n")

    print("Samples:", sequences)
    return sequences


if __name__ == "__main__":
    main()
