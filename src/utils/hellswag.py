import requests
import os
import json
import tqdm
import torch
import logging
import tiktoken
from torch.nn import functional as F

source = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def download_hellswag_dataset(source, dirname, chunk_size=1024):
    os.makedirs(dirname, exist_ok=True)

    full_filename = os.path.join(dirname, 'hellswag.jsonl')
    if os.path.exists(full_filename):
        logging.debug(f"Dataset already exists at {dirname}")
        return

    logging.debug(f"Downloading dataset from {source} to {dirname}")
    response = requests.get(source)

    with open(full_filename, 'wb') as f, tqdm.tqdm(
        full_filename,
        total=len(response.content),
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc="Storing dataset",
    ) as bar:
        for chunk in response.iter_content(chunk_size):
            written = f.write(chunk)
            bar.update(written)


def load_example(example, enc, device):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    tokens = enc.encode(ctx)
    tokens_rows = []
    mask_rows = []

    for ending in endings:
        end_tokens = enc.encode(" " + ending)
        tokens_rows.append(tokens + end_tokens)
        mask_rows.append([0] * len(tokens) + [1] * len(end_tokens))

    max_tokens = max(len(tokens) for tokens in tokens_rows)
    tokens = torch.zeros((4, max_tokens), dtype=torch.long, device=device)
    mask = torch.zeros((4, max_tokens), dtype=torch.long, device=device)

    for i, (tokens_r, mask_r) in enumerate(zip(tokens_rows, mask_rows)):
        tokens[i, :len(tokens_r)] = torch.tensor(tokens_r, dtype=torch.long, device=device)
        mask[i, :len(mask_r)] = torch.tensor(mask_r, dtype=torch.long, device=device)

    return tokens, mask, label


def evaluate_hellswag(model, device, dataset_target_dir, ddp_world_size, ddp_rank):

    with open(os.path.join(dataset_target_dir, "hellswag.jsonl"), "r") as f:
        examples = []
        for line in f:
            examples.append(json.loads(line))

    enc = tiktoken.get_encoding("gpt2")
    total = 0
    correct = 0
    correct_normalized = 0
    try:
        for i, example in enumerate(examples):
            if i % ddp_world_size != ddp_rank:
                continue
            tokens, mask, labels = load_example(example, enc, device)

            tokens = tokens.to(device)
            mask = mask.to(device)
            # with torch.no_grad(), torch._dynamo.disable():
            with torch.no_grad():
                model.eval()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(tokens)  # Pass both tokens and labels
            epsilon = 1e-8
            shift_logits = (logits[:, :-1, :]).contiguous()  # B T V
            shift_tokens = (tokens[:, 1:]).contiguous()  # B T

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_tokens = shift_tokens.view(-1)

            loss = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
            loss = loss.view(tokens.size(0), -1)

            shift_mask = (mask[..., 1:]).contiguous()
            masked_shift_losses = loss * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / (shift_mask.sum(dim=1) + epsilon)

            predictions = sum_loss.argmin().item()
            predictions_normalized = avg_loss.argmin().item()

            total += 1
            correct += int(predictions == labels)
            correct_normalized += int(predictions_normalized == labels)

        return total, correct, correct_normalized
    except Exception as e:
        logging.error(f"Error in evaluation: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    download_hellswag_dataset(source, "data")
