import requests
import os
import json
import tqdm
import torch
import logging
import tiktoken
from torch.nn import functional as F
import torch._dynamo

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

@torch._dynamo.disable
def evaluate_hellswag(model, device, dataset_target_dir, ddp_world_size, ddp_rank):
    with open(os.path.join(dataset_target_dir, "hellswag.jsonl"), "r") as f:
        examples = []
        for line in f:
            examples.append(json.loads(line))

    enc = tiktoken.get_encoding("gpt2")
    total = torch.tensor(0, dtype=torch.long, device=device)
    correct = torch.tensor(0, dtype=torch.long, device=device)
    correct_normalized = torch.tensor(0, dtype=torch.long, device=device)

    torch._dynamo.config.optimize_ddp = False

    try:
        for i, example in enumerate(examples):
            if i % ddp_world_size != ddp_rank:
                continue
            tokens, mask, labels = load_example(example, enc, device)

            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                model.eval()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(tokens)

            epsilon = 1e-8
            shift_logits = (logits[:, :-1, :]).contiguous()
            shift_tokens = (tokens[:, 1:]).contiguous()

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

            total = total + torch.tensor(1, dtype=torch.long, device=device)
            correct = correct + torch.tensor(int(predictions == labels), dtype=torch.long, device=device)
            correct_normalized = correct_normalized + torch.tensor(int(predictions_normalized == labels), dtype=torch.long, device=device)

        return total, correct, correct_normalized

    except Exception as e:
        logging.error(f"Error in evaluation: {e}", exc_info=True)
        raise e
    finally:
        torch._dynamo.config.optimize_ddp = True


def process_hellswag(model, log_file, step, device, ddp, ddp_world_size, ddp_rank, master_process):
    """
    Process HellSwag evaluation with proper distributed handling
    """
    import torch.distributed as dist
    from datetime import datetime

    # Create the dataset directory if it doesn't exist
    dataset_target_dir = "data"
    os.makedirs(dataset_target_dir, exist_ok=True)

    # Get raw tensors from evaluation
    total_tensor, correct_tensor, correct_normalized_tensor = evaluate_hellswag(
        model, device, dataset_target_dir, ddp_world_size, ddp_rank
    )

    # Perform all_reduce operations if using DDP
    if ddp and ddp_world_size > 1:
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_normalized_tensor, op=dist.ReduceOp.SUM)

    # Convert to Python numbers after all_reduce
    total = total_tensor.item()
    correct = correct_tensor.item()
    correct_normalized = correct_normalized_tensor.item()

    # Calculate accuracies
    accuracy = (correct / total * 100) if total > 0 else 0.0
    accuracy_normalized = (correct_normalized / total * 100) if total > 0 else 0.0

    # Log results (only master process should do this)
    if master_process:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "total_examples": total,
            "correct": correct,
            "correct_normalized": correct_normalized,
            "accuracy": accuracy,
            "accuracy_normalized": accuracy_normalized
        }

        with open(log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

        print(f"Step {step}: Accuracy = {accuracy:.2f}%, Normalized Accuracy = {accuracy_normalized:.2f}%")

    return accuracy, accuracy_normalized


if __name__ == "__main__":
    download_hellswag_dataset(source, "data")
