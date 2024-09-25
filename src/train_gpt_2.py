import os
import time
import torch
from models.gpt_2_basic import GPT2Configuration, GPT2Basic
from utils.schedulers import CosineScheduler
from utils.optimizer import Optimizer
from utils.data_loader_fineweb import FinewebEduDataset
from utils.hellswag import evaluate_hellswag, download_hellswag_dataset
import tiktoken
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist


# Constants
DATASET_TARGET_DIR = os.path.join(os.path.dirname(__file__), '../resources/hellswag')
VALIDATION_PER_STEPS = 500
HELLSWAG_STEPS = 500
SOURCE = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
GPT2_SMALL = "gpt2"
CONFIGURATION = GPT2Configuration(num_layers=12, num_heads=12, d_model=768)
RESULT_DIR = "result"
LOG_FILE = os.path.join(RESULT_DIR, "log.txt")


def setup_environment():
    """Set up the environment for distributed training."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


def initialize_ddp():
    """Initialize the distributed data parallel (DDP) setup."""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_rank = int(os.environ.get('RANK', 1))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK', 1))
        ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))

        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        init_process_group(backend='nccl', rank=ddp_rank, world_size=ddp_world_size)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def load_data(tokenizer, ddp_rank, micro_batch_size, max_seq_len):
    """Load the training and validation datasets."""
    dataset_folder = os.path.join(os.path.dirname(__file__), '../resources/edu_fineweb')
    data = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train') if not os.path.exists(dataset_folder) else None

    train_data_loader = FinewebEduDataset(data=data, dataset_folder=dataset_folder, tokenizer=tokenizer, batch_size=micro_batch_size,
                                          max_seq_len=max_seq_len, process_rank=ddp_rank, split='train')

    valid_data_loader = FinewebEduDataset(data=data, dataset_folder=dataset_folder, tokenizer=tokenizer, batch_size=micro_batch_size,
                                          max_seq_len=max_seq_len, process_rank=ddp_rank, split='val')
    return train_data_loader, valid_data_loader


def setup_model(device, ddp, ddp_local_rank):
    """Set up the model for training."""
    model = GPT2Basic(CONFIGURATION)
    model = model.to(device)
    model = torch.compile(model)
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    return model, raw_model


def process_validation(model, valid_data_loader, log_file, step, device, ddp, master_process):
    """Process validation data."""
    valid_time_start = time.time()
    total_loss = 0
    model.eval()
    valid_data_loader.reset()
    valid_steps = 20

    for validation_steps in range(valid_steps):
        x, y = valid_data_loader.next_batch()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x.to(device), y.to(device))
                loss = loss / valid_steps
                total_loss += loss.detach()
        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)

    if master_process:
        valid_time_end = time.time()
        valid_time = valid_time_end - valid_time_start
        print(f'Validation took: {(valid_time):.2f} sec, validation loss: {total_loss.item()}')
        with open(log_file, "a") as f:
            f.write(f"{step} val {total_loss.item():.4f}\n")

    return total_loss


def process_hellswag(model, log_file, step, device, ddp, ddp_world_size, ddp_rank, master_process):
    """Process Hellswag evaluation."""
    total, correct, correct_normalized = evaluate_hellswag(model, device, DATASET_TARGET_DIR, ddp_world_size, ddp_rank)
    if ddp:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_normalized, op=dist.ReduceOp.SUM)

    if master_process:
        print(f"Hellswag acc {correct / total}, hellswag acc normalized {correct_normalized / total}")
        with open(log_file, "a") as f:
            f.write(f"{step} hellan {(correct_normalized / total):.4f}\n")
            f.write(f"{step} hella {(correct / total):.4f}\n")


def train_model():
    """Main function to train the model."""
    setup_environment()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = initialize_ddp()
    download_hellswag_dataset(SOURCE, dirname=DATASET_TARGET_DIR)
    model, raw_model = setup_model(device, ddp, ddp_local_rank)
    tokenizer = tiktoken.get_encoding('gpt2')
    train_data_loader, valid_data_loader = load_data(tokenizer, ddp_rank, micro_batch_size=8, max_seq_len=1024)

    if master_process:
        print("Loaded dataset!")

    optimizer = Optimizer(raw_model, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scheduler = CosineScheduler(max_lr=6e-4, warmup_steps=715, min_lr=6e-5, total_steps=19073)
    grad_accumulation_steps = 16384 // (8 * 1024 * ddp_world_size)

    if master_process:
        print('Starting training')
        print(f'Total batch size: {16384}')
        print(f'Micro batch size: {8}')
        print(f'Grad accumulation steps: {grad_accumulation_steps}')

    train_start = time.time()
    tokens_per_batch = 8 * 1024 * grad_accumulation_steps
    total_tokens = 0

    torch.set_float32_matmul_precision('high')
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        pass

    for step in range(scheduler.total_steps):
        last_step = (step == scheduler.total_steps - 1)
        start = time.time()
        model.train()
        optimizer.zero_grad()
        loss_accumulated = 0.0

        for micro_step in range(grad_accumulation_steps):
            x, y = train_data_loader.next_batch()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x.to(device), y.to(device))
                loss /= grad_accumulation_steps
            loss_accumulated += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accumulated, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.set_learning_rate(scheduler.get_lr(step))
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()
        tokens_per_sec = tokens_per_batch / (end - start)
        total_tokens += tokens_per_batch

        if master_process:
            print(f'<Step {step}> loss:{loss_accumulated}, norm:{norm} tok/sec:{tokens_per_sec:.2f}, time:{(end - start)*1000:.2f} ms')
            with open(LOG_FILE, "a") as f:
                f.write(f"{step} train {loss_accumulated:.6f}\n")

        if (step > 0 and step % VALIDATION_PER_STEPS == 0) or last_step:
            total_valid_loss = process_validation(model, valid_data_loader, LOG_FILE, step, device, ddp, master_process)
            process_hellswag(model, LOG_FILE, step, device, ddp, ddp_world_size, ddp_rank, master_process)

        if (step > 0 and step % HELLSWAG_STEPS == 0) or last_step:
            process_hellswag(model, LOG_FILE, step, device, ddp, ddp_world_size, ddp_rank, master_process)

        if master_process and (step > 0 and (step % VALIDATION_PER_STEPS == 0 or last_step)):
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': total_valid_loss.item()
            }
            torch.save(checkpoint, os.path.join(RESULT_DIR, f"gpt_2_model_{step:05d}.pt"))

    train_end = time.time()
    print(f'Training took: {(train_end - train_start):.2f} sec, avg tok/sec: {total_tokens / (train_end - train_start):.2f}')

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    train_model()
