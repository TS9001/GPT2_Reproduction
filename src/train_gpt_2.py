import os
import time
import torch
from models.transformer_plus_baseline import GPT2Basic as TransformerPlusBasic
from models.gpt_2_baseline import GPT2Basic
from models.model_configuration import ModelConfiguration
from utils.schedulers import CosineScheduler
from utils.optimizer import Optimizer
from utils.data_loader_fineweb import FinewebEduDataset
from utils.hellswag import evaluate_hellswag, download_hellswag_dataset
import tiktoken
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import wandb
from datetime import datetime

if os.name == 'nt':
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

# Constants
DATASET_TARGET_DIR = os.path.join(os.path.dirname(__file__), '../resources/hellswag')
VALIDATION_PER_STEPS = int(os.environ.get('VALIDATION_PER_STEPS', 500))
HELLSWAG_STEPS = int(os.environ.get('HELLSWAG_STEPS', 500))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 500))
SOURCE = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
GPT2_SMALL = "gpt2"
USE_LIGER = os.environ.get('USE_LIGER', 'True') == 'True'
RESULT_DIR = "result"
LOG_FILE = os.path.join(RESULT_DIR, "log.txt")
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 64))
SEQUENCE_LENGTH = int(os.environ.get('SEQUENCE_LENGTH', 1024))
TOTAL_BATCH_SIZE = int(os.environ.get('TOTAL_BATCH_SIZE', 524288))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 6e-4))
WARMUP_STEPS = int(os.environ.get('WARMUP_STEPS', 715))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 0.1))
EPSILON = float(os.environ.get('EPSILON', 1e-8))
BETAS1 = float(os.environ.get('BETAS1', 0.9))
BETA2 = float(os.environ.get('BETA2', 0.95))
TOTAL_STEPS = int(os.environ.get('TOTAL_STEPS', 19073))
PRINT_STEPS = int(os.environ.get('PRINT_STEPS', 50))
BETAS = (BETAS1, BETA2)
EPOCHS = int(os.environ.get('EPOCHS', 1))
SAVE_ON_LAST = os.environ.get('SAVE_ON_LAST', 'True') == 'True'
MIN_LR = float(os.environ.get('MIN_LR', LEARNING_RATE * 0.1))
ARCHITECTURE = os.environ.get('ARCHITECTURE', 'DEFAULT')

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


def load_data(tokenizer, ddp_rank, micro_batch_size, max_seq_len, ddp_world_size):
    """Load the training and validation datasets."""
    dataset_folder = os.path.join(os.path.dirname(__file__), '../resources/edu_fineweb')

    train_data_loader = FinewebEduDataset(
        dataset_folder=dataset_folder,
        batch_size=micro_batch_size,
        max_seq_len=max_seq_len,
        num_processes=ddp_world_size,
        process_rank=ddp_rank,
        split='train'
    )
    valid_data_loader = FinewebEduDataset(
        dataset_folder=dataset_folder,
        batch_size=micro_batch_size,
        max_seq_len=max_seq_len,
        num_processes=ddp_world_size,
        process_rank=ddp_rank,
        split='val'
    )

    return train_data_loader, valid_data_loader


def setup_model(device, ddp, ddp_local_rank, model_config):
    """Set up the model for training."""
    if ARCHITECTURE == "TRANSFORMER_PLUS":
        model = TransformerPlusBasic(model_config)
    else:  # Default to GPT2Basic
        model = GPT2Basic(model_config)

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
    torch._dynamo.config.optimize_ddp = False
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

    torch._dynamo.config.optimize_ddp = True
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


def process_micro_batch(model, train_data_loader, device, training_config):
    loss_accumulated = 0.0
    for micro_step in range(training_config['grad_accumulation_steps']):
        x, y = train_data_loader.next_batch()
        if training_config['ddp']:
            model.require_backward_grad_sync = (micro_step == training_config['grad_accumulation_steps'] - 1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x.to(device), y.to(device), return_logits=False)
            loss = loss / training_config['grad_accumulation_steps']
        loss_accumulated += loss.detach()
        loss.backward()
    return loss_accumulated


def train_model():
    """Main function to train the model."""
    setup_environment()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = initialize_ddp()

    # Create model configuration for all processes
    model_config = ModelConfiguration(num_layers=12, num_heads=12, d_model=768, use_liger=USE_LIGER)

    # Initialize wandb (only on master process)
    if master_process:
        # Get current datetime and format it
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get the script name without extension
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        # Create run name
        run_name = f"{script_name}_{current_time}_{ARCHITECTURE}"
        # Initialize wandb
        wandb.init(
            project="gpt2-baseline",
            name=run_name,
            config={
                "epochs": EPOCHS,
                "batch_size": TOTAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "sequence_length": SEQUENCE_LENGTH,
                "warmup_steps": WARMUP_STEPS,
                "weight_decay": WEIGHT_DECAY,
                "epsilon": EPSILON,
                "betas": BETAS,
                "use_liger": USE_LIGER,
                "model_config": model_config.__dict__
            }
        )

    download_hellswag_dataset(SOURCE, dirname=DATASET_TARGET_DIR)
    model, raw_model = setup_model(device, ddp, ddp_local_rank, model_config)
    tokenizer = tiktoken.get_encoding('gpt2')
    train_data_loader, valid_data_loader = load_data(
        tokenizer=tokenizer,
        ddp_rank=ddp_rank,
        micro_batch_size=MICRO_BATCH_SIZE,
        max_seq_len=SEQUENCE_LENGTH,
        ddp_world_size=ddp_world_size
    )

    if master_process:
        print("Loaded dataset!")

    optimizer = Optimizer(raw_model, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=WEIGHT_DECAY)
    scheduler = CosineScheduler(max_lr=LEARNING_RATE, warmup_steps=WARMUP_STEPS, min_lr=MIN_LR, total_steps=TOTAL_STEPS)
    grad_accumulation_steps = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * SEQUENCE_LENGTH * ddp_world_size)

    training_config = {
        'ddp': ddp,
        'grad_accumulation_steps': grad_accumulation_steps,
    }

    if master_process:
        print('Starting training')
        print(f'Running for epochs: {EPOCHS}')
        print(f'Validation per steps: {VALIDATION_PER_STEPS}')
        print(f'Hellswag steps: {HELLSWAG_STEPS}')
        print(f'Save steps: {SAVE_STEPS}')
        print(f'Total batch size: {TOTAL_BATCH_SIZE}')
        print(f'Micro batch size: {MICRO_BATCH_SIZE}')
        print(f'Grad accumulation steps: {grad_accumulation_steps}')
        print(f'Sequence length: {SEQUENCE_LENGTH}')
        print(f'Total steps: {TOTAL_STEPS}')
        print(f'Warmup steps: {WARMUP_STEPS}')
        print(f'Learning rate: {LEARNING_RATE}')
        print(f'Weight decay: {WEIGHT_DECAY}')
        print(f'Epsilon: {EPSILON}')
        print(f'Betas: {BETAS}')
        print(f'DDP: {ddp}')
        print(f'DDP rank: {ddp_rank}')
        print(f'DDP local rank: {ddp_local_rank}')
        print(f'DDP world size: {ddp_world_size}')
        print(f'Device: {device}')
        print(f'Save on last: {SAVE_ON_LAST}')
        print(f'Use liger: {USE_LIGER}')

    train_start = time.time()
    tokens_per_batch = MICRO_BATCH_SIZE * SEQUENCE_LENGTH * grad_accumulation_steps * ddp_world_size
    total_tokens = 0

    torch.set_float32_matmul_precision('high')
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        pass
    for epoch in range(EPOCHS):  # TODO only for testing and overfitting
        if master_process:
            print(f"Epoch {epoch}")
        for step in range(scheduler.total_steps):
            last_step = (step == scheduler.total_steps - 1)
            start = time.time()

            if (step > 0 and step % VALIDATION_PER_STEPS == 0) or (last_step and SAVE_ON_LAST):
                total_valid_loss = process_validation(model, valid_data_loader, LOG_FILE, step, device, ddp, master_process)
                if master_process:
                    wandb.log({
                        "validation_loss": total_valid_loss.item(),
                        "step": step
                    })

            if (step > 0 and step % HELLSWAG_STEPS == 0 or (last_step and SAVE_ON_LAST)):
                with torch.compiler.disable():
                    total, correct, correct_normalized = evaluate_hellswag(model, device, DATASET_TARGET_DIR, ddp_world_size, ddp_rank)
                    if ddp:
                        dist.all_reduce(total, op=dist.ReduceOp.SUM)
                        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
                        dist.all_reduce(correct_normalized, op=dist.ReduceOp.SUM)

                    if master_process:
                        wandb.log({
                            "hellswag_acc": correct / total,
                            "hellswag_acc_normalized": correct_normalized / total,
                            "step": step
                        })

            if master_process and ((step > 0 and step % SAVE_STEPS == 0) or (last_step and SAVE_ON_LAST)):
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': total_valid_loss.item()
                }
                torch.save(checkpoint, os.path.join(RESULT_DIR, f"gpt_2_model_{step:05d}.pt"))

            model.train()
            optimizer.zero_grad()

            loss_accumulated = process_micro_batch(
                model,
                train_data_loader,
                device,
                training_config
            )

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
                if step % PRINT_STEPS == 0:
                    print(f'<Step {step}> loss:{loss_accumulated}, lr: {optimizer.lr}, norm:{norm}, tok/sec:{tokens_per_sec:.2f}, time:{(end - start)*1000:.2f} ms')
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{step} train {loss_accumulated:.6f}\n")
                wandb.log({
                    "train_loss": loss_accumulated,
                    "learning_rate": optimizer.lr,
                    "gradient_norm": norm,
                    "tokens_per_second": tokens_per_sec,
                    "step": step
                })

        epoch_end = time.time()
        if master_process:
            if step % PRINT_STEPS == 0:
                print(f'<Epoch {epoch}> loss:{loss_accumulated} tok/sec:{tokens_per_sec:.2f}, time:{(epoch_end - start)*1000:.2f} ms')
                with open(LOG_FILE, "a") as f:
                    f.write(f"{step} epoch {loss_accumulated:.6f}\n")
        train_data_loader.reset()
        valid_data_loader.reset()

    train_end = time.time()
    print(f'Training took: {(train_end - train_start):.2f} sec, avg tok/sec: {total_tokens / (train_end - train_start):.2f}')

    if master_process:
        wandb.finish()

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    train_model()
