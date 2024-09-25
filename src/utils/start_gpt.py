from transformers import GPT2LMHeadModel
from models.gpt_2_basic import GPT2Basic, GPT2Configuration
import torch
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_keys(title, keys):
    logger.info((f"\n{title}:"))
    for key in sorted(keys):
        logger.info(f"  {key}")


def load_gpt_from_pretrained(config: GPT2Configuration, pretrained_model_name: str, device: torch.device):
    logger.info("Starting to load GPT model from pretrained weights")

    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name).to(device)
    hf_sd = model.state_dict()  # Use the correct variable
    hf_sd_keys = [k for k in hf_sd if not k.endswith(
        '.attn.masked_bias') and not k.endswith('.attn.bias')]  # ignore these, just a buffer

    local_model = GPT2Basic(config)
    our_sd = local_model.state_dict()
    our_sd_keys = [k for k in our_sd.keys() if not k.endswith(".attn.bias")]

    to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight',
                    'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(hf_sd_keys) == len(
        our_sd_keys), "Model has different number of layers"
    for key in hf_sd_keys:
        if not any(key.endswith(tt) for tt in to_transpose):
            assert hf_sd[key].shape == our_sd[
                key].shape, f"Shape mismatch for key {key}, {hf_sd[key].shape}, {our_sd[key].shape}"
            with torch.no_grad():
                our_sd[key].copy_(hf_sd[key])
        else:
            assert hf_sd[key].shape == our_sd[key].t(
            ).shape, f"Shape mismatch for key {key}, {hf_sd[key].shape}, {our_sd[key].t().shape}"
            with torch.no_grad():
                our_sd[key].copy_(hf_sd[key].t())

    logger.info("Model loading complete")
    print(local_model)
    return local_model, model
