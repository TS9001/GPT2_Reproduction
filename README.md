# GPT2_Reproduction
Reproduction of GPT2, simple clean, redone after Karpaty videos.
I plan to use this repo as a playground for my experiments with LLMs architecture and finally transfer
all my local experiments to this repo.

Code is definatelly not production-grade engineering code but it should be understandable and readable.

# Experiments
1. <DONE> Plain clean GPT2
2. <DONE> LIGER Kernels added for basic operations
3. <DONE> Move to Transformer++ Add ROPE and RMSNorm, SILU..
2. <IN_PROGRESS> NGPT https://arxiv.org/html/2410.01131v1
3. <TODO> HYPERCONNECTIONS https://arxiv.org/abs/2409.19606
4. <TODO> BYTE LATENT TRANSFORMER https://arxiv.org/abs/2412.09871
5. <TODO> Special versions of attentintion architectures
6. <TODO> ....

# Results
<TODO> (links to experiments in wandb will go here)

### Structure
* Optimized for cursor IDE.
* `notebooks` - notebooks for visualization and debugging - basically to track my tought process and ideas and reference them later.
* `src` - source code. Pretty much standard pytorch code.
* `tests` - tests. Usually generated with LLMs acording to specs.

### LIGER Kernels
Liger is very unstable lately, use it wisely.

### Big TODO:
* Fix the code structure to organize experiments better.
