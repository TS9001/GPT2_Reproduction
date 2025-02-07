# GPT2_Reproduction
Reproduction of GPT2, simple clean, redone after Karpaty videos.
I plan to use this repo as a playground for my experiments with LLMs architecture and finally transfer
all my local experiments to this repo.

Code is definatelly not production-grade engineering code but it should be understandable and readable.

# Experiments
1. <DONE> Plain clean GPT2
2. <DONE> LIGER Kernels added for basic operations
3. <DONE> Move to Transformer++ Add ROPE and RMSNorm, SILU..
2. <DONE> NGPT https://arxiv.org/html/2410.01131v1 - Unable to reproduce speedups from paper as rest of the internet.
3. <TODO> HYPERCONNECTIONS https://arxiv.org/abs/2409.19606
4. <TODO> BYTE LATENT TRANSFORMER https://arxiv.org/abs/2412.09871
5. <TODO> Special versions of attentintion architectures
6. <TODO> ....

# Results
The results are approximate. I just want to track if architecture works as expected.
* [LIGER version of GPT2](https://api.wandb.ai/links/ttomassikora-gpt_experiments/ywonl9vr)
* [Transformer++ version of GPT2](https://api.wandb.ai/links/ttomassikora-gpt_experiments/w8uw20hi)

### Structure
* Optimized for cursor IDE.
* `notebooks` - notebooks for visualization and debugging - basically to track my tought process and ideas and reference them later.
* `src` - source code. Pretty much standard pytorch code.
* `tests` - tests. Usually generated with LLMs acording to specs.

## Notes
### Benchmarks
Sometimes i benchmark something. Take it with a grain of salt. My GPU is running on steam engine and my OS is Windows with WSL (ok i know i am greavely ashamed).
I usualy train on remote that i have to pay myself, as someone from central europe, it is rather expensive. That is why I like to benchmark little bit before i change something.
This is not to measure anything, just to see if my intuition is correct and underlying system works as I expect. Also you know the saying: "Measure twice, cut once". (or "Better safe than sorry" in other words).

### LIGER Kernels
Liger is very unstable lately, use it wisely. In addition, liger is optimized for small batch sizes and its speed-up on scale of experiments that i do is not that big.
On the other hand, liger is good also for verification of correctness of the code.

### Big TODO:
* Fix the code structure to organize experiments better. I do this in my free time. I do not have time to make this code production-grade, but I assure you,
one day I will take time off and make it shiny as a diamond.
