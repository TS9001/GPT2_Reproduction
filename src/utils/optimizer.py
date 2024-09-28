import torch


class Optimizer:
    def __init__(self, model, lr, betas, eps, weight_decay):
        self.lr = lr
        param_groups = model.parameters()
        if weight_decay > 0:
            params = {
                param_name: param for param_name,
                param in model.named_parameters() if param.requires_grad
            }

            decay_params = [
                param for _,
                param in params.items() if param.dim() >= 2
            ]

            no_decay_params = [
                param for _,
                param in params.items() if param.dim() < 2]

            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0}
            ]
        # Defaultly using AdamW
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
