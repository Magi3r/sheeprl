import torch


@torch.no_grad()
def adaptive_gradient_clipping(parameters, clipping=0.3, eps=1e-3):
    """
    PyTorch implementation of DreamerV3 Adaptive Gradient Clipping.

    Args:
        parameters: iterable of model parameters
        clipping: lambda in the paper (0.3 for DreamerV3)
        eps: numerical stability (1e-3 in DreamerV3)
    """
    for p in parameters:
        if p.grad is None:
            continue

        param_norm = torch.norm(p.detach())
        grad_norm = torch.norm(p.grad.detach())

        if param_norm == 0 or grad_norm == 0:
            continue

        max_grad_norm = clipping * (param_norm + eps)

        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + eps)
            p.grad.mul_(scale)
