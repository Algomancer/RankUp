import torch
import torch.nn as nn
import torch.nn.functional as F

class RankUpFunction(torch.autograd.Function):
    """
    This custom Function passes 'x' forward unchanged, then
    in backward it projects out the component of grad that lowers the rank.
    """
    @staticmethod
    def forward(ctx, x, smooth_rank_func):
        # Save tensors (and any needed functions) for backward
        ctx.save_for_backward(x)
        ctx.smooth_rank_func = smooth_rank_func
        # Return x unchanged
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        smooth_rank_func = ctx.smooth_rank_func

        # We only need first-order gradients wrt x here, so we do a small subgraph:
        with torch.enable_grad():
            # clone/detach x so we can compute ∇R
            x_for_rank = x.detach().clone().float().requires_grad_(True)
            rank_val = smooth_rank_func(x_for_rank)
            rank_grad = torch.autograd.grad(
                outputs=rank_val,
                inputs=x_for_rank,
                grad_outputs=torch.ones_like(rank_val),
                retain_graph=False,
                create_graph=False
            )[0]

        # If rank_grad is very small, skip modifying the gradient
        rg_norm_sq = rank_grad.pow(2).sum()
        if rg_norm_sq < 1e-20:
            return grad_output, None

        # Dot product of ∇R with ∇L
        dot_val = (rank_grad * grad_output).sum()
        if dot_val > 0:
            # This means the update direction -grad_output is "rank-lowering"
            # so we project it out:
            beta = -dot_val / rg_norm_sq
            grad_output = grad_output + beta * rank_grad

        # Return gradient wrt x, and None for smooth_rank_func
        return grad_output, None


class RankUp(nn.Module):
    def __init__(self, smooth_rank_func=None, eps=1e-12):
        super().__init__()
        # default: entropy.exp of SVD singular values
        self.eps = eps
        self.smooth_rank_func = smooth_rank_func or self._smooth_rank

    def _smooth_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default 'smooth rank' = exp(Entropy of singular values).
        """
        s = torch.linalg.svdvals(x)  # singular values
        s_sum = s.sum() + self.eps
        p = s / s_sum
        log_p = (p + self.eps).log()
        entropy = -(p * log_p).sum()
        return entropy.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RankUpFunction.apply(x, self.smooth_rank_func)

