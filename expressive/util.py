from typing import Optional, Tuple

import torch
import torch.nn as nn
import math
from torch import Tensor
from itertools import product
from torch.nn import functional as F


def log1mexp(x):
    assert torch.all(x >= 0)
    EPS = 1e-15
    out = torch.ones_like(x)
    cond1 = x <= 0.6931471805599453094
    out[cond1] = torch.log(-torch.expm1(-x[cond1]) + EPS)
    out[~cond1] = torch.log1p(-torch.exp(-x[~cond1]) + EPS)
    return out

def log_not(log_p: torch.Tensor):
    return log1mexp(-log_p)


def safe_sample_categorical(
    distr: torch.distributions.Categorical, shape: Optional[torch.Size] = None
):
    if shape == None:
        shape = torch.Size([])
    if distr.logits.device.type == "mps":
        # Sample on CPU if using MPS since multinomial is bugged on MPS
        distr_cpu = torch.distributions.Categorical(logits=distr.logits.cpu())
        return distr_cpu.sample(shape).to(distr.logits.device)
    return distr.sample(shape)


def marginal_mode(x_SBD: Tensor, dim: int=0):
    if torch.backends.mps.is_available() and x_SBD.device.type == "mps":
        xc_SBD = x_SBD.cpu()
        xc_SBD = torch.mode(xc_SBD, dim=dim)[0]
        return xc_SBD.to(x_SBD.device)
    return torch.mode(x_SBD, dim=dim)[0]


def true_mode(x_SBD: Tensor):
    """
    Compute the mode of a tensor across all dimensions.
    That is, the most frequently occuring D-dimensional vector, sample-wise
    Has much higher variance than mode_dim_wise, but is more accurate.
    """

    # The unique operation cannot (?) be done batched, so iterate over batch dimension
    x_mode_BD = torch.zeros_like(x_SBD[0])
    for b in range(x_SBD.shape[1]):
        x_SD = x_SBD[:, b]
        if x_SBD.device.type == "mps":
            x_unique_LD, x_counts_LD = torch.unique(x_SD.cpu(), return_counts=True, dim=0)
            x_unique_LD = x_unique_LD.to(x_SD.device)
            x_counts_LD = x_counts_LD.to(x_SD.device)
        else:
            x_unique_LD, x_counts_LD = torch.unique(x_SD, return_counts=True, dim=0)
        x_mode_D = x_unique_LD[torch.argmax(x_counts_LD)]
        x_mode_BD[b] = x_mode_D
    return x_mode_BD


def safe_reward(
    violations_SBY: Tensor,
    beta: float,
    min_exp_val: float = 80,
    max_exp_val: float = 60,
) -> Tuple[Tensor, Tensor]:
    # Numerically stable version of reward function as explained in Section "Numerically stable reward function"
    # Assumes samples are in the first dimension, so (samples, batch, violations)
    weighted_violations_SB = beta * violations_SBY.sum(dim=-1)
    mean_violations_B = torch.mean(weighted_violations_SB, dim=0)
    min_violations_B = torch.min(weighted_violations_SB, dim=0)[0]
    L_B = torch.minimum(mean_violations_B, max_exp_val + min_violations_B)
    unnorm_rewards_SB = torch.exp(
        -torch.clamp(weighted_violations_SB - L_B.unsqueeze(0), max=min_exp_val)
    )
    norm_rewards_SB = torch.exp(-weighted_violations_SB)
    assert (unnorm_rewards_SB > 0.0).all() and (norm_rewards_SB >= 0).all()
    return unnorm_rewards_SB, norm_rewards_SB

def get_device(args):
    # Check for available GPUs
    use_cuda = args.use_cuda and torch.cuda.is_available()
    # Check for MPS (Metal Performance Shaders) availability
    use_mps = args.use_mps and torch.backends.mps.is_available()
    if use_mps:
        # set fallback to True
        import os

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    elif use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    if args.DEBUG:
        torch.autograd.set_detect_anomaly(True)
        print("Debugging mode")
    return device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

def get_models(problem): 
    # Extremely naive computation of models. 
    # Should only be used for the smallest of problems
    num_dims_w, num_classes_w = problem.shape_w()

    # Calculate total number of assignments
    total_assignments = num_classes_w ** num_dims_w

    # Initialize the assignments tensor
    all_assignments_MW = torch.zeros((total_assignments, num_dims_w), dtype=torch.long)

    for i, assignment_W in enumerate(product(range(num_classes_w), repeat=num_dims_w)):
        all_assignments_MW[i] = torch.tensor(assignment_W)

    # Get all y outputs from all assignments
    all_y_outs_MY = problem.y_from_w(all_assignments_MW)
    return all_assignments_MW, all_y_outs_MY

def compute_ece_sampled(hat_w_0_SBW: Tensor, w_0_BW: Tensor, ECE_bins: int, num_classes_w: int) -> float:
    # Note: This only works in full-batch eval. 
    # Compute approximate ECE over concepts
    hat_w_one_hot_SBWD = F.one_hot(hat_w_0_SBW, num_classes=num_classes_w).float()
    # Distribution of w predictions (estimated by averaging over samples because diffusion models are not tractable)
    dist_hat_w_BWD = torch.mean(hat_w_one_hot_SBWD, dim=0)
    return compute_ece(dist_hat_w_BWD, w_0_BW, ECE_bins)

def compute_ece(p_w_BWD: Tensor, w_0_BW: Tensor, ECE_bins: int) -> float:
    # Get the maximum probability for each sample
    max_probs_w_BW, pred_w_BW = torch.max(p_w_BWD, dim=-1)
    # Get the bin boundaries
    bin_boundaries_Mp1 = torch.linspace(0, 1, ECE_bins + 1, device=p_w_BWD.device)

    # Compute the assignment to bins
    bin_assignments_BW = torch.bucketize(max_probs_w_BW, bin_boundaries_Mp1)
    range_M = torch.arange(ECE_bins, device=p_w_BWD.device)
    # Count the number of samples in each bin
    count_card_bin_WM = torch.sum(bin_assignments_BW[:, :, None] == range_M, dim=0)
    # Compute the confidence of each bin
    # TODO: Should this be micro or macro averaging?
    bin_confidences_WM = torch.zeros((p_w_BWD.shape[1], ECE_bins), device=p_w_BWD.device)
    bin_accuracies_WM = torch.zeros_like(bin_confidences_WM)
    for i in range(ECE_bins):
        for w in range(w_0_BW.shape[1]):
            if count_card_bin_WM[w, i] > 0:
                mask = bin_assignments_BW[:, w] == i
                bin_confidences_WM[w, i] = torch.sum(max_probs_w_BW[:, w][mask], dim=0) / count_card_bin_WM[w, i]
                bin_accuracies_WM[w, i] = torch.sum(pred_w_BW[:, w][mask] == w_0_BW[:, w][mask], dim=0) / count_card_bin_WM[w, i]
    # Compute the ECE
    ece_W = torch.sum((count_card_bin_WM / p_w_BWD.shape[0]) * torch.abs(bin_accuracies_WM - bin_confidences_WM), dim=-1)
    return ece_W.mean().item()
