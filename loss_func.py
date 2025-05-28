import torch
import numpy as np

def trimmed_mean_loss(loss, trim_ratio=0.1):
    # loss: tensor of shape (batch_size, ...)
    loss_flat = loss.reshape(-1)
    n = loss_flat.numel()
    k = int(trim_ratio * n)
    if n < 2 * k + 1:
        # Not enough elements to trim, just return mean
        return loss_flat.mean()
    loss_sorted, _ = torch.sort(loss_flat)
    trimmed = loss_sorted[k : n - k]
    return trimmed.mean()

def smooth_custom_loss(y_pred, y_true, c=1.5, k=10.0, clamp_pred=float('inf')):
    """
    Gradient-friendly version of the custom stock prediction loss.
    
    Args:
        y_pred (Tensor): Predicted values (float tensor).
        y_true (Tensor): True values (float tensor).
        c (float): Penalty base for incorrect predictions.
        k (float): Sharpness of sigmoid approximation for sign agreement.
        
    Returns:
        Tensor: Loss values.
    """
    # Cap y_true at a maximum value of 1
    y_pred = torch.clamp(y_pred, max=clamp_pred, min=-clamp_pred)

    # Element-wise sign agreement
    agreement = y_pred * y_true

    # Soft gating functions
    correctness_gate = torch.sigmoid(k * agreement)
    wrongness_gate = 1.0 - correctness_gate

    # Base reward and penalty terms
    reward_term = -agreement
    penalty_term = -agreement * torch.exp(torch.abs(y_pred) * np.log(c))

    # Combine them with smooth gates
    loss = correctness_gate * reward_term + wrongness_gate * penalty_term

    # Reduce the loss to a scalar (mean of the batch losses)
    # return trimmed_mean_loss(loss, trim_ratio=0.4)
    # return torch.median(loss)
    return torch.mean(loss)
