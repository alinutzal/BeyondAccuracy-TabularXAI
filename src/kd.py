"""
Knowledge Distillation utilities for tabular data.

This module provides functions for knowledge distillation from teacher models
(e.g., XGBoost) to student deep learning models (MLP, Transformer).

References:
    Hinton, G., Vinyals, O., & Dean, J. (2015). 
    "Distilling the Knowledge in a Neural Network"
"""

import torch
import torch.nn as nn
from typing import Optional


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    temperature: float = 2.0,
    alpha: float = 0.7
) -> torch.Tensor:
    """
    Compute knowledge distillation loss.
    
    The distillation loss combines soft targets from the teacher with hard labels:
    L_distill = α * KL(P_teacher || P_student) + (1-α) * CE(y, P_student)
    
    Where:
    - P_teacher = softmax(logits_teacher / T): Teacher's soft targets
    - P_student = softmax(logits_student / T): Student's soft predictions
    - T: Temperature parameter (makes distributions "softer")
    - y: Ground truth labels
    - α: Weight for distillation loss (lambda in original paper)
    
    Args:
        student_logits: Raw logits from student model (batch_size, num_classes)
        teacher_logits: Raw logits from teacher model (batch_size, num_classes)
        labels: Ground truth labels (batch_size,)
        criterion: Loss function for hard labels (e.g., CrossEntropyLoss)
        temperature: Temperature for soft targets (default: 2.0)
        alpha: Weight for distillation loss, range [0, 1] (default: 0.7)
            - Higher values (0.8-0.9) rely more on teacher
            - Lower values (0.5-0.6) rely more on ground truth
    
    Returns:
        Combined distillation loss (scalar tensor)
    
    Example:
        >>> student_logits = model(batch_X)
        >>> loss = compute_distillation_loss(
        ...     student_logits, teacher_logits, labels, 
        ...     criterion=nn.CrossEntropyLoss(),
        ...     temperature=2.0, alpha=0.7
        ... )
    """
    # Apply temperature scaling for soft targets
    student_log_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs_temp = nn.functional.softmax(teacher_logits / temperature, dim=1)
    
    # KL divergence loss (with temperature scaling squared)
    # The temperature squared factor maintains gradient magnitude
    kl_loss = nn.functional.kl_div(
        student_log_probs,
        teacher_probs_temp,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard label loss (cross-entropy)
    ce_loss = criterion(student_logits, labels)
    
    # Combined loss: weighted sum of distillation and hard label losses
    loss = alpha * kl_loss + (1.0 - alpha) * ce_loss
    
    return loss


def compute_consistency_penalty(
    student_logits: torch.Tensor,
    inputs: torch.Tensor,
    top_k_features: list,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute consistency penalty on important features.
    
    This penalty encourages the model to be more sensitive to important features
    by maximizing the Jacobian norm (gradient magnitude) for those features.
    
    The penalty is computed as the mean squared gradient of each output class
    with respect to the top-k most important input features (e.g., identified
    by SHAP values from the teacher model).
    
    Args:
        student_logits: Raw logits from student model (batch_size, num_classes)
        inputs: Input features with gradient tracking enabled (batch_size, num_features)
        top_k_features: List of feature indices to compute penalty on
        device: Device for tensor operations ('cpu' or 'cuda')
    
    Returns:
        Jacobian penalty (scalar tensor)
        
    Example:
        >>> # Enable gradient tracking for inputs
        >>> batch_X.requires_grad_(True)
        >>> outputs = model(batch_X)
        >>> penalty = compute_consistency_penalty(
        ...     outputs, batch_X, top_k_features=[0, 3, 7], device='cuda'
        ... )
        >>> # Use negative penalty to encourage larger gradients
        >>> loss = base_loss - penalty_weight * penalty
    """
    num_classes = student_logits.shape[1]
    jacobian_penalty = 0.0
    
    for class_idx in range(num_classes):
        # Get gradients of output w.r.t. input for specific class
        grad_outputs = torch.zeros_like(student_logits)
        grad_outputs[:, class_idx] = 1.0
        
        grads = torch.autograd.grad(
            outputs=student_logits,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute norm only for top-k features (indices)
        top_k_indices = torch.tensor(top_k_features, device=device, dtype=torch.long)
        top_k_grads = grads[:, top_k_indices]
        jacobian_penalty += torch.mean(top_k_grads ** 2)
    
    return jacobian_penalty


def should_enable_distillation(
    distillation_config: dict,
    teacher_probs: Optional[torch.Tensor]
) -> bool:
    """
    Check if distillation should be enabled.
    
    Args:
        distillation_config: Distillation configuration dictionary
        teacher_probs: Teacher probabilities/logits (can be None)
    
    Returns:
        True if distillation should be enabled, False otherwise
    """
    return distillation_config.get('enabled', False) or teacher_probs is not None


def get_distillation_params(distillation_config: dict, enabled: bool = True):
    """
    Extract distillation parameters from configuration.
    
    Args:
        distillation_config: Distillation configuration dictionary
        enabled: Whether distillation is enabled
    
    Returns:
        Tuple of (alpha, temperature, consistency_config) where:
        - alpha: Weight for distillation loss (0.7 default)
        - temperature: Temperature for soft targets (2.0 default)
        - consistency_config: Configuration dict for consistency penalty
    """
    if not enabled:
        return 0.0, 1.0, {}
    
    alpha = float(distillation_config.get('lambda', 0.7))
    temperature = float(distillation_config.get('temperature', 2.0))
    consistency_config = distillation_config.get('consistency_penalty', {})
    
    return alpha, temperature, consistency_config
