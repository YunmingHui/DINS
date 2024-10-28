import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWeighedFocalLoss(nn.Module):
  def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
    """
    Initialize the Focal Loss class
    :param alpha: Balance factor for each class, should be a vector
    :param gamma: Adjustment factor to focus the model on hard-to-classify samples
    :param reduction: Specifies the output method for the loss, can be 'none', 'mean' or 'sum'
    """
    super(ClassWeighedFocalLoss, self).__init__()
    if alpha is not None:
      if isinstance(alpha, (list, tuple)):
        self.alpha = torch.tensor(alpha)
      else:
        self.alpha = alpha
    else:
      self.alpha = None
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, inputs, targets):
    """
    Compute the Focal Loss
    :param inputs: Logits output by the model
    :param targets: True labels
    """
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # pt is the probability of being predicted as the true class

    # If alpha is a tensor, extend it to the same shape as targets
    if self.alpha is not None:
      alpha = self.alpha[targets.long()].to(inputs.device)
    else:
      alpha = 1.0

    focal_loss = alpha * ((1 - pt) ** self.gamma) * BCE_loss

    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()
    else:
      return focal_loss

# Example usage
# Suppose there is an output and labels of batch_size = 3, and alpha values for three classes
# inputs = torch.randn(3, requires_grad=True)
# targets = torch.tensor([0, 1, 2], dtype=torch.float32)  # Labels for three different classes
# alpha_values = [0.25, 0.5, 0.75]  # Alpha values for different classes

# loss_fn = ClassWeighedFocalLoss(alpha=alpha_values, gamma=2.0)
# loss = loss_fn(inputs, targets)
# print(loss)

class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Initialize the Focal Loss class
    :param alpha: Balance factor, can be a single value or a list of values for each class
    :param gamma: Adjustment factor to focus the model on hard-to-classify samples
    :param reduction: Specifies the output method for the loss, can be 'none', 'mean' or 'sum'
    """
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, inputs, targets):
    """
    Compute the Focal Loss
    :param inputs: Logits output by the model
    :param targets: True labels
    """
    # First, convert logits to probabilities using the sigmoid function
    probs = torch.sigmoid(inputs)
    # Calculate the components of the cross-entropy loss
    pt = probs * targets + (1 - probs) * (1 - targets)
    log_pt = torch.log(pt)
    # Calculate the main body of the Focal Loss
    focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt

    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()
    else:
      return focal_loss

# Example usage
# # Suppose there is an output and labels of batch_size = 3
# inputs = torch.randn(3, requires_grad=True)
# targets = torch.tensor([1, 0, 1], dtype=torch.float32)

# loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
# loss = loss_fn(inputs, targets)
# print(loss)
