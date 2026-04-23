import torch
import torch.nn as nn
import numpy as np

class CryostasisManager:
    """
    v0.8.0 Feature: The Gradient Vault / Cryostasis Mechanism.
    Autonomously detects when neurons achieve harmonic resonance (error < 2^-9)
    and permanently zeros their gradients. 
    Prevents catastrophic forgetting during continued training/fine-tuning.
    """
    def __init__(self, model, threshold=0.001953125): # 2^-9 threshold
        self.model = model
        self.threshold = threshold
        self.locked_masks = {}
        
        # Initialize open masks for all weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.locked_masks[name] = torch.ones_like(param.data)
                
    def check_and_lock(self, layer_errors, layer_name_prefix):
        """
        layer_errors: A tensor of errors corresponding to the neurons in a layer.
        """
        with torch.no_grad():
            for name, mask in self.locked_masks.items():
                if layer_name_prefix in name:
                    if mask.dim() == 2:
                        for i, err in enumerate(layer_errors):
                            if err < self.threshold and i < mask.shape[0]:
                                mask[i] = 0.0 
                    
    def apply_gradient_vault(self):
        """
        Call this immediately after loss.backward() and before optimizer.step()
        Flushes the gradients of permanently locked neurons to zero.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.locked_masks:
                param.grad.data *= self.locked_masks[name]


class DynamicBrakingLoss(nn.Module):
    """
    v0.3.2 Feature: Active Brake
    Modulates the loss gradient magnitude based on phase correlation.
    Allows rapid initial exploration, then stabilizes fine-tuning as it locks in.
    """
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        
    def forward(self, outputs, targets, model_confidence):
        base_loss = self.base_loss_fn(outputs, targets)
        
        # use avg confidence directly from helix output
        brake_factor = torch.clamp((model_confidence - 0.387) / 0.113, min=0.001, max=1.0)
        
        return base_loss * brake_factor.detach()


class MnemonicShieldLR:
    """
    v0.6.4 Feature: Mnemonic Shielding
    Context-aware LR policy that protects established memories during continuous learning.
    """
    def __init__(self, optimizer, base_lr, shield_floor=0.015625): # 2^-6
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.shield_floor = shield_floor
        self.seen_tasks = set()
        
    def step(self, task_id):
        is_old = task_id in self.seen_tasks
        self.seen_tasks.add(task_id)
        lr = self.shield_floor if is_old else self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
