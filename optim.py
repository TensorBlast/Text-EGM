import numpy as np


class ScheduledOptim():

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        
        # Store the original learning rates for each param group
        self.original_lrs = [group['lr'] for group in optimizer.param_groups]

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr_scale = self._get_lr_scale()
        
        # Apply warmup scaling while maintaining relative learning rates
        for param_group, original_lr in zip(self._optimizer.param_groups, self.original_lrs):
            param_group['lr'] = original_lr * lr_scale


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def step(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False
        
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_loss
            self.counter = 0
        
        return False


def early_stopping(validation_losses, patience=5, delta=0):
    """Legacy function, kept for backward compatibility"""
    if len(validation_losses) < patience + 1:
        return False
    
    best_loss = min(validation_losses[:-patience])
    current_loss = validation_losses[-1]
    
    if current_loss > best_loss + delta:
        return True
    
    return False