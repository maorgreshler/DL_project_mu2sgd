import torch
from torch.optim.optimizer import Optimizer, required
import copy
from ..utils import proj


class AdaptiveMu2SGD(Optimizer):
    """
    A custom implementation of the μ²-SGD optimizer with RMSprop-style adaptive learning rates.
    """

    def __init__(self, params, lr=required, weight_decay=0., momentum=0.9, gamma=0.9, use_alpha_t=False,
                 use_beta_t=False, projection_radius=None, rms_decay=0.99, epsilon=1e-8):
        """
        Same as original μ²-SGD with additional parameters:
        - rms_decay (float, optional): Decay rate for RMSprop moving average. Default is 0.99.
        - epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        """
        defaults = dict(lr=lr, beta=momentum, weight_decay=weight_decay, gamma=gamma,
                        rms_decay=rms_decay, epsilon=epsilon)  # Added rms_decay and epsilon
        super(AdaptiveMu2SGD, self).__init__(params, defaults)

        # Initialize optimizer state for each parameter (same as original)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d_t'] = torch.full_like(p.data, 0.)
                state['current_grad'] = torch.full_like(p.data, 0.)
                state['correction_grad'] = torch.full_like(p.data, 0.)
                # NEW: Add RMSprop second moment estimate
                state['v_t'] = torch.full_like(p.data, epsilon)  # Initialize with epsilon to avoid division issues

        self.gamma = gamma
        self.projection_radius = projection_radius
        self.iter = 0
        self.sum_iter = 0
        self.use_alpha_t = use_alpha_t
        self.use_beta_t = use_beta_t

        # Create a deep copy of parameter groups to maintain intermediate states (same as original)
        self.w = []
        for group in self.param_groups:
            cloned_group = copy.deepcopy(group)
            for i, p in enumerate(group['params']):
                cloned_group['params'][i] = p.clone().detach().requires_grad_(p.requires_grad)
            self.w.append(cloned_group)

    def __setstate__(self, state):
        """
        Set the state of the optimizer (same as original).
        """
        super(AdaptiveMu2SGD, self).__setstate__(state)

    def compute_estimator(self):
        """
        Compute the gradient estimator for the μ²-SGD optimizer with RMSprop update.
        """
        self.iter += 1
        self.sum_iter += self.iter

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            rms_decay = group['rms_decay']  # NEW: Get RMSprop decay rate

            for p in group['params']:
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                # Update state with current gradient and compute the gradient estimator (same as original)
                state = self.state[p]
                state['current_grad'] = p_grad.detach()
                if self.use_beta_t:
                    beta = 1 / self.iter
                    state['d_t'] = (state['current_grad'] + (1. - beta) * (
                            state['d_t'] - state['correction_grad'])).detach()
                else:
                    state['d_t'] = (state['current_grad'] + (1. - group['beta']) * (
                            state['d_t'] - state['correction_grad'])).detach()

                # NEW: Update RMSprop second moment estimate
                state['v_t'] = rms_decay * state['v_t'] + (1 - rms_decay) * state['d_t'].pow(2)

    def step(self, closure=None):
        """
        Performs a single optimization step with adaptive learning rates.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update correction gradients (same as original)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                state = self.state[p]
                state['correction_grad'] = p_grad.detach()

        # Update parameters using the gradient estimator with adaptive scaling
        for group, group_w in zip(self.param_groups, self.w):
            lr = group['lr']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']  # NEW: Get epsilon for RMSprop

            for p, pw in zip(group['params'], group_w['params']):
                if p.grad is None:
                    continue
                p_grad = p.grad.data
                if weight_decay != 0:
                    p_grad.add_(weight_decay, p.data)

                # Apply gradient-based update with adaptive scaling
                state = self.state[p]

                # NEW: Apply RMSprop-style adaptive scaling
                # Instead of: pw.data.add_(state['d_t'], alpha=-lr)
                adaptive_d_t = state['d_t'] / (state['v_t'].sqrt() + epsilon)
                pw.data.add_(adaptive_d_t, alpha=-lr)

                # Apply projection if specified (same as original)
                if self.projection_radius:
                    pw.data = proj(pw.data, self.projection_radius)

                # Update parameters using alpha or gamma interpolation (same as original)
                if self.use_alpha_t:
                    a = self.iter / self.sum_iter
                    b = (self.sum_iter - self.iter) / self.sum_iter
                    p.data = a * pw.data + b * p.data
                else:
                    p.data = self.gamma * pw.data + (1 - self.gamma) * p.data

        return loss