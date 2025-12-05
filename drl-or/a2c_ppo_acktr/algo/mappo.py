# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical
# import numpy as np


# class ValueNorm:
#     """
#     PopArt value normalization from the paper
#     Normalizes value targets during training for stability
#     """
#     def __init__(self, input_shape, device, epsilon=1e-5, beta=0.99995):
#         self.epsilon = epsilon
#         self.beta = beta
#         self.device = device
        
#         self.mean = torch.zeros(input_shape).to(device)
#         self.var = torch.ones(input_shape).to(device)
#         self.std = torch.ones(input_shape).to(device)
#         self.count = epsilon

#     def update(self, x):
#         """Update running statistics"""
#         batch_mean = torch.mean(x, dim=0)
#         batch_var = torch.var(x, dim=0, unbiased=False)
#         batch_count = x.shape[0]
        
#         # Update with exponential moving average
#         self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
#         self.var = self.beta * self.var + (1 - self.beta) * batch_var
#         self.std = torch.sqrt(self.var + self.epsilon)
#         self.count += batch_count

#     def normalize(self, x):
#         """Normalize values"""
#         return (x - self.mean) / (self.std + self.epsilon)

#     def denormalize(self, x):
#         """Denormalize values"""
#         return x * (self.std + self.epsilon) + self.mean


# class MAPPO:
#     """
#     Multi-Agent Proximal Policy Optimization (MAPPO)
    
#     Based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
#     https://github.com/marlbenchmark/on-policy
    
#     Key features from the paper:
#     1. Value normalization (PopArt) for stable training
#     2. Advantage normalization
#     3. Value clipping
#     4. Huber loss for value function
#     5. Proper hyperparameters tuned for MARL
#     """
    
#     def __init__(self,
#                  actor_critics,
#                  clip_param=0.2,
#                  ppo_epoch=15,
#                  num_mini_batch=1,
#                  value_loss_coef=1.0,
#                  entropy_coef=0.01,
#                  lr=5e-4,
#                  eps=1e-5,
#                  max_grad_norm=10.0,
#                  use_clipped_value_loss=True,
#                  use_value_normalization=True,
#                  use_huber_loss=True,
#                  huber_delta=10.0,
#                  use_linear_lr_decay=False,
#                  use_popart=True,
#                  device='cpu'):
#         """
#         Args:
#             actor_critics: List of actor-critic networks (one per agent)
#             clip_param: PPO clipping parameter (epsilon), recommended: 0.2
#             ppo_epoch: Number of PPO epochs per update, recommended: 5-15
#             num_mini_batch: Number of mini-batches, recommended: 1
#             value_loss_coef: Value loss coefficient
#             entropy_coef: Entropy coefficient for exploration
#             lr: Learning rate
#             eps: Adam epsilon
#             max_grad_norm: Max gradient norm for clipping
#             use_clipped_value_loss: Whether to clip value loss
#             use_value_normalization: Use PopArt normalization (recommended: True)
#             use_huber_loss: Use Huber loss for value function (recommended: True)
#             huber_delta: Huber loss delta
#             use_linear_lr_decay: Linearly decay learning rate
#             use_popart: Use PopArt (better than simple normalization)
#             device: torch device
#         """
        
#         self.actor_critics = actor_critics
#         self.num_agents = len(actor_critics)
#         self.device = device

#         # Hyperparameters (tuned based on MAPPO paper)
#         self.clip_param = clip_param
#         self.ppo_epoch = ppo_epoch
#         self.num_mini_batch = num_mini_batch
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#         self.max_grad_norm = max_grad_norm
        
#         self.use_clipped_value_loss = use_clipped_value_loss
#         self.use_value_normalization = use_value_normalization
#         self.use_huber_loss = use_huber_loss
#         self.huber_delta = huber_delta
#         self.use_linear_lr_decay = use_linear_lr_decay
#         self.use_popart = use_popart

#         # Learning rate and optimizer
#         self.lr = lr
#         self.eps = eps
        
#         # Create optimizers for each agent
#         self.optimizers = []
#         for actor_critic in actor_critics:
#             optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
#             self.optimizers.append(optimizer)
        
#         # Value normalization (PopArt)
#         if self.use_value_normalization:
#             if self.use_popart:
#                 self.value_normalizers = [ValueNorm(1, device) for _ in range(self.num_agents)]
#             else:
#                 # Simple running mean/std
#                 self.value_normalizers = [ValueNorm(1, device) for _ in range(self.num_agents)]

#     def update(self, rollouts_list, agent_masks=None):
#         """
#         Update all agents using MAPPO
        
#         Args:
#             rollouts_list: List of RolloutStorage for each agent
#             agent_masks: Tensor indicating which agents are active (on path)
        
#         Returns:
#             value_loss, action_loss, dist_entropy (averaged over agents and updates)
#         """
        
#         # Compute advantages for each agent with normalization
#         advantages_list = []
#         for i, rollouts in enumerate(rollouts_list):
#             # Get returns and value predictions
#             returns = rollouts.returns[:-1]
#             value_preds = rollouts.value_preds[:-1]
            
#             # Update value normalizer
#             if self.use_value_normalization:
#                 self.value_normalizers[i].update(returns)
            
#             # Compute advantages
#             advantages = returns - value_preds
            
#             # Normalize advantages (per-agent, following paper)
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
#             advantages_list.append(advantages)

#         # Training metrics
#         total_value_loss = 0
#         total_action_loss = 0
#         total_dist_entropy = 0
#         total_ratio = 0
#         num_updates = 0

#         # Multiple epochs over the data (following paper: 5-15 epochs)
#         for epoch in range(self.ppo_epoch):
            
#             # Update each agent
#             for agent_id in range(self.num_agents):
#                 rollouts = rollouts_list[agent_id]
#                 advantages = advantages_list[agent_id]
                
#                 # Data generator (supports mini-batches)
#                 if self.num_mini_batch > 1:
#                     if self.actor_critics[agent_id].is_recurrent:
#                         data_generator = rollouts.recurrent_generator(
#                             advantages, self.num_mini_batch)
#                     else:
#                         data_generator = rollouts.feed_forward_generator(
#                             advantages, self.num_mini_batch)
#                 else:
#                     # Single batch (recommended in paper for small-scale)
#                     data_generator = rollouts.feed_forward_generator(advantages, 1)

#                 # Update over mini-batches
#                 for sample in data_generator:
#                     obs_batch, recurrent_hidden_states_batch, condition_states_batch, \
#                         actions_batch, value_preds_batch, return_batch, \
#                         old_action_log_probs_batch, adv_targ, masks_batch = sample

#                     # Evaluate actions with current policy
#                     values, action_log_probs, dist_entropy, _ = \
#                         self.actor_critics[agent_id].evaluate_actions(
#                             obs_batch.unsqueeze(1), 
#                             recurrent_hidden_states_batch.unsqueeze(0), 
#                             condition_states_batch.unsqueeze(1), 
#                             actions_batch.unsqueeze(1))
                    
#                     action_log_probs = action_log_probs.squeeze(1)
#                     values = values.squeeze(1)

#                     # ============================================================
#                     # POLICY LOSS (with PPO clipping)
#                     # ============================================================
#                     ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
#                     surr1 = ratio * adv_targ
#                     surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 
#                                        1.0 + self.clip_param) * adv_targ
                    
#                     # Mask out agents not on path (if agent_masks provided)
#                     if agent_masks is not None:
#                         mask = masks_batch * agent_masks[agent_id]
#                         action_loss = -(torch.min(surr1, surr2) * mask).sum() / (mask.sum() + 1e-8)
#                     else:
#                         action_loss = -(torch.min(surr1, surr2) * masks_batch).sum() / \
#                                      (masks_batch.sum() + 1e-8)

#                     # ============================================================
#                     # VALUE LOSS (with clipping and Huber loss)
#                     # ============================================================
#                     if self.use_value_normalization:
#                         # Normalize targets
#                         return_batch_normalized = self.value_normalizers[agent_id].normalize(
#                             return_batch)
#                     else:
#                         return_batch_normalized = return_batch
                    
#                     if self.use_clipped_value_loss:
#                         # Value clipping (prevents large value updates)
#                         value_pred_clipped = value_preds_batch + \
#                             (values - value_preds_batch).clamp(-self.clip_param, 
#                                                                self.clip_param)
                        
#                         if self.use_huber_loss:
#                             value_losses = self._huber_loss(
#                                 values, return_batch_normalized, self.huber_delta)
#                             value_losses_clipped = self._huber_loss(
#                                 value_pred_clipped, return_batch_normalized, self.huber_delta)
#                         else:
#                             value_losses = (values - return_batch_normalized).pow(2)
#                             value_losses_clipped = (value_pred_clipped - 
#                                                    return_batch_normalized).pow(2)
                        
#                         value_loss = 0.5 * torch.max(value_losses, 
#                                                      value_losses_clipped).mean()
#                     else:
#                         if self.use_huber_loss:
#                             value_loss = self._huber_loss(
#                                 values, return_batch_normalized, self.huber_delta).mean()
#                         else:
#                             value_loss = 0.5 * (return_batch_normalized - values).pow(2).mean()

#                     # ============================================================
#                     # TOTAL LOSS
#                     # ============================================================
#                     total_loss = (value_loss * self.value_loss_coef + 
#                                  action_loss - 
#                                  dist_entropy * self.entropy_coef)

#                     # Backward and optimize
#                     self.optimizers[agent_id].zero_grad()
#                     total_loss.backward()
                    
#                     # Gradient clipping
#                     nn.utils.clip_grad_norm_(
#                         self.actor_critics[agent_id].parameters(),
#                         self.max_grad_norm)
                    
#                     self.optimizers[agent_id].step()

#                     # Accumulate metrics
#                     total_value_loss += value_loss.item()
#                     total_action_loss += action_loss.item()
#                     total_dist_entropy += dist_entropy.item()
#                     total_ratio += ratio.mean().item()
#                     num_updates += 1

#         # Average metrics
#         avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
#         avg_action_loss = total_action_loss / num_updates if num_updates > 0 else 0
#         avg_dist_entropy = total_dist_entropy / num_updates if num_updates > 0 else 0
#         avg_ratio = total_ratio / num_updates if num_updates > 0 else 0

#         return avg_value_loss, avg_action_loss, avg_dist_entropy, avg_ratio

#     def _huber_loss(self, pred, target, delta):
#         """
#         Huber loss for robust value function training
#         Less sensitive to outliers than MSE
#         """
#         error = pred - target
#         condition = torch.abs(error) <= delta
#         small_error = 0.5 * error ** 2
#         large_error = delta * (torch.abs(error) - 0.5 * delta)
#         return torch.where(condition, small_error, large_error)

#     def lr_decay(self, step, total_steps):
#         """
#         Linear learning rate decay (optional, but recommended in paper)
#         """
#         if self.use_linear_lr_decay:
#             lr_now = self.lr - (self.lr * (step / total_steps))
#             for optimizer in self.optimizers:
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = lr_now

#     def prep_training(self):
#         """Set all networks to training mode"""
#         for actor_critic in self.actor_critics:
#             actor_critic.train()

#     def prep_rollout(self):
#         """Set all networks to evaluation mode"""
#         for actor_critic in self.actor_critics:
#             actor_critic.eval()









"""
MAPPO (Multi-Agent PPO) with Full CTDE Paradigm for DRL-OR Project

CTDE = Centralized Training with Decentralized Execution

Key CTDE Design:
- ACTOR (Decentralized): Uses LOCAL partial state (s_i) + conditional state (c_i)
  -> Only local info needed at execution time (scalable deployment)
- CRITIC (Centralized): Uses GLOBAL state (full network information)
  -> Better value estimation during training (not used at execution)

For DRL-OR:
- Local state (s_i): flow features, neighbor info, local link usage
- Conditional state (c_i): zero-one vector of nodes on current path
- Global state (s): ALL link capacities, ALL link usage, ALL link losses, flow info

Based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Yu et al., 2022)
Paper: https://arxiv.org/abs/2103.01955
Code: https://github.com/marlbenchmark/on-policy

File location: a2c_ppo_acktr/algo/mappo.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# =============================================================================
# PopArt: Preserving Outputs Precisely, while Adaptively Rescaling Targets
# =============================================================================

class PopArt(nn.Module):
    """
    PopArt value normalization with weight preservation.
    
    When normalization statistics change, adjusts linear layer weights
    to keep denormalized outputs unchanged. Critical for stable MARL training.
    
    Reference: "Learning values across many orders of magnitude" (van Hasselt et al., 2016)
    """
    
    def __init__(self, input_shape, output_shape=1, norm_axes=1, 
                 beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()
        
        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Linear layer parameters (trainable)
        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))
        self.bias = nn.Parameter(torch.Tensor(output_shape))
        
        # Running statistics (not trainable)
        self.register_buffer('stddev', torch.ones(output_shape))
        self.register_buffer('mean', torch.zeros(output_shape))
        self.register_buffer('mean_sq', torch.zeros(output_shape))
        self.register_buffer('debiasing_term', torch.tensor(0.0))
        
        self.reset_parameters()
        self.to(device)
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()
    
    def forward(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    def update(self, input_vector):
        """Update statistics and adjust weights to preserve outputs."""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        
        # Store old statistics
        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)
        
        # Compute batch statistics
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
        
        # Update with EMA
        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))
        
        self.stddev.copy_((self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4))
        
        # Get new statistics
        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)
        
        # KEY: Preserve outputs by adjusting weights
        self.weight.data.copy_(self.weight * old_stddev.unsqueeze(1) / new_stddev.unsqueeze(1))
        self.bias.data.copy_((old_stddev * self.bias + old_mean - new_mean) / new_stddev)
    
    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var
    
    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        return (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
    
    def denormalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        return input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]


# =============================================================================
# ValueNorm: Simple value normalization (alternative to PopArt)
# =============================================================================

class ValueNorm(nn.Module):
    """Simple value normalization using running mean and variance."""
    
    def __init__(self, input_shape=1, norm_axes=1, beta=0.99999, 
                 per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()
        
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.register_buffer('running_mean', torch.zeros(input_shape))
        self.register_buffer('running_mean_sq', torch.zeros(input_shape))
        self.register_buffer('debiasing_term', torch.tensor(0.0))
        
        self.reset_parameters()
        self.to(device)
    
    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()
    
    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var
    
    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))
        
        weight = self.beta
        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))
    
    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.running_mean_var()
        return (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
    
    def denormalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.running_mean_var()
        return input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]


# =============================================================================
# Centralized Critic Network (CTDE - uses GLOBAL state)
# =============================================================================

class CentralizedCritic(nn.Module):
    """
    Centralized Critic for CTDE MAPPO.
    
    This critic takes GLOBAL network state as input during TRAINING.
    It is NOT used during execution (only actors are used).
    
    For DRL-OR, global state includes:
    - All link residual capacities (node_num x node_num)
    - All link loss rates (node_num x node_num)
    - Current flow features (src, dst, type, demand)
    - Optionally: agent-specific info for AS (Agent-Specific) variant
    
    Two variants from MAPPO paper:
    - FP (Feature-Pruned): global_state only (removes redundant local info)
    - AS (Agent-Specific): global_state + agent's local observation
    """
    
    def __init__(self, global_state_dim, hidden_size=64, num_layers=2,
                 use_feature_normalization=True, use_orthogonal=True,
                 use_popart=True, device='cpu'):
        """
        Args:
            global_state_dim: Dimension of global state input
            hidden_size: Hidden layer size (default 64)
            num_layers: Number of hidden layers (default 2)
            use_feature_normalization: Normalize input features
            use_orthogonal: Use orthogonal initialization
            use_popart: Use PopArt for value head
            device: torch device
        """
        super(CentralizedCritic, self).__init__()
        
        self.hidden_size = hidden_size
        self._use_feature_normalization = use_feature_normalization
        self._use_orthogonal = use_orthogonal
        self._use_popart = use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Feature normalization layer
        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(global_state_dim)
        
        # Build MLP layers
        layers = []
        input_dim = global_state_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        
        # Value head (with or without PopArt)
        if use_popart:
            self.v_out = PopArt(hidden_size, 1, device=torch.device(device))
        else:
            self.v_out = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        if use_orthogonal:
            self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Orthogonal initialization (recommended by MAPPO paper)."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        if not self._use_popart:
            nn.init.orthogonal_(self.v_out.weight, gain=1)
            nn.init.constant_(self.v_out.bias, 0)
    
    def forward(self, global_state):
        """
        Forward pass to compute state value.
        
        Args:
            global_state: Global network state [batch, global_state_dim]
            
        Returns:
            value: State value estimate [batch, 1]
        """
        if isinstance(global_state, np.ndarray):
            global_state = torch.from_numpy(global_state)
        global_state = global_state.to(**self.tpdv)
        
        if self._use_feature_normalization:
            global_state = self.feature_norm(global_state)
        
        x = self.mlp(global_state)
        value = self.v_out(x)
        
        return value
    
    def get_value(self, global_state):
        """Alias for forward (compatibility)."""
        return self.forward(global_state)


# =============================================================================
# Utility functions
# =============================================================================

def huber_loss(e, d):
    """Huber loss - less sensitive to outliers than MSE."""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    """Mean squared error loss."""
    return e ** 2 / 2


# =============================================================================
# MAPPO with CTDE (Centralized Training, Decentralized Execution)
# =============================================================================

class MAPPO_CTDE:
    """
    Multi-Agent PPO with full CTDE paradigm for DRL-OR.
    
    CTDE Architecture:
    ==================
    - ACTORS (Decentralized): Each agent has its own actor using LOCAL observations
      -> Used during both training and execution
      -> Input: partial state (s_i) + conditional state (c_i)
      
    - CRITIC (Centralized): Single shared critic using GLOBAL state
      -> Used during TRAINING only (not needed at execution)
      -> Input: global network state (all link info, flow info)
    
    This separation allows:
    1. Better value estimation during training (centralized critic sees everything)
    2. Scalable execution (actors only need local info, no communication)
    
    Key MAPPO improvements:
    - PopArt value normalization for stable training
    - Huber loss for robust value learning
    - Value clipping to prevent large updates
    - Advantage normalization per mini-batch
    - Separate actor/critic learning rates
    """
    
    def __init__(self,
                 actor_critics,           # List of actor networks (Policy objects)
                 centralized_critic,      # Single centralized critic (CentralizedCritic)
                 clip_param=0.2,
                 ppo_epoch=15,
                 num_mini_batch=1,
                 value_loss_coef=1.0,
                 entropy_coef=0.01,
                 actor_lr=5e-4,
                 critic_lr=5e-4,
                 eps=1e-5,
                 max_grad_norm=10.0,
                 use_huber_loss=True,
                 huber_delta=10.0,
                 use_clipped_value_loss=True,
                 use_popart=True,
                 use_valuenorm=False,
                 use_linear_lr_decay=True,
                 device='cpu'):
        """
        Args:
            actor_critics: List of actor networks (one per agent)
            centralized_critic: Shared centralized critic network
            clip_param: PPO clipping parameter (epsilon)
            ppo_epoch: Number of PPO epochs per update
            num_mini_batch: Number of mini-batches
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate (can be different from actor)
            eps: Adam epsilon
            max_grad_norm: Max gradient norm for clipping
            use_huber_loss: Use Huber loss for value function
            huber_delta: Huber loss delta parameter
            use_clipped_value_loss: Use clipped value loss
            use_popart: Use PopArt normalization (in critic)
            use_valuenorm: Use simple value normalization
            use_linear_lr_decay: Use linear LR decay
            device: torch device
        """
        self.actors = actor_critics  # Decentralized actors
        self.critic = centralized_critic  # Centralized critic
        self.num_agents = len(actor_critics)
        self.device = device
        
        # PPO hyperparameters
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        
        # Loss coefficients
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Optimizer settings
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        
        # Value loss options
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_linear_lr_decay = use_linear_lr_decay
        
        # Value normalization
        self._use_popart = use_popart
        self._use_valuenorm = use_valuenorm
        
        # Create SEPARATE optimizers for actors and critic
        self.actor_optimizers = []
        for actor in self.actors:
            optimizer = optim.Adam(actor.parameters(), lr=actor_lr, eps=eps)
            self.actor_optimizers.append(optimizer)
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=eps)
        
        # Value normalizer (if not using PopArt in critic)
        if use_valuenorm and not use_popart:
            self.value_normalizer = ValueNorm(1, device=torch.device(device))
        else:
            self.value_normalizer = None
    
    def reset_optimizers(self):
        """Reset all optimizers (useful after loading models)."""
        self.actor_optimizers = []
        for actor in self.actors:
            optimizer = optim.Adam(actor.parameters(), lr=self.actor_lr, eps=self.eps)
            self.actor_optimizers.append(optimizer)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.eps)
    
    def lr_decay(self, step, total_steps):
        """Linear learning rate decay for both actors and critic."""
        if self.use_linear_lr_decay:
            lr_actor_now = self.actor_lr * (1 - step / float(total_steps))
            lr_critic_now = self.critic_lr * (1 - step / float(total_steps))
            
            for optimizer in self.actor_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_actor_now
            
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr_critic_now
    
    def get_values(self, global_state):
        """
        Get value estimates from centralized critic.
        
        Args:
            global_state: Global network state
            
        Returns:
            value: Value estimate
        """
        return self.critic.get_value(global_state)
    
    def cal_value_loss(self, values, value_preds_batch, return_batch, masks_batch):
        """
        Calculate value function loss with optional clipping and Huber loss.
        
        Args:
            values: Current value predictions from centralized critic
            value_preds_batch: Old value predictions
            return_batch: Target returns
            masks_batch: Active masks
            
        Returns:
            value_loss: Computed value loss
        """
        # Normalize returns if using PopArt (handled in critic) or ValueNorm
        if self._use_popart and hasattr(self.critic.v_out, 'update'):
            self.critic.v_out.update(return_batch)
            return_batch_normalized = self.critic.v_out.normalize(return_batch)
        elif self._use_valuenorm and self.value_normalizer is not None:
            self.value_normalizer.update(return_batch)
            return_batch_normalized = self.value_normalizer.normalize(return_batch)
        else:
            return_batch_normalized = return_batch
        
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            
            error_clipped = return_batch_normalized - value_pred_clipped
            error_original = return_batch_normalized - values
            
            if self.use_huber_loss:
                value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
                value_loss_original = huber_loss(error_original, self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)
            
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            error = return_batch_normalized - values
            if self.use_huber_loss:
                value_loss = huber_loss(error, self.huber_delta)
            else:
                value_loss = mse_loss(error)
        
        # Apply masks
        value_loss = (value_loss * masks_batch).sum() / masks_batch.sum().clamp(min=1.0)
        
        return value_loss
    
    def update(self, rollouts, global_states_batch, agent_masks=None):
        """
        Perform CTDE MAPPO update.
        
        CTDE Update Process:
        1. Compute advantages using CENTRALIZED critic values
        2. Update each ACTOR using local observations and advantages
        3. Update CENTRALIZED CRITIC using global states
        
        Args:
            rollouts: List of RolloutStorage objects (one per agent)
            global_states_batch: Global state tensor for critic [num_steps, global_dim]
            agent_masks: Tensor indicating which agents are on path (m_i)
            
        Returns:
            value_loss: Average value loss
            action_loss: Average policy loss
            dist_entropy: Average entropy
            ratio: Average importance sampling ratio
        """
        # ================================================================
        # Step 1: Compute advantages using centralized critic
        # ================================================================
        
        # Get value predictions from centralized critic for all steps
        with torch.no_grad():
            # Use the value_preds stored during rollout (from centralized critic)
            # These should have been computed using global_state
            pass
        
        # Compute advantages for each agent
        all_advantages = []
        for k in range(self.num_agents):
            rollout = rollouts[k]
            
            if self._use_popart and hasattr(self.critic.v_out, 'denormalize'):
                advantages = rollout.returns[:-1] - self.critic.v_out.denormalize(
                    rollout.value_preds[:-1])
            elif self._use_valuenorm and self.value_normalizer is not None:
                advantages = rollout.returns[:-1] - self.value_normalizer.denormalize(
                    rollout.value_preds[:-1])
            else:
                advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
            
            # Normalize advantages (important for MAPPO)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            all_advantages.append(advantages)
        
        # ================================================================
        # Step 2 & 3: Update actors and critic
        # ================================================================
        
        total_value_loss = 0
        total_action_loss = 0
        total_dist_entropy = 0
        total_ratio = 0
        update_count = 0
        
        for epoch in range(self.ppo_epoch):
            # Generate mini-batches
            for mini_batch_idx in range(self.num_mini_batch):
                # ============================================================
                # Update each ACTOR (Decentralized)
                # ============================================================
                for agent_idx in range(self.num_agents):
                    actor = self.actors[agent_idx]
                    rollout = rollouts[agent_idx]
                    advantages = all_advantages[agent_idx]
                    optimizer = self.actor_optimizers[agent_idx]
                    
                    # Get agent-specific mask
                    if agent_masks is not None:
                        agent_mask = agent_masks[agent_idx].item() \
                            if hasattr(agent_masks[agent_idx], 'item') else agent_masks[agent_idx]
                    else:
                        agent_mask = 1.0
                    
                    # Get data generator for this agent
                    if actor.is_recurrent:
                        data_generator = rollout.recurrent_generator(
                            advantages, self.num_mini_batch)
                    else:
                        data_generator = rollout.feed_forward_generator(
                            advantages, self.num_mini_batch)
                    
                    for sample in data_generator:
                        obs_batch, recurrent_hidden_states_batch, condition_states_batch, \
                            actions_batch, value_preds_batch, return_batch, \
                            old_action_log_probs_batch, adv_targ, masks_batch = sample
                        
                        # Evaluate actions with current actor (LOCAL observations)
                        _, action_log_probs, dist_entropy, _ = actor.evaluate_actions(
                            obs_batch.unsqueeze(1),
                            recurrent_hidden_states_batch.unsqueeze(0),
                            condition_states_batch.unsqueeze(1),
                            actions_batch.unsqueeze(1)
                        )
                        action_log_probs = action_log_probs.squeeze(1)
                        
                        # PPO clipped objective
                        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                        surr1 = ratio * adv_targ
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 
                                           1.0 + self.clip_param) * adv_targ
                        
                        # Policy loss with masks
                        action_loss = -(torch.min(surr1, surr2) * masks_batch).sum() / \
                                      masks_batch.sum().clamp(min=1.0)
                        
                        # Actor loss = policy loss - entropy bonus
                        actor_loss = action_loss - dist_entropy * self.entropy_coef
                        
                        # Update actor
                        optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                        optimizer.step()
                        
                        total_action_loss += action_loss.item()
                        total_dist_entropy += dist_entropy.item()
                        total_ratio += ratio.mean().item()
                
                # ============================================================
                # Update CENTRALIZED CRITIC (using global state)
                # ============================================================
                
                # For critic, we use data from agent 0's rollout for indices
                # but value is computed using GLOBAL state
                rollout = rollouts[0]
                advantages = all_advantages[0]
                
                if self.actors[0].is_recurrent:
                    data_generator = rollout.recurrent_generator(
                        advantages, self.num_mini_batch)
                else:
                    data_generator = rollout.feed_forward_generator(
                        advantages, self.num_mini_batch)
                
                for batch_idx, sample in enumerate(data_generator):
                    _, _, _, _, value_preds_batch, return_batch, _, _, masks_batch = sample
                    
                    # Get corresponding global states for this batch
                    batch_size = value_preds_batch.shape[0]
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, global_states_batch.shape[0])
                    global_state_batch = global_states_batch[start_idx:end_idx]
                    
                    # Compute values using CENTRALIZED critic with GLOBAL state
                    values = self.critic.get_value(global_state_batch)
                    values = values.squeeze(-1) if values.dim() > 1 else values
                    
                    # Ensure dimensions match
                    if values.shape[0] != value_preds_batch.shape[0]:
                        values = values[:value_preds_batch.shape[0]]
                    
                    # Value loss
                    value_loss = self.cal_value_loss(
                        values, value_preds_batch, return_batch, masks_batch)
                    
                    # Update critic
                    critic_loss = value_loss * self.value_loss_coef
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    
                    total_value_loss += value_loss.item()
                
                update_count += 1
        
        # Average metrics
        num_updates = max(update_count, 1)
        avg_value_loss = total_value_loss / num_updates
        avg_action_loss = total_action_loss / (num_updates * self.num_agents)
        avg_dist_entropy = total_dist_entropy / (num_updates * self.num_agents)
        avg_ratio = total_ratio / (num_updates * self.num_agents)
        
        return avg_value_loss, avg_action_loss, avg_dist_entropy, avg_ratio
    
    def prep_training(self):
        """Set all networks to training mode."""
        for actor in self.actors:
            actor.train()
        self.critic.train()
    
    def prep_rollout(self):
        """Set all networks to evaluation mode."""
        for actor in self.actors:
            actor.eval()
        self.critic.eval()


# =============================================================================
# Helper function to construct global state for DRL-OR
# =============================================================================

def construct_global_state(link_capa, link_usage, link_losses, 
                          flow_src, flow_dst, flow_type, flow_demand,
                          num_node, num_type, device='cpu'):
    """
    Construct global state for centralized critic.
    
    Global state contains ALL network information:
    - All link residual capacities (normalized)
    - All link loss rates
    - Current flow features (one-hot encoded)
    
    Args:
        link_capa: Link capacity matrix [num_node, num_node]
        link_usage: Link usage matrix [num_node, num_node]
        link_losses: Link loss rate matrix [num_node, num_node]
        flow_src: Source node of current flow
        flow_dst: Destination node of current flow
        flow_type: Service type of current flow
        flow_demand: Bandwidth demand of current flow
        num_node: Number of nodes in network
        num_type: Number of flow types
        device: torch device
        
    Returns:
        global_state: Tensor of global state [global_state_dim]
    """
    # Flatten link residual capacities (normalized by capacity)
    link_residual = []
    for i in range(num_node):
        for j in range(num_node):
            if link_capa[i][j] > 0:
                residual = (link_capa[i][j] - link_usage[i][j]) / link_capa[i][j]
                link_residual.append(max(0, min(1, residual)))
            else:
                link_residual.append(0)
    
    # Flatten link loss rates (already normalized 0-1)
    link_loss = []
    for i in range(num_node):
        for j in range(num_node):
            link_loss.append(link_losses[i][j] / 100.0)  # Assuming loss is in percentage
    
    # Flow features (one-hot encoded)
    type_onehot = [0] * num_type
    type_onehot[flow_type] = 1
    
    src_onehot = [0] * num_node
    src_onehot[flow_src] = 1
    
    dst_onehot = [0] * num_node
    dst_onehot[flow_dst] = 1
    
    # Normalize demand (assuming max demand is ~2000 Kbps based on DRL-OR)
    demand_normalized = [flow_demand / 2000.0]
    
    # Concatenate all components
    global_state = link_residual + link_loss + type_onehot + src_onehot + dst_onehot + demand_normalized
    
    return torch.tensor(global_state, dtype=torch.float32, device=device)


def get_global_state_dim(num_node, num_type):
    """
    Calculate global state dimension.
    
    Global state components:
    - Link residual capacities: num_node * num_node
    - Link loss rates: num_node * num_node
    - Flow type (one-hot): num_type
    - Flow source (one-hot): num_node
    - Flow destination (one-hot): num_node
    - Flow demand (normalized): 1
    
    Returns:
        global_state_dim: Total dimension of global state
    """
    return num_node * num_node * 2 + num_type + num_node * 2 + 1