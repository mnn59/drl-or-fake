# """
# Main training script for MAPPO (Multi-Agent PPO) routing algorithm

# Based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
# Paper: https://arxiv.org/abs/2103.01955
# Code: https://github.com/marlbenchmark/on-policy

# Usage:
#     python3 main_mappo.py --env-name Abi --demand-matrix Abi_500.txt \
#         --log-dir ./log/mappo --model-save-path ./model/mappo
# """

# import copy
# import glob
# import os
# import time
# from collections import deque

# import gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from a2c_ppo_acktr import algo, utils
# # from a2c_ppo_acktr.arguments import get_args  # for ppo
# from a2c_ppo_acktr.arguments import get_mappo_args, print_mappo_config  # for mappo
# from a2c_ppo_acktr.model import Policy
# from a2c_ppo_acktr.storage import RolloutStorage
# from a2c_ppo_acktr.algo.mappo import MAPPO

# from net_env.simenv import NetEnv


# def main():
#     # Get arguments
#     # args = get_args()  # for ppo
#     args = get_mappo_args()  # for mappo
#     print_mappo_config(args)
    
#     # Set random seeds for reproducibility
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     np.random.seed(args.seed)
    
#     if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True

#     # Setup directories
#     log_dir = os.path.expanduser(args.log_dir)
#     eval_log_dir = log_dir + "/eval"
#     utils.cleanup_log_dir(log_dir)
#     utils.cleanup_log_dir(eval_log_dir)
    
#     model_save_path = args.model_save_path
#     model_load_path = args.model_load_path
#     ckpt_step = args.ckpt_steps
    
#     # Setup device
#     torch.set_num_threads(1)
#     device = torch.device("cuda:0" if args.cuda else "cpu")
    
#     print("\n" + "="*70)
#     print("MAPPO Training for SDN Routing")
#     print("="*70)
#     print(f"Device: {device}")
#     print(f"Algorithm: MAPPO (Multi-Agent PPO)")
#     print(f"Based on: 'The Surprising Effectiveness of PPO in MARL'")
#     print("="*70 + "\n")

#     # Setup environment
#     print("Setting up environment...")
#     envs = NetEnv(args) 
#     num_agent, num_node, observation_spaces, action_spaces, num_type = \
#         envs.setup(args.env_name, args.demand_matrix)
#     request, obses = envs.reset()
    
#     print(f"Environment: {args.env_name}")
#     print(f"Number of agents: {num_agent}")
#     print(f"Number of nodes: {num_node}")
#     print(f"Number of flow types: {num_type}")
#     print(f"Observation space: {observation_spaces[0].shape}")
#     print(f"Action space: {action_spaces[0]}\n")

#     # Open log files
#     log_dist_files = []
#     log_demand_files = []
#     log_delay_files = []
#     log_throughput_files = []
#     log_loss_files = []
    
#     for i in range(num_type):
#         log_dist_file = open(f"{log_dir}/dist_type{i}.log", "w", 1)
#         log_dist_files.append(log_dist_file)
#         log_demand_file = open(f"{log_dir}/demand_type{i}.log", "w", 1)
#         log_demand_files.append(log_demand_file)
#         log_delay_file = open(f"{log_dir}/delay_type{i}.log", "w", 1)
#         log_delay_files.append(log_delay_file)
#         log_throughput_file = open(f"{log_dir}/throughput_type{i}.log", "w", 1)
#         log_throughput_files.append(log_throughput_file)
#         log_loss_file = open(f"{log_dir}/loss_type{i}.log", "w", 1)
#         log_loss_files.append(log_loss_file)
    
#     log_globalrwd_file = open(f"{log_dir}/globalrwd.log", "w", 1)
#     log_circle_file = open(f"{log_dir}/circle.log", "w", 1)
#     log_value_loss_file = open(f"{log_dir}/value_loss.log", "w", 1)
#     log_action_loss_file = open(f"{log_dir}/action_loss.log", "w", 1)
#     log_ratio_file = open(f"{log_dir}/ratio.log", "w", 1)

#     # Build actor-critic models for all agents
#     print("Building actor-critic networks...")
#     actor_critics = []
#     rollouts = []
    
#     for i in range(num_agent):
#         actor_critic = Policy(
#             observation_spaces[i].shape, 
#             action_spaces[i], 
#             num_node, 
#             num_node, 
#             num_type,
#             base_kwargs={'recurrent': args.recurrent_policy}
#         )
        
#         # Load parameters if specified
#         if model_load_path is not None:
#             model_file = os.path.join(model_load_path, f'agent{i}.pth')
#             if os.path.exists(model_file):
#                 actor_critic.load_state_dict(torch.load(model_file, map_location=device))
#                 print(f"  Loaded model for agent {i} from {model_file}")
        
#         actor_critic.to(device)
#         actor_critics.append(actor_critic)

#         # Create rollout storage for pre-training
#         rollout = RolloutStorage(
#             args.num_pretrain_steps,
#             observation_spaces[i].shape, 
#             action_spaces[i],
#             actor_critic.recurrent_hidden_state_size, 
#             num_node
#         )
#         rollouts.append(rollout)
#         rollouts[i].obs[0].copy_(obses[i])
#         rollouts[i].to(device)

#     # ============================================================
#     # Create MAPPO agent with IMPROVED hyperparameters from paper
#     # ============================================================
#     print("\nCreating MAPPO agent...")
#     print("Hyperparameters (tuned based on MAPPO paper):")
#     print(f"  Learning rate: {args.lr}")
#     print(f"  Clip param (epsilon): {args.clip_param}")
#     print(f"  PPO epochs: {args.ppo_epoch}")
#     print(f"  Mini batches: {args.num_mini_batch}")
#     print(f"  Value loss coef: {args.value_loss_coef}")
#     print(f"  Entropy coef: {args.entropy_coef}")
#     print(f"  Max grad norm: {args.max_grad_norm}")
#     print(f"  Use value normalization: True (PopArt)")
#     print(f"  Use Huber loss: True")
#     print(f"  Use value clipping: True")
    
#     mappo_agent = MAPPO(
#         actor_critics,
#         clip_param=args.clip_param,
#         ppo_epoch=args.ppo_epoch,
#         num_mini_batch=args.num_mini_batch,
#         value_loss_coef=args.value_loss_coef,
#         entropy_coef=args.entropy_coef,
#         lr=args.lr,
#         eps=args.eps,
#         max_grad_norm=args.max_grad_norm,
#         use_value_normalization=True,  # Critical for stability (Paper Sec 5.1)
#         use_huber_loss=True,            # More robust (Paper recommendation)
#         use_clipped_value_loss=True,    # Prevents large value updates
#         use_popart=True,                # PopArt normalization
#         use_linear_lr_decay=args.use_linear_lr_decay,
#         device=device
#     )

#     # ==================== PRE-TRAINING PHASE ====================
#     print("\n" + "="*70)
#     print("PRE-TRAINING PHASE: Learning shortest path policy")
#     print("="*70)
    
#     mappo_agent.prep_training()  # Set to training mode
#     pretrain_start_time = time.time()
    
#     for epoch in range(args.num_pretrain_epochs):
#         epoch_start_time = time.time()
        
#         for step in range(args.num_pretrain_steps):
#             # Interact with environment
#             with torch.no_grad():
#                 values = [None] * num_agent
#                 actions = [None] * num_agent
#                 action_log_probs = [None] * num_agent
#                 recurrent_hidden_states = [None] * num_agent
#                 condition_states = [None] * num_agent
                
#                 # Generate routing action hop-by-hop
#                 curr_path = [0] * num_node
#                 agents_flag = [0] * num_agent
#                 curr_agent, path = envs.first_agent()
                
#                 while curr_agent is not None and agents_flag[curr_agent] != 1:
#                     for k in path:
#                         curr_path[k] = 1
#                     agents_flag[curr_agent] = 1
                    
#                     condition_state = torch.tensor(curr_path, dtype=torch.float32).to(device)
                    
#                     value, action, action_log_prob, recurrent_hidden_state = \
#                         actor_critics[curr_agent].act(
#                             rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
#                             rollouts[curr_agent].recurrent_hidden_states[
#                                 rollouts[curr_agent].step].unsqueeze(0),
#                             condition_state.unsqueeze(0)
#                         )

#                     values[curr_agent] = value
#                     actions[curr_agent] = action
#                     action_log_probs[curr_agent] = action_log_prob
#                     recurrent_hidden_states[curr_agent] = recurrent_hidden_state
#                     condition_states[curr_agent] = condition_state
#                     curr_agent, path = envs.next_agent(curr_agent, action)
                
#                 # Handle agents not on path
#                 condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
#                 for k in range(num_agent):
#                     if agents_flag[k] != 1:
#                         value, action, action_log_prob, recurrent_hidden_state = \
#                             actor_critics[k].act(
#                                 rollouts[k].obs[rollouts[k].step].unsqueeze(0),
#                                 rollouts[k].recurrent_hidden_states[
#                                     rollouts[k].step].unsqueeze(0),
#                                 condition_state.unsqueeze(0)
#                             )
                
#                         values[k] = value
#                         actions[k] = action
#                         action_log_probs[k] = action_log_prob
#                         recurrent_hidden_states[k] = recurrent_hidden_state
#                         condition_states[k] = condition_state

#             # Step environment (no real simulation during pre-training)
#             gfactors = [0.] * num_agent  # Local rewards only
#             obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, \
#                 globalrwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
            
#             # Log metrics
#             print(delta_dist, file=log_dist_files[rtype])
#             print(delta_demand, file=log_demand_files[rtype])
#             print(globalrwd, file=log_globalrwd_file)
#             print(circle_flag, file=log_circle_file)
            
#             # Insert into rollout storage
#             for k in range(num_agent):
#                 masks = torch.tensor([1.])
#                 rollouts[k].insert(
#                     obses[k], 
#                     recurrent_hidden_states[k].squeeze(0), 
#                     condition_states[k], 
#                     actions[k].squeeze(0), 
#                     action_log_probs[k].squeeze(0), 
#                     values[k].squeeze(0), 
#                     rewards[k], 
#                     masks
#                 )

#         # MAPPO update after collecting rollouts
#         agent_masks = torch.ones(num_agent)
        
#         # Compute returns for each agent
#         for k in range(num_agent):
#             with torch.no_grad():
#                 condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
#                 next_value = actor_critics[k].get_value(
#                     rollouts[k].obs[-1].unsqueeze(0),
#                     rollouts[k].recurrent_hidden_states[-1].unsqueeze(0),
#                     condition_state.unsqueeze(0)
#                 ).detach()
#                 rollouts[k].compute_returns(
#                     next_value, args.use_gae, args.gamma, args.gae_lambda)
        
#         # MAPPO update
#         value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
#             rollouts, agent_masks)
        
#         epoch_time = time.time() - epoch_start_time
        
#         if epoch % 5 == 0:
#             print(f"Epoch {epoch:3d}/{args.num_pretrain_epochs} | "
#                   f"V-Loss: {value_loss:.4f} | "
#                   f"A-Loss: {action_loss:.4f} | "
#                   f"Entropy: {dist_entropy:.4f} | "
#                   f"Ratio: {ratio:.4f} | "
#                   f"Time: {epoch_time:.1f}s")
        
#         # After update, reset rollouts
#         for k in range(num_agent):
#             rollouts[k].after_update()
    
#     pretrain_time = time.time() - pretrain_start_time
#     print(f"\nPre-training completed in {pretrain_time:.1f}s ({pretrain_time/60:.1f} min)\n")

#     # ==================== TRAINING PHASE ====================
#     print("="*70)
#     print("TRAINING PHASE: Online learning with environment feedback")
#     print("="*70)
    
#     # Reset environment
#     request, obses = envs.reset()
    
#     # Update rollouts for training (longer episodes)
#     rollouts = []
#     for i in range(num_agent):
#         rollout = RolloutStorage(
#             args.num_steps,
#             observation_spaces[i].shape, 
#             action_spaces[i],
#             actor_critics[i].recurrent_hidden_state_size, 
#             num_node
#         )
#         rollouts.append(rollout)
#         rollouts[i].obs[0].copy_(obses[i])
#         rollouts[i].to(device)
    
#     # Training statistics
#     episode_rewards = deque(maxlen=100)
#     start_time = time.time()
    
#     # Training loop
#     for step in range(args.num_env_steps):
#         # Learning rate decay (Paper recommendation)
#         if args.use_linear_lr_decay:
#             mappo_agent.lr_decay(step, args.num_env_steps)
        
#         # Periodic progress report
#         if step % 1000 == 0 and step > 0:
#             total_time = time.time() - start_time
#             fps = step / total_time
#             print(f"\nStep {step:6d}/{args.num_env_steps} | "
#                   f"FPS: {fps:5.1f} | "
#                   f"Time: {total_time/60:5.1f}min")
        
#         with torch.no_grad():
#             values = [None] * num_agent
#             actions = [None] * num_agent
#             action_log_probs = [None] * num_agent
#             recurrent_hidden_states = [None] * num_agent
#             condition_states = [None] * num_agent

#             # Generate routing action
#             curr_path = [0] * num_node
#             agents_flag = [0] * num_agent
#             curr_agent, path = envs.first_agent()
            
#             while curr_agent is not None and agents_flag[curr_agent] != 1:
#                 for k in path:
#                     curr_path[k] = 1
#                 agents_flag[curr_agent] = 1
                
#                 condition_state = torch.tensor(curr_path, dtype=torch.float32).to(device)
#                 value, action, action_log_prob, recurrent_hidden_state = \
#                     actor_critics[curr_agent].act(
#                         rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
#                         rollouts[curr_agent].recurrent_hidden_states[
#                             rollouts[curr_agent].step].unsqueeze(0),
#                         condition_state.unsqueeze(0)
#                     )
                
#                 values[curr_agent] = value
#                 actions[curr_agent] = action
#                 action_log_probs[curr_agent] = action_log_prob
#                 recurrent_hidden_states[curr_agent] = recurrent_hidden_state
#                 condition_states[curr_agent] = condition_state
#                 curr_agent, path = envs.next_agent(curr_agent, action)
            
#             # Handle agents not on path
#             condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
#             for k in range(num_agent):
#                 if agents_flag[k] != 1:
#                     value, action, action_log_prob, recurrent_hidden_state = \
#                         actor_critics[k].act(
#                             rollouts[k].obs[rollouts[k].step].unsqueeze(0),
#                             rollouts[k].recurrent_hidden_states[
#                                 rollouts[k].step].unsqueeze(0),
#                             condition_state.unsqueeze(0)
#                         )
            
#                     values[k] = value
#                     actions[k] = action
#                     action_log_probs[k] = action_log_prob
#                     recurrent_hidden_states[k] = recurrent_hidden_state
#                     condition_states[k] = condition_state
        
#         # Step environment with global optimization
#         gfactors = [1.] * num_agent
#         obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, \
#             globalrwd, delay, throughput_rate, loss_rate = envs.step(actions, gfactors)
        
#         # Log metrics
#         print(delta_dist, file=log_dist_files[rtype])
#         print(delta_demand, file=log_demand_files[rtype])
#         print(delay, file=log_delay_files[rtype])
#         print(throughput_rate, file=log_throughput_files[rtype])
#         print(loss_rate, file=log_loss_files[rtype])
#         print(globalrwd, file=log_globalrwd_file)
#         print(circle_flag, file=log_circle_file)
        
#         episode_rewards.append(globalrwd)
        
#         # Insert into rollouts
#         agent_masks_tensor = torch.tensor(agents_flag, dtype=torch.float32)
        
#         for k in range(num_agent):
#             masks = torch.tensor([1.0 if agents_flag[k] == 1 else 0.0])
            
#             rollouts[k].insert(
#                 obses[k], 
#                 recurrent_hidden_states[k].squeeze(0), 
#                 condition_states[k], 
#                 actions[k].squeeze(0), 
#                 action_log_probs[k].squeeze(0), 
#                 values[k].squeeze(0), 
#                 rewards[k], 
#                 masks
#             )

#             # Update when rollout is full
#             if rollouts[k].step == 0:
#                 with torch.no_grad():
#                     condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
#                     next_value = actor_critics[k].get_value(
#                         rollouts[k].obs[-1].unsqueeze(0),
#                         rollouts[k].recurrent_hidden_states[-1].unsqueeze(0),
#                         condition_state.unsqueeze(0)
#                     ).detach()
#                     rollouts[k].compute_returns(
#                         next_value, args.use_gae, args.gamma, args.gae_lambda)
        
#         # Update all agents when rollout is full
#         if rollouts[0].step == 0:
#             value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
#                 rollouts, agent_masks_tensor)
            
#             # Log losses
#             print(value_loss, file=log_value_loss_file)
#             print(action_loss, file=log_action_loss_file)
#             print(ratio, file=log_ratio_file)
            
#             if step % 100 == 0:
#                 avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
#                 print(f"  V-Loss: {value_loss:.4f} | "
#                       f"A-Loss: {action_loss:.4f} | "
#                       f"Entropy: {dist_entropy:.4f} | "
#                       f"Ratio: {ratio:.4f} | "
#                       f"Avg-Rwd: {avg_reward:.4f}")
            
#             for k in range(num_agent):
#                 rollouts[k].after_update()
        
#         # Save checkpoint
#         if step % ckpt_step == 0 and step > 0:
#             if model_save_path is not None:
#                 save_dir = os.path.expanduser(model_save_path)
#                 os.makedirs(save_dir, exist_ok=True)
#                 for i in range(num_agent):
#                     torch.save(
#                         actor_critics[i].state_dict(), 
#                         os.path.join(model_save_path, f'agent{i}.pth')
#                     )
#                 print(f"\n[Checkpoint] Model saved at step {step}")

#     # Final save
#     if model_save_path is not None:
#         save_dir = os.path.expanduser(model_save_path)
#         os.makedirs(save_dir, exist_ok=True)
#         for i in range(num_agent):
#             torch.save(
#                 actor_critics[i].state_dict(), 
#                 os.path.join(model_save_path, f'agent{i}.pth')
#             )
#         print(f"\n[Final] Training completed. Model saved to {model_save_path}")
    
#     # Close log files
#     for f in (log_dist_files + log_demand_files + log_delay_files + 
#               log_throughput_files + log_loss_files):
#         f.close()
#     log_globalrwd_file.close()
#     log_circle_file.close()
#     log_value_loss_file.close()
#     log_action_loss_file.close()
#     log_ratio_file.close()
    
#     total_training_time = time.time() - start_time
#     print(f"\n{'='*70}")
#     print(f"Total training time: {total_training_time/60:.1f} minutes "
#           f"({total_training_time/3600:.2f} hours)")
#     print(f"Logs saved to: {log_dir}")
#     print(f"{'='*70}\n")


# if __name__ == "__main__":
#     main()









"""
Main training script for MAPPO with CTDE (Centralized Training, Decentralized Execution)

CTDE Architecture:
- ACTORS (Decentralized): Use LOCAL partial state + conditional state
  -> Deployed on each node for execution
- CRITIC (Centralized): Uses GLOBAL network state
  -> Used only during training for better value estimation

Based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
Paper: https://arxiv.org/abs/2103.01955

Usage:
    python3 main_mappo.py --env-name Abi --demand-matrix Abi_500.txt \
        --log-dir ./log/mappo --model-save-path ./model/mappo
"""

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.algo.mappo import (
    MAPPO_CTDE, 
    CentralizedCritic, 
    construct_global_state,
    get_global_state_dim
)
from a2c_ppo_acktr.arguments import get_mappo_args
from net_env.simenv import NetEnv





def print_ctde_config(args, num_agent, num_node, num_type, global_state_dim):
    """Print CTDE MAPPO configuration."""
    print("\n" + "="*70)
    print("MAPPO-CTDE Configuration (Centralized Training, Decentralized Execution)")
    print("="*70)
    print("\n[CTDE Architecture]")
    print(f"  Actors (Decentralized): {num_agent} agents using LOCAL observations")
    print(f"  Critic (Centralized): 1 shared critic using GLOBAL state")
    print(f"  Global state dimension: {global_state_dim}")
    print(f"    - Link residuals: {num_node * num_node}")
    print(f"    - Link losses: {num_node * num_node}")
    print(f"    - Flow features: {num_type + num_node * 2 + 1}")
    print("\n[Hyperparameters]")
    print(f"  Actor LR: {args.actor_lr}")
    print(f"  Critic LR: {args.critic_lr}")
    print(f"  Clip parameter: {args.clip_param}")
    print(f"  PPO epochs: {args.ppo_epoch}")
    print(f"  Mini batches: {args.num_mini_batch}")
    print(f"  Value loss coef: {args.value_loss_coef}")
    print(f"  Entropy coef: {args.entropy_coef}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"  Huber delta: {args.huber_delta}")
    print("\n[Features]")
    print(f"  Use PopArt: {args.use_popart}")
    print(f"  Use Huber loss: {args.use_huber_loss}")
    print(f"  Use clipped value loss: {args.use_clipped_value_loss}")
    print(f"  Use GAE: {args.use_gae}")
    print(f"  Use linear LR decay: {args.use_linear_lr_decay}")
    print("\n[Centralized Critic]")
    print(f"  Hidden size: {args.critic_hidden_size}")
    print(f"  Num layers: {args.critic_num_layers}")
    print(f"  Feature normalization: {args.use_feature_normalization}")
    print("="*70 + "\n")


def main():
    # Get arguments
    args = get_mappo_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setup directories
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "/eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    ckpt_step = args.ckpt_steps
    
    # Setup device
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    print("\n" + "="*70)
    print("MAPPO-CTDE Training for SDN Routing")
    print("="*70)
    print(f"Device: {device}")
    print(f"Algorithm: MAPPO with CTDE paradigm")
    print("="*70 + "\n")

    # Setup environment
    print("Setting up environment...")
    envs = NetEnv(args) 
    num_agent, num_node, observation_spaces, action_spaces, num_type = \
        envs.setup(args.env_name, args.demand_matrix)
    request, obses = envs.reset()
    
    print(f"Environment: {args.env_name}")
    print(f"Number of agents: {num_agent}")
    print(f"Number of nodes: {num_node}")
    print(f"Number of flow types: {num_type}")
    print(f"Observation space: {observation_spaces[0].shape}")
    print(f"Action space: {action_spaces[0]}")
    
    # Calculate global state dimension for centralized critic
    global_state_dim = get_global_state_dim(num_node, num_type)
    print(f"Global state dimension: {global_state_dim}\n")
    
    # Print full configuration
    print_ctde_config(args, num_agent, num_node, num_type, global_state_dim)

    # Open log files
    log_dist_files = []
    log_demand_files = []
    log_delay_files = []
    log_throughput_files = []
    log_loss_files = []
    
    for i in range(num_type):
        log_dist_file = open(f"{log_dir}/dist_type{i}.log", "w", 1)
        log_dist_files.append(log_dist_file)
        log_demand_file = open(f"{log_dir}/demand_type{i}.log", "w", 1)
        log_demand_files.append(log_demand_file)
        log_delay_file = open(f"{log_dir}/delay_type{i}.log", "w", 1)
        log_delay_files.append(log_delay_file)
        log_throughput_file = open(f"{log_dir}/throughput_type{i}.log", "w", 1)
        log_throughput_files.append(log_throughput_file)
        log_loss_file = open(f"{log_dir}/loss_type{i}.log", "w", 1)
        log_loss_files.append(log_loss_file)
    
    log_globalrwd_file = open(f"{log_dir}/globalrwd.log", "w", 1)
    log_circle_file = open(f"{log_dir}/circle.log", "w", 1)
    log_value_loss_file = open(f"{log_dir}/value_loss.log", "w", 1)
    log_action_loss_file = open(f"{log_dir}/action_loss.log", "w", 1)
    log_ratio_file = open(f"{log_dir}/ratio.log", "w", 1)

    # ================================================================
    # Build ACTORS (Decentralized - one per agent)
    # ================================================================
    print("Building decentralized actors...")
    actor_critics = []
    rollouts = []
    
    for i in range(num_agent):
        # Each actor uses LOCAL observations only
        actor_critic = Policy(
            observation_spaces[i].shape, 
            action_spaces[i], 
            num_node, 
            num_node, 
            num_type,
            base_kwargs={'recurrent': args.recurrent_policy}
        )
        
        if model_load_path is not None:
            model_file = os.path.join(model_load_path, f'actor{i}.pth')
            if os.path.exists(model_file):
                actor_critic.load_state_dict(torch.load(model_file, map_location=device))
                print(f"  Loaded actor {i} from {model_file}")
        
        actor_critic.to(device)
        actor_critics.append(actor_critic)

        rollout = RolloutStorage(
            args.num_pretrain_steps,
            observation_spaces[i].shape, 
            action_spaces[i],
            actor_critic.recurrent_hidden_state_size, 
            num_node
        )
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)
    
    print(f"  Created {num_agent} actors with local observation spaces")

    # ================================================================
    # Build CENTRALIZED CRITIC (uses GLOBAL state)
    # ================================================================
    print("\nBuilding centralized critic...")
    centralized_critic = CentralizedCritic(
        global_state_dim=global_state_dim,
        hidden_size=args.critic_hidden_size,
        num_layers=args.critic_num_layers,
        use_feature_normalization=args.use_feature_normalization,
        use_orthogonal=True,
        use_popart=args.use_popart,
        device=device
    )
    
    if model_load_path is not None:
        critic_file = os.path.join(model_load_path, 'critic.pth')
        if os.path.exists(critic_file):
            centralized_critic.load_state_dict(torch.load(critic_file, map_location=device))
            print(f"  Loaded centralized critic from {critic_file}")
    
    print(f"  Critic input: global state ({global_state_dim} dims)")
    print(f"  Critic architecture: {args.critic_num_layers} layers x {args.critic_hidden_size} hidden")

    # ================================================================
    # Create MAPPO-CTDE agent
    # ================================================================
    print("\nCreating MAPPO-CTDE agent...")
    mappo_agent = MAPPO_CTDE(
        actor_critics=actor_critics,
        centralized_critic=centralized_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_huber_loss=args.use_huber_loss,
        huber_delta=args.huber_delta,
        use_clipped_value_loss=args.use_clipped_value_loss,
        use_popart=args.use_popart,
        use_valuenorm=args.use_valuenorm,
        use_linear_lr_decay=args.use_linear_lr_decay,
        device=device
    )

    # ==================== PRE-TRAINING PHASE ====================
    print("\n" + "="*70)
    print("PRE-TRAINING PHASE: Learning shortest path policy")
    print("="*70)
    
    mappo_agent.prep_training()
    pretrain_start_time = time.time()
    
    # Storage for global states during rollout
    global_states_buffer = []
    
    for epoch in range(args.num_pretrain_epochs):
        epoch_start_time = time.time()
        global_states_buffer = []  # Reset for each epoch
        
        for step in range(args.num_pretrain_steps):
            with torch.no_grad():
                values = [None] * num_agent
                actions = [None] * num_agent
                action_log_probs = [None] * num_agent
                recurrent_hidden_states = [None] * num_agent
                condition_states = [None] * num_agent
                
                # Generate routing action hop-by-hop
                curr_path = [0] * num_node
                agents_flag = [0] * num_agent
                curr_agent, path = envs.first_agent()
                
                # ============================================================
                # Construct GLOBAL STATE for centralized critic
                # ============================================================
                global_state = construct_global_state(
                    link_capa=envs._link_capa,
                    link_usage=envs._link_usage,
                    link_losses=envs._link_losses,
                    flow_src=envs._request.s,
                    flow_dst=envs._request.t,
                    flow_type=envs._request.rtype,
                    flow_demand=envs._request.demand,
                    num_node=num_node,
                    num_type=num_type,
                    device=device
                )
                global_states_buffer.append(global_state)
                
                # Get centralized value estimate (using GLOBAL state)
                central_value = mappo_agent.get_values(global_state.unsqueeze(0))
                
                while curr_agent is not None and agents_flag[curr_agent] != 1:
                    for k in path:
                        curr_path[k] = 1
                    agents_flag[curr_agent] = 1
                    
                    # Actor uses LOCAL observation only
                    condition_state = torch.tensor(curr_path, dtype=torch.float32).to(device)
                    
                    value, action, action_log_prob, recurrent_hidden_state = \
                        actor_critics[curr_agent].act(
                            rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                            rollouts[curr_agent].recurrent_hidden_states[
                                rollouts[curr_agent].step].unsqueeze(0),
                            condition_state.unsqueeze(0)
                        )

                    # Use centralized value for all agents
                    values[curr_agent] = central_value
                    actions[curr_agent] = action
                    action_log_probs[curr_agent] = action_log_prob
                    recurrent_hidden_states[curr_agent] = recurrent_hidden_state
                    condition_states[curr_agent] = condition_state
                    curr_agent, path = envs.next_agent(curr_agent, action)
                
                # Handle agents not on path
                condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
                for k in range(num_agent):
                    if agents_flag[k] != 1:
                        value, action, action_log_prob, recurrent_hidden_state = \
                            actor_critics[k].act(
                                rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                                rollouts[k].recurrent_hidden_states[
                                    rollouts[k].step].unsqueeze(0),
                                condition_state.unsqueeze(0)
                            )
                
                        values[k] = central_value  # Centralized value
                        actions[k] = action
                        action_log_probs[k] = action_log_prob
                        recurrent_hidden_states[k] = recurrent_hidden_state
                        condition_states[k] = condition_state

            # Step environment
            gfactors = [0.] * num_agent
            obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, \
                globalrwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
            
            # Log metrics
            print(delta_dist, file=log_dist_files[rtype])
            print(delta_demand, file=log_demand_files[rtype])
            print(globalrwd, file=log_globalrwd_file)
            print(circle_flag, file=log_circle_file)
            
            # Insert into rollout storage
            for k in range(num_agent):
                masks = torch.tensor([1.])
                rollouts[k].insert(
                    obses[k], 
                    recurrent_hidden_states[k].squeeze(0), 
                    condition_states[k], 
                    actions[k].squeeze(0), 
                    action_log_probs[k].squeeze(0), 
                    values[k].squeeze(0),  # Centralized value
                    rewards[k], 
                    masks
                )

        # Compute returns using centralized critic
        for k in range(num_agent):
            with torch.no_grad():
                # Get final global state
                final_global_state = construct_global_state(
                    link_capa=envs._link_capa,
                    link_usage=envs._link_usage,
                    link_losses=envs._link_losses,
                    flow_src=envs._request.s,
                    flow_dst=envs._request.t,
                    flow_type=envs._request.rtype,
                    flow_demand=envs._request.demand,
                    num_node=num_node,
                    num_type=num_type,
                    device=device
                )
                # Use CENTRALIZED critic for next value
                next_value = mappo_agent.get_values(final_global_state.unsqueeze(0)).detach()
                rollouts[k].compute_returns(
                    next_value, args.use_gae, args.gamma, args.gae_lambda)
        
        # MAPPO-CTDE update
        global_states_tensor = torch.stack(global_states_buffer)
        agent_masks = torch.ones(num_agent)
        
        value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
            rollouts, global_states_tensor, agent_masks)
        
        # Reset rollouts
        for k in range(num_agent):
            rollouts[k].after_update()
        
        epoch_time = time.time() - epoch_start_time
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{args.num_pretrain_epochs} | "
                  f"V-Loss: {value_loss:.4f} | "
                  f"A-Loss: {action_loss:.4f} | "
                  f"Entropy: {dist_entropy:.4f} | "
                  f"Ratio: {ratio:.4f} | "
                  f"Time: {epoch_time:.1f}s")
    
    pretrain_time = time.time() - pretrain_start_time
    print(f"\nPre-training completed in {pretrain_time:.1f}s ({pretrain_time/60:.1f} min)\n")

    # ==================== TRAINING PHASE ====================
    print("="*70)
    print("TRAINING PHASE: Online learning with CTDE")
    print("="*70)
    
    # Reset environment
    request, obses = envs.reset()
    
    # Update rollouts for training
    rollouts = []
    for i in range(num_agent):
        rollout = RolloutStorage(
            args.num_steps,
            observation_spaces[i].shape, 
            action_spaces[i],
            actor_critics[i].recurrent_hidden_state_size, 
            num_node
        )
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)
    
    mappo_agent.reset_optimizers()
    
    # Training statistics
    episode_rewards = deque(maxlen=100)
    start_time = time.time()
    global_states_buffer = []
    
    # Training loop
    for step in range(args.num_env_steps):
        if args.use_linear_lr_decay:
            mappo_agent.lr_decay(step, args.num_env_steps)
        
        if step % 1000 == 0 and step > 0:
            total_time = time.time() - start_time
            fps = step / total_time
            print(f"\nStep {step:6d}/{args.num_env_steps} | "
                  f"FPS: {fps:5.1f} | "
                  f"Time: {total_time/60:5.1f}min")
        
        with torch.no_grad():
            values = [None] * num_agent
            actions = [None] * num_agent
            action_log_probs = [None] * num_agent
            recurrent_hidden_states = [None] * num_agent
            condition_states = [None] * num_agent

            curr_path = [0] * num_node
            agents_flag = [0] * num_agent
            curr_agent, path = envs.first_agent()
            
            # Construct GLOBAL STATE for centralized critic
            global_state = construct_global_state(
                link_capa=envs._link_capa,
                link_usage=envs._link_usage,
                link_losses=envs._link_losses,
                flow_src=envs._request.s,
                flow_dst=envs._request.t,
                flow_type=envs._request.rtype,
                flow_demand=envs._request.demand,
                num_node=num_node,
                num_type=num_type,
                device=device
            )
            global_states_buffer.append(global_state)
            
            # Get centralized value
            central_value = mappo_agent.get_values(global_state.unsqueeze(0))
            
            while curr_agent is not None and agents_flag[curr_agent] != 1:
                for k in path:
                    curr_path[k] = 1
                agents_flag[curr_agent] = 1
                
                condition_state = torch.tensor(curr_path, dtype=torch.float32).to(device)
                value, action, action_log_prob, recurrent_hidden_state = \
                    actor_critics[curr_agent].act(
                        rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                        rollouts[curr_agent].recurrent_hidden_states[
                            rollouts[curr_agent].step].unsqueeze(0),
                        condition_state.unsqueeze(0)
                    )
                
                values[curr_agent] = central_value
                actions[curr_agent] = action
                action_log_probs[curr_agent] = action_log_prob
                recurrent_hidden_states[curr_agent] = recurrent_hidden_state
                condition_states[curr_agent] = condition_state
                curr_agent, path = envs.next_agent(curr_agent, action)
            
            condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
            for k in range(num_agent):
                if agents_flag[k] != 1:
                    value, action, action_log_prob, recurrent_hidden_state = \
                        actor_critics[k].act(
                            rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                            rollouts[k].recurrent_hidden_states[
                                rollouts[k].step].unsqueeze(0),
                            condition_state.unsqueeze(0)
                        )
            
                    values[k] = central_value
                    actions[k] = action
                    action_log_probs[k] = action_log_prob
                    recurrent_hidden_states[k] = recurrent_hidden_state
                    condition_states[k] = condition_state
        
        # Step environment
        gfactors = [1.] * num_agent
        obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, \
            globalrwd, delay, throughput_rate, loss_rate = envs.step(actions, gfactors)
        
        # Log metrics
        print(delta_dist, file=log_dist_files[rtype])
        print(delta_demand, file=log_demand_files[rtype])
        print(delay, file=log_delay_files[rtype])
        print(throughput_rate, file=log_throughput_files[rtype])
        print(loss_rate, file=log_loss_files[rtype])
        print(globalrwd, file=log_globalrwd_file)
        print(circle_flag, file=log_circle_file)
        
        episode_rewards.append(globalrwd)
        
        agent_masks_tensor = torch.tensor(agents_flag, dtype=torch.float32)
        
        for k in range(num_agent):
            if agents_flag[k] == 1:
                masks = torch.tensor([1.])
            else:
                masks = torch.tensor([0.])
            
            rollouts[k].insert(
                obses[k], 
                recurrent_hidden_states[k].squeeze(0), 
                condition_states[k], 
                actions[k].squeeze(0), 
                action_log_probs[k].squeeze(0), 
                values[k].squeeze(0),
                rewards[k], 
                masks
            )

            if rollouts[k].step == 0:
                with torch.no_grad():
                    final_global_state = construct_global_state(
                        link_capa=envs._link_capa,
                        link_usage=envs._link_usage,
                        link_losses=envs._link_losses,
                        flow_src=envs._request.s,
                        flow_dst=envs._request.t,
                        flow_type=envs._request.rtype,
                        flow_demand=envs._request.demand,
                        num_node=num_node,
                        num_type=num_type,
                        device=device
                    )
                    next_value = mappo_agent.get_values(final_global_state.unsqueeze(0)).detach()
                    rollouts[k].compute_returns(
                        next_value, args.use_gae, args.gamma, args.gae_lambda)
        
        # Update when rollout is full
        if rollouts[0].step == 0:
            global_states_tensor = torch.stack(global_states_buffer)
            
            value_loss, action_loss, dist_entropy, ratio = mappo_agent.update(
                rollouts, global_states_tensor, agent_masks_tensor)
            
            global_states_buffer = []  # Reset buffer
            
            print(value_loss, file=log_value_loss_file)
            print(action_loss, file=log_action_loss_file)
            print(ratio, file=log_ratio_file)
            
            if step % 100 == 0:
                avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
                print(f"  V-Loss: {value_loss:.4f} | "
                      f"A-Loss: {action_loss:.4f} | "
                      f"Entropy: {dist_entropy:.4f} | "
                      f"Ratio: {ratio:.4f} | "
                      f"Avg-Rwd: {avg_reward:.4f}")
            
            for k in range(num_agent):
                rollouts[k].after_update()
        
        # Save checkpoint
        if step % ckpt_step == 0 and step > 0:
            if model_save_path is not None:
                save_dir = os.path.expanduser(model_save_path)
                os.makedirs(save_dir, exist_ok=True)
                # Save actors
                for i in range(num_agent):
                    torch.save(
                        actor_critics[i].state_dict(), 
                        os.path.join(model_save_path, f'actor{i}.pth')
                    )
                # Save centralized critic
                torch.save(
                    centralized_critic.state_dict(),
                    os.path.join(model_save_path, 'critic.pth')
                )
                print(f"\n[Checkpoint] Model saved at step {step}")

    # Final save
    if model_save_path is not None:
        save_dir = os.path.expanduser(model_save_path)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(num_agent):
            torch.save(
                actor_critics[i].state_dict(), 
                os.path.join(model_save_path, f'actor{i}.pth')
            )
        torch.save(
            centralized_critic.state_dict(),
            os.path.join(model_save_path, 'critic.pth')
        )
        print(f"\n[Final] Training completed. Model saved to {model_save_path}")
    
    # Close log files
    for f in (log_dist_files + log_demand_files + log_delay_files + 
              log_throughput_files + log_loss_files):
        f.close()
    log_globalrwd_file.close()
    log_circle_file.close()
    log_value_loss_file.close()
    log_action_loss_file.close()
    log_ratio_file.close()
    
    total_training_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total training time: {total_training_time/60:.1f} minutes "
          f"({total_training_time/3600:.2f} hours)")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()