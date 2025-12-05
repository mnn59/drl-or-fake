import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # model config
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo')
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    
    # running config
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-steps', type=int, default=512,
                        help='number of forward steps (default: 512) i.e.num-step for each update epoch')
    parser.add_argument('--num-pretrain-epochs', type=int, default=30,
                        help='number of pretraining steps  (default: 500)')
    parser.add_argument('--num-pretrain-steps', type=int, default=128,
                        help='number of forward steps for pretraining (default: 128)')
    parser.add_argument('--ckpt-steps', type=int, default=10000,
                        help='number of iteration steps for each checkpoint when training')
    parser.add_argument('--num-env-steps', type=int, default=10000000,
                        help='number of environment steps to train (default: 1000000)')
    parser.add_argument('--env-name', default='Abi',
                        help='environment to train on (default: Abi)') #temporarily deprecated
    parser.add_argument('--log-dir', default='/tmp/DRL-OR',
                        help='directory to save agent logs (default: /tmp/DRL-OR)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--demand-matrix", default='test.txt', 
                        help='demand matrix input file name (default:test.txt)')
    parser.add_argument("--model-load-path", default=None,
                        help='load model parameters from the model-load-path')
    parser.add_argument("--model-save-path", default=None,
                        help='save model parameters at the model-save-path')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo']
    
    return args


"""
Argument parser for MAPPO with hyperparameters tuned based on the paper

Add this to your existing arguments.py or create a new mappo_arguments.py
"""
# def get_mappo_args():
#     """
#     Get arguments with MAPPO-specific hyperparameters
#     Based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
#     """
#     parser = argparse.ArgumentParser(
#         description='MAPPO for SDN Routing',
#         formatter_class=argparse.RawDescriptionHelpFormatter)

#     # ============================================================
#     # Environment Configuration
#     # ============================================================
#     parser.add_argument(
#         '--env-name',
#         type=str,
#         default='Abi',
#         help='Network topology (Abi or GEA)')
    
#     parser.add_argument(
#         '--demand-matrix',
#         type=str,
#         default='Abi_500.txt',
#         help='Traffic demand matrix file')

#     # ============================================================
#     # MAPPO Algorithm Hyperparameters (Tuned based on paper)
#     # ============================================================
    
#     # Learning rate (Paper: typically 5e-4 to 1e-4)
#     parser.add_argument(
#         '--lr',
#         type=float,
#         default=5e-4,
#         help='Learning rate (paper recommendation: 5e-4)')
    
#     # PPO clip parameter (Paper Sec 5.4: keep under 0.2)
#     parser.add_argument(
#         '--clip-param',
#         type=float,
#         default=0.2,
#         help='PPO clipping parameter epsilon (paper: 0.1-0.2)')
    
#     # Number of PPO epochs (Paper Sec 5.3: 5-15 epochs)
#     parser.add_argument(
#         '--ppo-epoch',
#         type=int,
#         default=15,
#         help='Number of PPO epochs per update (paper: 5-15)')
    
#     # Number of mini-batches (Paper: typically 1 for small-scale)
#     parser.add_argument(
#         '--num-mini-batch',
#         type=int,
#         default=1,
#         help='Number of mini-batches (paper: 1 for small-scale)')
    
#     # Batch size / rollout length (Paper Sec 5.5: large batch)
#     parser.add_argument(
#         '--num-steps',
#         type=int,
#         default=512,
#         help='Number of steps per rollout (paper: 128-2048)')
    
#     # Value loss coefficient
#     parser.add_argument(
#         '--value-loss-coef',
#         type=float,
#         default=1.0,
#         help='Value loss coefficient (paper: typically 1.0)')
    
#     # Entropy coefficient (for exploration)
#     parser.add_argument(
#         '--entropy-coef',
#         type=float,
#         default=0.01,
#         help='Entropy coefficient (paper: 0.01)')
    
#     # Max gradient norm for clipping
#     parser.add_argument(
#         '--max-grad-norm',
#         type=float,
#         default=10.0,
#         help='Max gradient norm (paper: 10.0)')
    
#     # GAE parameters
#     parser.add_argument(
#         '--gamma',
#         type=float,
#         default=0.99,
#         help='Discount factor')
    
#     parser.add_argument(
#         '--gae-lambda',
#         type=float,
#         default=0.95,
#         help='GAE lambda parameter')
    
#     parser.add_argument(
#         '--use-gae',
#         action='store_true',
#         default=False,
#         help='Use Generalized Advantage Estimation')
    
#     # Learning rate decay (Paper recommendation)
#     parser.add_argument(
#         '--use-linear-lr-decay',
#         action='store_true',
#         default=False,
#         help='Use linear learning rate decay')

#     # ============================================================
#     # Training Configuration
#     # ============================================================
#     parser.add_argument(
#         '--num-env-steps',
#         type=int,
#         default=100000,
#         help='Number of environment steps for training')
    
#     parser.add_argument(
#         '--num-pretrain-epochs',
#         type=int,
#         default=30,
#         help='Number of pre-training epochs')
    
#     parser.add_argument(
#         '--num-pretrain-steps',
#         type=int,
#         default=128,
#         help='Steps per pre-training epoch')

#     # ============================================================
#     # Model Configuration
#     # ============================================================
#     parser.add_argument(
#         '--recurrent-policy',
#         action='store_true',
#         default=False,
#         help='Use recurrent policy (LSTM/GRU)')
    
#     parser.add_argument(
#         '--use-parameter-sharing',
#         action='store_true',
#         default=True,
#         help='Share parameters across agents (paper: recommended for homogeneous)')

#     # ============================================================
#     # System Configuration
#     # ============================================================
#     parser.add_argument(
#         '--cuda',
#         action='store_true',
#         default=False,
#         help='Use CUDA')
    
#     parser.add_argument(
#         '--cuda-deterministic',
#         action='store_true',
#         default=False,
#         help='Make CUDA deterministic')
    
#     parser.add_argument(
#         '--seed',
#         type=int,
#         default=1,
#         help='Random seed')
    
#     parser.add_argument(
#         '--eps',
#         type=float,
#         default=1e-5,
#         help='Adam epsilon')

#     # ============================================================
#     # Logging and Checkpointing
#     # ============================================================
#     parser.add_argument(
#         '--log-dir',
#         type=str,
#         default='./log/mappo',
#         help='Directory to save logs')
    
#     parser.add_argument(
#         '--model-save-path',
#         type=str,
#         default='./model/mappo',
#         help='Directory to save models')
    
#     parser.add_argument(
#         '--model-load-path',
#         type=str,
#         default=None,
#         help='Path to load pre-trained models')
    
#     parser.add_argument(
#         '--ckpt-steps',
#         type=int,
#         default=10000,
#         help='Steps between checkpoints')

#     args = parser.parse_args()
    
#     # Auto-detect CUDA
#     args.cuda = args.cuda and torch.cuda.is_available()
    
#     return args


# Hyperparameter presets for different scenarios
MAPPO_PRESETS = {
    'small_scale': {
        'lr': 5e-4,
        'clip_param': 0.2,
        'ppo_epoch': 15,
        'num_mini_batch': 1,
        'num_steps': 256,
        'batch_size': 256,
    },
    'medium_scale': {
        'lr': 5e-4,
        'clip_param': 0.2,
        'ppo_epoch': 10,
        'num_mini_batch': 4,
        'num_steps': 512,
        'batch_size': 2048,
    },
    'large_scale': {
        'lr': 1e-4,
        'clip_param': 0.1,
        'ppo_epoch': 5,
        'num_mini_batch': 8,
        'num_steps': 1024,
        'batch_size': 8192,
    }
}


def print_mappo_config(args):
    """Print MAPPO configuration for verification"""
    print("\n" + "="*70)
    print("MAPPO Configuration")
    print("="*70)
    print(f"Environment: {args.env_name} ({args.demand_matrix})")
    print(f"\nCore Hyperparameters:")
    print(f"  Learning Rate:        {args.lr}")
    print(f"  Clip Parameter:       {args.clip_param}")
    print(f"  PPO Epochs:           {args.ppo_epoch}")
    print(f"  Mini Batches:         {args.num_mini_batch}")
    print(f"  Rollout Length:       {args.num_steps}")
    print(f"  Value Loss Coef:      {args.value_loss_coef}")
    print(f"  Entropy Coef:         {args.entropy_coef}")
    print(f"  Max Grad Norm:        {args.max_grad_norm}")
    print(f"\nTraining:")
    print(f"  Total Steps:          {args.num_env_steps}")
    print(f"  Pre-train Epochs:     {args.num_pretrain_epochs}")
    print(f"  GAE:                  {args.use_gae}")
    print(f"  LR Decay:             {args.use_linear_lr_decay}")
    print(f"\nSystem:")
    print(f"  CUDA:                 {args.cuda}")
    print(f"  Seed:                 {args.seed}")
    print(f"  Log Dir:              {args.log_dir}")
    print(f"  Model Save:           {args.model_save_path}")
    print("="*70 + "\n")


def get_mappo_args():
    """
    Get MAPPO-specific arguments with hyperparameters from the paper.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MAPPO-CTDE for DRL-OR')
    
    # ============================================================
    # MAPPO hyperparameters (from paper)
    # ============================================================
    parser.add_argument('--actor-lr', type=float, default=5e-4,
                        help='actor learning rate (MAPPO: 5e-4)')
    parser.add_argument('--critic-lr', type=float, default=5e-4,
                        help='critic learning rate (MAPPO: 5e-4)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate (for compatibility)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='Adam optimizer epsilon')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use GAE (default: True)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=1.0,
                        help='value loss coefficient (MAPPO: 1.0)')
    parser.add_argument('--max-grad-norm', type=float, default=10.0,
                        help='max gradient norm (MAPPO: 10.0)')
    parser.add_argument('--ppo-epoch', type=int, default=15,
                        help='PPO epochs (MAPPO: 5-15)')
    parser.add_argument('--num-mini-batch', type=int, default=1,
                        help='mini-batches (MAPPO: 1)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='PPO clip parameter (MAPPO: 0.2)')
    parser.add_argument('--huber-delta', type=float, default=10.0,
                        help='Huber loss delta (MAPPO: 10.0)')
    
    # ============================================================
    # CTDE-specific options
    # ============================================================
    parser.add_argument('--critic-hidden-size', type=int, default=64,
                        help='centralized critic hidden size')
    parser.add_argument('--critic-num-layers', type=int, default=2,
                        help='centralized critic number of layers')
    parser.add_argument('--use-feature-normalization', action='store_true', default=True,
                        help='use layer normalization in critic')
    
    # ============================================================
    # MAPPO feature flags
    # ============================================================
    parser.add_argument('--use-popart', action='store_true', default=True,
                        help='use PopArt normalization (recommended)')
    parser.add_argument('--use-valuenorm', action='store_true', default=False,
                        help='use simple value normalization')
    parser.add_argument('--use-huber-loss', action='store_true', default=True,
                        help='use Huber loss for value function')
    parser.add_argument('--use-clipped-value-loss', action='store_true', default=True,
                        help='use clipped value loss')
    
    # ============================================================
    # Policy settings
    # ============================================================
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=True,
                        help='use linear LR decay')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use linear clip decay')
    
    # ============================================================
    # Training configuration
    # ============================================================
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help='CUDA deterministic mode')
    parser.add_argument('--num-steps', type=int, default=512,
                        help='rollout steps per update')
    parser.add_argument('--num-pretrain-epochs', type=int, default=30,
                        help='pretrain epochs')
    parser.add_argument('--num-pretrain-steps', type=int, default=128,
                        help='pretrain steps per epoch')
    parser.add_argument('--ckpt-steps', type=int, default=10000,
                        help='checkpoint interval')
    parser.add_argument('--num-env-steps', type=int, default=10000000,
                        help='total training steps')
    
    # ============================================================
    # Environment
    # ============================================================
    parser.add_argument('--env-name', default='Abi',
                        help='environment name')
    parser.add_argument('--demand-matrix', default='test.txt',
                        help='demand matrix file')
    
    # ============================================================
    # Logging and saving
    # ============================================================
    parser.add_argument('--log-dir', default='./log/mappo_ctde',
                        help='log directory')
    parser.add_argument('--model-load-path', default=None,
                        help='model load path')
    parser.add_argument('--model-save-path', default=None,
                        help='model save path')
    
    # ============================================================
    # Device
    # ============================================================
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args