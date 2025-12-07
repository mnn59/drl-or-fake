# """
# Compare PPO vs MAPPO Performance in Initialization Scenario
# Based on DRL-OR Paper Figure 5

# This script compares the original PPO implementation with MAPPO-CTDE
# for the initialization scenario, showing:
# - Latency comparison for each flow type
# - Throughput ratio comparison for each flow type
# - Global reward comparison
# - Unsafe route rate comparison

# Note: Original DRL-OR PPO does not log value_loss, action_loss, ratio.
#       These metrics are only available for MAPPO.

# Usage:
#     python3 compare_ppo_mappo.py \
#         --ppo-log ./log/initialization \
#         --mappo-log ./log/mappo_initialization \
#         --output comparison_ppo_mappo.png
# """

# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import os


# def read_log_file(filepath):
#     """Read values from a log file"""
#     values = []
#     try:
#         with open(filepath, 'r') as f:
#             for line in f:
#                 try:
#                     values.append(float(line.strip()))
#                 except ValueError:
#                     continue
#     except FileNotFoundError:
#         return []
#     return values


# def smooth_curve(values, window=1000):
#     """Apply moving average smoothing"""
#     if len(values) < window:
#         return values
    
#     smoothed = []
#     for i in range(len(values)):
#         start = max(0, i - window // 2)
#         end = min(len(values), i + window // 2)
#         smoothed.append(np.mean(values[start:end]))
#     return smoothed


# def plot_comparison(ppo_log_dir, mappo_log_dir, output_file='comparison_ppo_mappo.png', 
#                     window=1000, max_steps=None):
#     """
#     Create comparison plots for PPO vs MAPPO
    
#     Layout: 3 rows × 3 columns
#     Row 1: Latency for type0, type1, type2
#     Row 2: Throughput for type0, type1, type2
#     Row 3: Global Reward, Unsafe Route Rate, Packet Loss Rate
#     """
    
#     # Create figure
#     fig = plt.figure(figsize=(18, 14))
#     gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    
#     # Flow types configuration (based on DRL-OR paper)
#     flow_types = [
#         {'id': 0, 'name': 'Type 1\n(Latency-sensitive)'},
#         {'id': 1, 'name': 'Type 2\n(Throughput-sensitive)'},
#         {'id': 2, 'name': 'Type 3\n(Latency-Throughput)'},
#     ]
    
#     # Colors
#     PPO_COLOR = '#1f77b4'      # Blue
#     MAPPO_COLOR = '#d62728'    # Red
    
#     print("\n" + "="*80)
#     print("COMPARING PPO vs MAPPO-CTDE - Initialization Scenario")
#     print("="*80)
    
#     # ========================================================================
#     # Row 1: Latency Comparison
#     # ========================================================================
#     print("\n[LATENCY COMPARISON]")
    
#     for i, flow_type in enumerate(flow_types):
#         ax = fig.add_subplot(gs[0, i])
        
#         # Read PPO data
#         ppo_file = os.path.join(ppo_log_dir, f"delay_type{flow_type['id']}.log")
#         ppo_delays = read_log_file(ppo_file)
        
#         # Read MAPPO data
#         mappo_file = os.path.join(mappo_log_dir, f"delay_type{flow_type['id']}.log")
#         mappo_delays = read_log_file(mappo_file)
        
#         # Determine max steps
#         if max_steps is None:
#             plot_max = max(len(ppo_delays) if ppo_delays else 0, 
#                           len(mappo_delays) if mappo_delays else 0)
#         else:
#             plot_max = max_steps
        
#         # Plot PPO
#         if ppo_delays:
#             ppo_smooth = smooth_curve(ppo_delays, window=window)
#             timesteps = np.arange(len(ppo_smooth)) / 1000.0
#             max_idx = min(len(ppo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], ppo_smooth[:max_idx], 
#                    label='PPO (Original)', color=PPO_COLOR, 
#                    linewidth=2, alpha=0.9)
#             ppo_final = np.mean(ppo_delays[-10000:]) if len(ppo_delays) > 10000 else np.mean(ppo_delays)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: PPO avg={np.mean(ppo_delays):.2f}ms, final={ppo_final:.2f}ms")
        
#         # Plot MAPPO
#         if mappo_delays:
#             mappo_smooth = smooth_curve(mappo_delays, window=window)
#             timesteps = np.arange(len(mappo_smooth)) / 1000.0
#             max_idx = min(len(mappo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], mappo_smooth[:max_idx], 
#                    label='MAPPO-CTDE', color=MAPPO_COLOR, 
#                    linewidth=2, linestyle='--', alpha=0.9)
#             mappo_final = np.mean(mappo_delays[-10000:]) if len(mappo_delays) > 10000 else np.mean(mappo_delays)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: MAPPO avg={np.mean(mappo_delays):.2f}ms, final={mappo_final:.2f}ms")
            
#             # Calculate improvement
#             if ppo_delays:
#                 improvement = (ppo_final - mappo_final) / ppo_final * 100
#                 print(f"    → Latency Reduction: {improvement:+.2f}%")
        
#         ax.set_xlabel('Timeslot (×10³)', fontsize=11)
#         ax.set_ylabel('Latency (ms)', fontsize=11)
#         ax.set_title(f'Latency - {flow_type["name"]}', fontsize=12, fontweight='bold')
#         ax.legend(fontsize=10, loc='upper right')
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.set_xlim(left=0)
    
#     # ========================================================================
#     # Row 2: Throughput Comparison
#     # ========================================================================
#     print("\n[THROUGHPUT COMPARISON]")
    
#     for i, flow_type in enumerate(flow_types):
#         ax = fig.add_subplot(gs[1, i])
        
#         # Read PPO data
#         ppo_file = os.path.join(ppo_log_dir, f"throughput_type{flow_type['id']}.log")
#         ppo_throughput = read_log_file(ppo_file)
        
#         # Read MAPPO data
#         mappo_file = os.path.join(mappo_log_dir, f"throughput_type{flow_type['id']}.log")
#         mappo_throughput = read_log_file(mappo_file)
        
#         # Determine max steps
#         if max_steps is None:
#             plot_max = max(len(ppo_throughput) if ppo_throughput else 0, 
#                           len(mappo_throughput) if mappo_throughput else 0)
#         else:
#             plot_max = max_steps
        
#         # Plot PPO
#         if ppo_throughput:
#             ppo_smooth = smooth_curve(ppo_throughput, window=window)
#             timesteps = np.arange(len(ppo_smooth)) / 1000.0
#             max_idx = min(len(ppo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], ppo_smooth[:max_idx], 
#                    label='PPO (Original)', color=PPO_COLOR, 
#                    linewidth=2, alpha=0.9)
#             ppo_final = np.mean(ppo_throughput[-10000:]) if len(ppo_throughput) > 10000 else np.mean(ppo_throughput)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: PPO avg={np.mean(ppo_throughput):.4f}, final={ppo_final:.4f}")
        
#         # Plot MAPPO
#         if mappo_throughput:
#             mappo_smooth = smooth_curve(mappo_throughput, window=window)
#             timesteps = np.arange(len(mappo_smooth)) / 1000.0
#             max_idx = min(len(mappo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], mappo_smooth[:max_idx], 
#                    label='MAPPO-CTDE', color=MAPPO_COLOR, 
#                    linewidth=2, linestyle='--', alpha=0.9)
#             mappo_final = np.mean(mappo_throughput[-10000:]) if len(mappo_throughput) > 10000 else np.mean(mappo_throughput)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: MAPPO avg={np.mean(mappo_throughput):.4f}, final={mappo_final:.4f}")
            
#             # Calculate improvement
#             if ppo_throughput:
#                 improvement = (mappo_final - ppo_final) / ppo_final * 100
#                 print(f"    → Throughput Improvement: {improvement:+.2f}%")
        
#         ax.set_xlabel('Timeslot (×10³)', fontsize=11)
#         ax.set_ylabel('Throughput Ratio', fontsize=11)
#         ax.set_title(f'Throughput - {flow_type["name"]}', fontsize=12, fontweight='bold')
#         ax.legend(fontsize=10, loc='lower right')
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.set_xlim(left=0)
#         ax.set_ylim([0.85, 1.02])
    
#     # ========================================================================
#     # Row 3: Global Reward, Unsafe Rate, Packet Loss
#     # ========================================================================
    
#     # --- Global Reward Comparison ---
#     ax_reward = fig.add_subplot(gs[2, 0])
#     print("\n[GLOBAL REWARD COMPARISON]")
    
#     ppo_reward = read_log_file(os.path.join(ppo_log_dir, "globalrwd.log"))
#     mappo_reward = read_log_file(os.path.join(mappo_log_dir, "globalrwd.log"))
    
#     if ppo_reward:
#         ppo_smooth = smooth_curve(ppo_reward, window=window)
#         timesteps = np.arange(len(ppo_smooth)) / 1000.0
#         ax_reward.plot(timesteps, ppo_smooth, label='PPO (Original)', 
#                       color=PPO_COLOR, linewidth=2, alpha=0.9)
#         ppo_final = np.mean(ppo_reward[-10000:]) if len(ppo_reward) > 10000 else np.mean(ppo_reward)
#         print(f"  PPO: avg={np.mean(ppo_reward):.4f}, final={ppo_final:.4f}")
    
#     if mappo_reward:
#         mappo_smooth = smooth_curve(mappo_reward, window=window)
#         timesteps = np.arange(len(mappo_smooth)) / 1000.0
#         ax_reward.plot(timesteps, mappo_smooth, label='MAPPO-CTDE', 
#                       color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
#         mappo_final = np.mean(mappo_reward[-10000:]) if len(mappo_reward) > 10000 else np.mean(mappo_reward)
#         print(f"  MAPPO: avg={np.mean(mappo_reward):.4f}, final={mappo_final:.4f}")
        
#         if ppo_reward:
#             # Higher reward is better
#             improvement = (mappo_final - ppo_final) / abs(ppo_final) * 100 if ppo_final != 0 else 0
#             print(f"    → Reward Improvement: {improvement:+.2f}%")
    
#     ax_reward.set_xlabel('Timeslot (×10³)', fontsize=11)
#     ax_reward.set_ylabel('Global Reward', fontsize=11)
#     ax_reward.set_title('Global Reward', fontsize=12, fontweight='bold')
#     ax_reward.legend(fontsize=10)
#     ax_reward.grid(True, alpha=0.3, linestyle='--')
#     ax_reward.set_xlim(left=0)
    
#     # --- Unsafe Route Rate (Circle Flag) ---
#     ax_circle = fig.add_subplot(gs[2, 1])
#     print("\n[UNSAFE ROUTE RATE (Fallback Policy Usage)]")
    
#     ppo_circle = read_log_file(os.path.join(ppo_log_dir, "circle.log"))
#     print("circle log ppo have been read!")
#     mappo_circle = read_log_file(os.path.join(mappo_log_dir, "circle.log"))
#     print("circle log mappo have been read!")
    
    
#     if ppo_circle:
#         # Calculate cumulative unsafe rate
#         ppo_cumulative = [np.mean(ppo_circle[:i+1]) * 100 for i in range(len(ppo_circle))]
#         ppo_smooth = smooth_curve(ppo_cumulative, window=window)
#         timesteps = np.arange(len(ppo_smooth)) / 1000.0
#         ax_circle.plot(timesteps, ppo_smooth, label='PPO (Original)', 
#                       color=PPO_COLOR, linewidth=2, alpha=0.9)
#         print(f"  PPO: total unsafe rate = {np.mean(ppo_circle)*100:.2f}%")
    
#     if mappo_circle:
#         mappo_cumulative = [np.mean(mappo_circle[:i+1]) * 100 for i in range(len(mappo_circle))]
#         mappo_smooth = smooth_curve(mappo_cumulative, window=window)
#         timesteps = np.arange(len(mappo_smooth)) / 1000.0
#         ax_circle.plot(timesteps, mappo_smooth, label='MAPPO-CTDE', 
#                       color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
#         print(f"  MAPPO: total unsafe rate = {np.mean(mappo_circle)*100:.2f}%")
        
#         if ppo_circle:
#             # Lower unsafe rate is better
#             reduction = (np.mean(ppo_circle) - np.mean(mappo_circle)) * 100
#             print(f"    → Unsafe Rate Reduction: {reduction:+.2f}%")
    
#     ax_circle.set_xlabel('Timeslot (×10³)', fontsize=11)
#     ax_circle.set_ylabel('Cumulative Unsafe Rate (%)', fontsize=11)
#     ax_circle.set_title('Unsafe Route Rate\n(Fallback Policy Trigger)', fontsize=12, fontweight='bold')
#     ax_circle.legend(fontsize=10)
#     ax_circle.grid(True, alpha=0.3, linestyle='--')
#     ax_circle.set_xlim(left=0)
    
#     # --- Packet Loss Rate (Type 3 or average) ---
#     ax_loss = fig.add_subplot(gs[2, 2])
#     print("\n[PACKET LOSS RATE]")
    
#     # Use type 3 if available, otherwise use type 0
#     ppo_loss = read_log_file(os.path.join(ppo_log_dir, "loss_type3.log"))
#     if not ppo_loss:
#         ppo_loss = read_log_file(os.path.join(ppo_log_dir, "loss_type0.log"))
    
#     mappo_loss = read_log_file(os.path.join(mappo_log_dir, "loss_type3.log"))
#     if not mappo_loss:
#         mappo_loss = read_log_file(os.path.join(mappo_log_dir, "loss_type0.log"))
    
#     if ppo_loss:
#         ppo_smooth = smooth_curve(ppo_loss, window=window)
#         timesteps = np.arange(len(ppo_smooth)) / 1000.0
#         ax_loss.plot(timesteps, ppo_smooth, label='PPO (Original)', 
#                     color=PPO_COLOR, linewidth=2, alpha=0.9)
#         print(f"  PPO: avg loss rate = {np.mean(ppo_loss):.4f}")
    
#     if mappo_loss:
#         mappo_smooth = smooth_curve(mappo_loss, window=window)
#         timesteps = np.arange(len(mappo_smooth)) / 1000.0
#         ax_loss.plot(timesteps, mappo_smooth, label='MAPPO-CTDE', 
#                     color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
#         print(f"  MAPPO: avg loss rate = {np.mean(mappo_loss):.4f}")
        
#         if ppo_loss:
#             # Lower loss is better
#             reduction = (np.mean(ppo_loss) - np.mean(mappo_loss)) / np.mean(ppo_loss) * 100 if np.mean(ppo_loss) > 0 else 0
#             print(f"    → Loss Reduction: {reduction:+.2f}%")
    
#     ax_loss.set_xlabel('Timeslot (×10³)', fontsize=11)
#     ax_loss.set_ylabel('Packet Loss Rate', fontsize=11)
#     ax_loss.set_title('Packet Loss Rate', fontsize=12, fontweight='bold')
#     ax_loss.legend(fontsize=10)
#     ax_loss.grid(True, alpha=0.3, linestyle='--')
#     ax_loss.set_xlim(left=0)
    
#     # Overall title
#     fig.suptitle('PPO vs MAPPO-CTDE Performance Comparison\nInitialization Scenario - Abilene Topology',
#                  fontsize=16, fontweight='bold', y=0.995)
    
#     # Save figure
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"\n{'='*80}")
#     print(f"✓ Figure saved to: {output_file}")
#     print(f"{'='*80}")
    
#     # plt.show()
    
#     return fig


# def print_summary_table(ppo_log_dir, mappo_log_dir):
#     """Print a summary comparison table"""
    
#     print("\n" + "="*100)
#     print("SUMMARY COMPARISON TABLE: PPO vs MAPPO-CTDE (Initialization Scenario)")
#     print("="*100)
    
#     flow_types = [
#         (0, "Type 1 (Latency-sensitive)"),
#         (1, "Type 2 (Throughput-sensitive)"),
#         (2, "Type 3 (Latency-Throughput)"),
#         (3, "Type 4 (Latency-Loss)"),
#     ]
    
#     print(f"\n{'='*100}")
#     print(f"{'LATENCY (ms)':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     for ft_id, ft_name in flow_types:
#         ppo_data = read_log_file(os.path.join(ppo_log_dir, f"delay_type{ft_id}.log"))
#         mappo_data = read_log_file(os.path.join(mappo_log_dir, f"delay_type{ft_id}.log"))
        
#         if ppo_data and mappo_data:
#             ppo_avg = np.mean(ppo_data)
#             mappo_avg = np.mean(mappo_data)
#             improvement = (ppo_avg - mappo_avg) / ppo_avg * 100
#             print(f"{ft_name:<30} {ppo_avg:<20.2f} {mappo_avg:<20.2f} {improvement:+.2f}% ↓")
#         elif ppo_data:
#             print(f"{ft_name:<30} {np.mean(ppo_data):<20.2f} {'N/A':<20} {'N/A':<20}")
#         elif mappo_data:
#             print(f"{ft_name:<30} {'N/A':<20} {np.mean(mappo_data):<20.2f} {'N/A':<20}")
    
#     print(f"\n{'='*100}")
#     print(f"{'THROUGHPUT RATIO':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     for ft_id, ft_name in flow_types:
#         ppo_data = read_log_file(os.path.join(ppo_log_dir, f"throughput_type{ft_id}.log"))
#         mappo_data = read_log_file(os.path.join(mappo_log_dir, f"throughput_type{ft_id}.log"))
        
#         if ppo_data and mappo_data:
#             ppo_avg = np.mean(ppo_data)
#             mappo_avg = np.mean(mappo_data)
#             improvement = (mappo_avg - ppo_avg) / ppo_avg * 100
#             print(f"{ft_name:<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {improvement:+.2f}% ↑")
#         elif ppo_data:
#             print(f"{ft_name:<30} {np.mean(ppo_data):<20.4f} {'N/A':<20} {'N/A':<20}")
#         elif mappo_data:
#             print(f"{ft_name:<30} {'N/A':<20} {np.mean(mappo_data):<20.4f} {'N/A':<20}")
    
#     print(f"\n{'='*100}")
#     print(f"{'PACKET LOSS RATE':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     for ft_id, ft_name in flow_types:
#         ppo_data = read_log_file(os.path.join(ppo_log_dir, f"loss_type{ft_id}.log"))
#         mappo_data = read_log_file(os.path.join(mappo_log_dir, f"loss_type{ft_id}.log"))
        
#         if ppo_data and mappo_data:
#             ppo_avg = np.mean(ppo_data)
#             mappo_avg = np.mean(mappo_data)
#             if ppo_avg > 0:
#                 improvement = (ppo_avg - mappo_avg) / ppo_avg * 100
#                 print(f"{ft_name:<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {improvement:+.2f}% ↓")
#             else:
#                 print(f"{ft_name:<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {'N/A':<20}")
#         elif ppo_data:
#             print(f"{ft_name:<30} {np.mean(ppo_data):<20.4f} {'N/A':<20} {'N/A':<20}")
#         elif mappo_data:
#             print(f"{ft_name:<30} {'N/A':<20} {np.mean(mappo_data):<20.4f} {'N/A':<20}")
    
#     # Global metrics
#     print(f"\n{'='*100}")
#     print(f"{'GLOBAL METRICS':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     ppo_reward = read_log_file(os.path.join(ppo_log_dir, "globalrwd.log"))
#     mappo_reward = read_log_file(os.path.join(mappo_log_dir, "globalrwd.log"))
    
#     if ppo_reward and mappo_reward:
#         ppo_avg = np.mean(ppo_reward)
#         mappo_avg = np.mean(mappo_reward)
#         improvement = (mappo_avg - ppo_avg) / abs(ppo_avg) * 100 if ppo_avg != 0 else 0
#         print(f"{'Global Reward':<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {improvement:+.2f}% ↑")
    
#     ppo_circle = read_log_file(os.path.join(ppo_log_dir, "circle.log"))
#     mappo_circle = read_log_file(os.path.join(mappo_log_dir, "circle.log"))
    
#     if ppo_circle and mappo_circle:
#         ppo_rate = np.mean(ppo_circle) * 100
#         mappo_rate = np.mean(mappo_circle) * 100
#         reduction = ppo_rate - mappo_rate
#         print(f"{'Unsafe Route Rate (%)':<30} {ppo_rate:<20.2f} {mappo_rate:<20.2f} {reduction:+.2f}% ↓")
    
#     print("="*100)
#     print("\nNote: ↓ means lower is better, ↑ means higher is better")
#     print("="*100)


# def main():
#     parser = argparse.ArgumentParser(
#         description='Compare PPO vs MAPPO performance in DRL-OR initialization scenario'
#     )
#     parser.add_argument(
#         '--ppo-log', 
#         type=str, 
#         default='./log/initialization',
#         help='Directory containing PPO logs (default: ./log/initialization)'
#     )
#     parser.add_argument(
#         '--mappo-log', 
#         type=str, 
#         default='./log/mappo_initialization',
#         help='Directory containing MAPPO logs (default: ./log/mappo_initialization)'
#     )
#     parser.add_argument(
#         '--output', 
#         type=str, 
#         default='comparison_ppo_mappo.png',
#         help='Output filename for the plot (default: comparison_ppo_mappo.png)'
#     )
#     parser.add_argument(
#         '--window', 
#         type=int, 
#         default=1000,
#         help='Smoothing window size (default: 1000)'
#     )
#     parser.add_argument(
#         '--max-steps', 
#         type=int, 
#         default=None,
#         help='Maximum timesteps to plot (default: auto)'
#     )
#     parser.add_argument(
#         '--table-only', 
#         action='store_true',
#         help='Print summary table without plotting'
#     )
    
#     args = parser.parse_args()
    
#     # Check directories
#     ppo_exists = os.path.exists(args.ppo_log)
#     mappo_exists = os.path.exists(args.mappo_log)
    
#     if not ppo_exists and not mappo_exists:
#         print(f"Error: Neither log directory exists!")
#         print(f"  PPO log: {args.ppo_log}")
#         print(f"  MAPPO log: {args.mappo_log}")
#         return
    
#     if not ppo_exists:
#         print(f"Warning: PPO log directory '{args.ppo_log}' not found!")
#     if not mappo_exists:
#         print(f"Warning: MAPPO log directory '{args.mappo_log}' not found!")
    
#     print(f"\n{'='*60}")
#     print(f"PPO logs:   {args.ppo_log} {'✓' if ppo_exists else '✗'}")
#     print(f"MAPPO logs: {args.mappo_log} {'✓' if mappo_exists else '✗'}")
#     print(f"{'='*60}")
    
#     if args.table_only:
#         print_summary_table(args.ppo_log, args.mappo_log)
#         return
    
#     # Create comparison plot
#     plot_comparison(
#         ppo_log_dir=args.ppo_log,
#         mappo_log_dir=args.mappo_log,
#         output_file=args.output,
#         window=args.window,
#         max_steps=args.max_steps
#     )
    
#     # Print summary table
#     print_summary_table(args.ppo_log, args.mappo_log)


# if __name__ == "__main__":
#     main()














######## it works
# """
# Compare PPO vs MAPPO Performance in Initialization Scenario
# Based on DRL-OR Paper Figure 5

# This script compares the original PPO implementation with MAPPO-CTDE
# for the initialization scenario, showing:
# - Latency comparison for each flow type
# - Throughput ratio comparison for each flow type
# - Global reward comparison
# - Unsafe route rate comparison

# Note: Original DRL-OR PPO does not log value_loss, action_loss, ratio.
#       These metrics are only available for MAPPO.

# Usage:
#     python3 compare_ppo_mappo.py \
#         --ppo-log ./log/initialization \
#         --mappo-log ./log/mappo_initialization \
#         --output comparison_ppo_mappo.png
# """

# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend for saving without display
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import os


# def read_log_file(filepath):
#     """Read values from a log file"""
#     values = []
#     try:
#         with open(filepath, 'r') as f:
#             for line in f:
#                 try:
#                     values.append(float(line.strip()))
#                 except ValueError:
#                     continue
#     except FileNotFoundError:
#         return []
#     return values


# def smooth_curve(values, window=1000):
#     """Apply moving average smoothing using fast numpy convolution"""
#     if len(values) < window:
#         return values
    
#     # Use numpy convolution for FAST smoothing
#     values_array = np.array(values)
#     kernel = np.ones(window) / window
#     smoothed = np.convolve(values_array, kernel, mode='same')
#     return smoothed.tolist()


# def plot_comparison(ppo_log_dir, mappo_log_dir, output_file='comparison_ppo_mappo.png', 
#                     window=1000, max_steps=None):
#     """
#     Create comparison plots for PPO vs MAPPO
    
#     Layout: 3 rows × 3 columns
#     Row 1: Latency for type0, type1, type2
#     Row 2: Throughput for type0, type1, type2
#     Row 3: Global Reward, Unsafe Route Rate, Packet Loss Rate
#     """
    
#     # Create figure
#     fig = plt.figure(figsize=(18, 14))
#     gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    
#     # Flow types configuration (based on DRL-OR paper)
#     flow_types = [
#         {'id': 0, 'name': 'Type 1\n(Latency-sensitive)'},
#         {'id': 1, 'name': 'Type 2\n(Throughput-sensitive)'},
#         {'id': 2, 'name': 'Type 3\n(Latency-Throughput)'},
#     ]
    
#     # Colors
#     PPO_COLOR = '#1f77b4'      # Blue
#     MAPPO_COLOR = '#d62728'    # Red
    
#     print("\n" + "="*80)
#     print("COMPARING PPO vs MAPPO-CTDE - Initialization Scenario")
#     print("="*80)
    
#     # ========================================================================
#     # Row 1: Latency Comparison
#     # ========================================================================
#     print("\n[LATENCY COMPARISON]")
    
#     for i, flow_type in enumerate(flow_types):
#         ax = fig.add_subplot(gs[0, i])
        
#         # Read PPO data
#         ppo_file = os.path.join(ppo_log_dir, f"delay_type{flow_type['id']}.log")
#         ppo_delays = read_log_file(ppo_file)
        
#         # Read MAPPO data
#         mappo_file = os.path.join(mappo_log_dir, f"delay_type{flow_type['id']}.log")
#         mappo_delays = read_log_file(mappo_file)
        
#         # Determine max steps
#         if max_steps is None:
#             plot_max = max(len(ppo_delays) if ppo_delays else 0, 
#                           len(mappo_delays) if mappo_delays else 0)
#         else:
#             plot_max = max_steps
        
#         # Plot PPO
#         if ppo_delays:
#             ppo_smooth = smooth_curve(ppo_delays, window=window)
#             timesteps = np.arange(len(ppo_smooth)) / 1000.0
#             max_idx = min(len(ppo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], ppo_smooth[:max_idx], 
#                    label='PPO (Original)', color=PPO_COLOR, 
#                    linewidth=2, alpha=0.9)
#             ppo_final = np.mean(ppo_delays[-10000:]) if len(ppo_delays) > 10000 else np.mean(ppo_delays)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: PPO avg={np.mean(ppo_delays):.2f}ms, final={ppo_final:.2f}ms")
        
#         # Plot MAPPO
#         if mappo_delays:
#             mappo_smooth = smooth_curve(mappo_delays, window=window)
#             timesteps = np.arange(len(mappo_smooth)) / 1000.0
#             max_idx = min(len(mappo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], mappo_smooth[:max_idx], 
#                    label='MAPPO-CTDE', color=MAPPO_COLOR, 
#                    linewidth=2, linestyle='--', alpha=0.9)
#             mappo_final = np.mean(mappo_delays[-10000:]) if len(mappo_delays) > 10000 else np.mean(mappo_delays)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: MAPPO avg={np.mean(mappo_delays):.2f}ms, final={mappo_final:.2f}ms")
            
#             # Calculate improvement
#             if ppo_delays:
#                 improvement = (ppo_final - mappo_final) / ppo_final * 100
#                 print(f"    → Latency Reduction: {improvement:+.2f}%")
        
#         ax.set_xlabel('Timeslot (×10³)', fontsize=11)
#         ax.set_ylabel('Latency (ms)', fontsize=11)
#         ax.set_title(f'Latency - {flow_type["name"]}', fontsize=12, fontweight='bold')
#         ax.legend(fontsize=10, loc='upper right')
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.set_xlim(left=0)
    
#     # ========================================================================
#     # Row 2: Throughput Comparison
#     # ========================================================================
#     print("\n[THROUGHPUT COMPARISON]")
    
#     for i, flow_type in enumerate(flow_types):
#         ax = fig.add_subplot(gs[1, i])
        
#         # Read PPO data
#         ppo_file = os.path.join(ppo_log_dir, f"throughput_type{flow_type['id']}.log")
#         ppo_throughput = read_log_file(ppo_file)
        
#         # Read MAPPO data
#         mappo_file = os.path.join(mappo_log_dir, f"throughput_type{flow_type['id']}.log")
#         mappo_throughput = read_log_file(mappo_file)
        
#         # Determine max steps
#         if max_steps is None:
#             plot_max = max(len(ppo_throughput) if ppo_throughput else 0, 
#                           len(mappo_throughput) if mappo_throughput else 0)
#         else:
#             plot_max = max_steps
        
#         # Plot PPO
#         if ppo_throughput:
#             ppo_smooth = smooth_curve(ppo_throughput, window=window)
#             timesteps = np.arange(len(ppo_smooth)) / 1000.0
#             max_idx = min(len(ppo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], ppo_smooth[:max_idx], 
#                    label='PPO (Original)', color=PPO_COLOR, 
#                    linewidth=2, alpha=0.9)
#             ppo_final = np.mean(ppo_throughput[-10000:]) if len(ppo_throughput) > 10000 else np.mean(ppo_throughput)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: PPO avg={np.mean(ppo_throughput):.4f}, final={ppo_final:.4f}")
        
#         # Plot MAPPO
#         if mappo_throughput:
#             mappo_smooth = smooth_curve(mappo_throughput, window=window)
#             timesteps = np.arange(len(mappo_smooth)) / 1000.0
#             max_idx = min(len(mappo_smooth), plot_max)
#             ax.plot(timesteps[:max_idx], mappo_smooth[:max_idx], 
#                    label='MAPPO-CTDE', color=MAPPO_COLOR, 
#                    linewidth=2, linestyle='--', alpha=0.9)
#             mappo_final = np.mean(mappo_throughput[-10000:]) if len(mappo_throughput) > 10000 else np.mean(mappo_throughput)
#             print(f"  {flow_type['name'].replace(chr(10), ' ')}: MAPPO avg={np.mean(mappo_throughput):.4f}, final={mappo_final:.4f}")
            
#             # Calculate improvement
#             if ppo_throughput:
#                 improvement = (mappo_final - ppo_final) / ppo_final * 100
#                 print(f"    → Throughput Improvement: {improvement:+.2f}%")
        
#         ax.set_xlabel('Timeslot (×10³)', fontsize=11)
#         ax.set_ylabel('Throughput Ratio', fontsize=11)
#         ax.set_title(f'Throughput - {flow_type["name"]}', fontsize=12, fontweight='bold')
#         ax.legend(fontsize=10, loc='lower right')
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.set_xlim(left=0)
#         ax.set_ylim([0.85, 1.02])
    
#     # ========================================================================
#     # Row 3: Global Reward, Unsafe Rate, Packet Loss
#     # ========================================================================
    
#     # --- Global Reward Comparison ---
#     ax_reward = fig.add_subplot(gs[2, 0])
#     print("\n[GLOBAL REWARD COMPARISON]")
    
#     ppo_reward = read_log_file(os.path.join(ppo_log_dir, "globalrwd.log"))
#     mappo_reward = read_log_file(os.path.join(mappo_log_dir, "globalrwd.log"))
    
#     if ppo_reward:
#         ppo_smooth = smooth_curve(ppo_reward, window=window)
#         timesteps = np.arange(len(ppo_smooth)) / 1000.0
#         ax_reward.plot(timesteps, ppo_smooth, label='PPO (Original)', 
#                       color=PPO_COLOR, linewidth=2, alpha=0.9)
#         ppo_final = np.mean(ppo_reward[-10000:]) if len(ppo_reward) > 10000 else np.mean(ppo_reward)
#         print(f"  PPO: avg={np.mean(ppo_reward):.4f}, final={ppo_final:.4f}")
    
#     if mappo_reward:
#         mappo_smooth = smooth_curve(mappo_reward, window=window)
#         timesteps = np.arange(len(mappo_smooth)) / 1000.0
#         ax_reward.plot(timesteps, mappo_smooth, label='MAPPO-CTDE', 
#                       color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
#         mappo_final = np.mean(mappo_reward[-10000:]) if len(mappo_reward) > 10000 else np.mean(mappo_reward)
#         print(f"  MAPPO: avg={np.mean(mappo_reward):.4f}, final={mappo_final:.4f}")
        
#         if ppo_reward:
#             # Higher reward is better
#             improvement = (mappo_final - ppo_final) / abs(ppo_final) * 100 if ppo_final != 0 else 0
#             print(f"    → Reward Improvement: {improvement:+.2f}%")
    
#     ax_reward.set_xlabel('Timeslot (×10³)', fontsize=11)
#     ax_reward.set_ylabel('Global Reward', fontsize=11)
#     ax_reward.set_title('Global Reward', fontsize=12, fontweight='bold')
#     ax_reward.legend(fontsize=10)
#     ax_reward.grid(True, alpha=0.3, linestyle='--')
#     ax_reward.set_xlim(left=0)
    
#     # --- Unsafe Route Rate (Circle Flag) ---
#     ax_circle = fig.add_subplot(gs[2, 1])
#     print("\n[UNSAFE ROUTE RATE (Fallback Policy Usage)]")
    
#     print("  Reading PPO circle.log...", end=" ", flush=True)
#     ppo_circle = read_log_file(os.path.join(ppo_log_dir, "circle.log"))
#     print(f"done ({len(ppo_circle)} samples)" if ppo_circle else "not found")
    
#     print("  Reading MAPPO circle.log...", end=" ", flush=True)
#     mappo_circle = read_log_file(os.path.join(mappo_log_dir, "circle.log"))
#     print(f"done ({len(mappo_circle)} samples)" if mappo_circle else "not found")
    
#     if ppo_circle:
#         # Use numpy cumsum for FAST cumulative calculation (instead of slow loop)
#         ppo_array = np.array(ppo_circle)
#         ppo_cumulative = (np.cumsum(ppo_array) / np.arange(1, len(ppo_array) + 1)) * 100
#         ppo_smooth = smooth_curve(ppo_cumulative.tolist(), window=window)
#         timesteps = np.arange(len(ppo_smooth)) / 1000.0
#         ax_circle.plot(timesteps, ppo_smooth, label='PPO (Original)', 
#                       color=PPO_COLOR, linewidth=2, alpha=0.9)
#         print(f"  PPO: total unsafe rate = {np.mean(ppo_circle)*100:.2f}%")
    
#     if mappo_circle:
#         # Use numpy cumsum for FAST cumulative calculation
#         mappo_array = np.array(mappo_circle)
#         mappo_cumulative = (np.cumsum(mappo_array) / np.arange(1, len(mappo_array) + 1)) * 100
#         mappo_smooth = smooth_curve(mappo_cumulative.tolist(), window=window)
#         timesteps = np.arange(len(mappo_smooth)) / 1000.0
#         ax_circle.plot(timesteps, mappo_smooth, label='MAPPO-CTDE', 
#                       color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
#         print(f"  MAPPO: total unsafe rate = {np.mean(mappo_circle)*100:.2f}%")
        
#         if ppo_circle:
#             # Lower unsafe rate is better
#             reduction = (np.mean(ppo_circle) - np.mean(mappo_circle)) * 100
#             print(f"    → Unsafe Rate Reduction: {reduction:+.2f}%")
    
#     ax_circle.set_xlabel('Timeslot (×10³)', fontsize=11)
#     ax_circle.set_ylabel('Cumulative Unsafe Rate (%)', fontsize=11)
#     ax_circle.set_title('Unsafe Route Rate\n(Fallback Policy Trigger)', fontsize=12, fontweight='bold')
#     ax_circle.legend(fontsize=10)
#     ax_circle.grid(True, alpha=0.3, linestyle='--')
#     ax_circle.set_xlim(left=0)
    
#     # --- Packet Loss Rate (Type 3 or average) ---
#     ax_loss = fig.add_subplot(gs[2, 2])
#     print("\n[PACKET LOSS RATE]")
    
#     # Use type 3 if available, otherwise use type 0
#     ppo_loss = read_log_file(os.path.join(ppo_log_dir, "loss_type3.log"))
#     if not ppo_loss:
#         ppo_loss = read_log_file(os.path.join(ppo_log_dir, "loss_type0.log"))
    
#     mappo_loss = read_log_file(os.path.join(mappo_log_dir, "loss_type3.log"))
#     if not mappo_loss:
#         mappo_loss = read_log_file(os.path.join(mappo_log_dir, "loss_type0.log"))
    
#     if ppo_loss:
#         ppo_smooth = smooth_curve(ppo_loss, window=window)
#         timesteps = np.arange(len(ppo_smooth)) / 1000.0
#         ax_loss.plot(timesteps, ppo_smooth, label='PPO (Original)', 
#                     color=PPO_COLOR, linewidth=2, alpha=0.9)
#         print(f"  PPO: avg loss rate = {np.mean(ppo_loss):.4f}")
    
#     if mappo_loss:
#         mappo_smooth = smooth_curve(mappo_loss, window=window)
#         timesteps = np.arange(len(mappo_smooth)) / 1000.0
#         ax_loss.plot(timesteps, mappo_smooth, label='MAPPO-CTDE', 
#                     color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
#         print(f"  MAPPO: avg loss rate = {np.mean(mappo_loss):.4f}")
        
#         if ppo_loss:
#             # Lower loss is better
#             reduction = (np.mean(ppo_loss) - np.mean(mappo_loss)) / np.mean(ppo_loss) * 100 if np.mean(ppo_loss) > 0 else 0
#             print(f"    → Loss Reduction: {reduction:+.2f}%")
    
#     ax_loss.set_xlabel('Timeslot (×10³)', fontsize=11)
#     ax_loss.set_ylabel('Packet Loss Rate', fontsize=11)
#     ax_loss.set_title('Packet Loss Rate', fontsize=12, fontweight='bold')
#     ax_loss.legend(fontsize=10)
#     ax_loss.grid(True, alpha=0.3, linestyle='--')
#     ax_loss.set_xlim(left=0)
    
#     # Overall title
#     fig.suptitle('PPO vs MAPPO-CTDE Performance Comparison\nInitialization Scenario - Abilene Topology',
#                  fontsize=16, fontweight='bold', y=0.995)
    
#     # Save figure
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"\n{'='*80}")
#     print(f"✓ Figure saved to: {output_file}")
#     print(f"{'='*80}")
    
#     # Close figure to free memory
#     plt.close(fig)
    
#     return fig


# def print_summary_table(ppo_log_dir, mappo_log_dir):
#     """Print a summary comparison table"""
    
#     print("\n" + "="*100)
#     print("SUMMARY COMPARISON TABLE: PPO vs MAPPO-CTDE (Initialization Scenario)")
#     print("="*100)
    
#     flow_types = [
#         (0, "Type 1 (Latency-sensitive)"),
#         (1, "Type 2 (Throughput-sensitive)"),
#         (2, "Type 3 (Latency-Throughput)"),
#         (3, "Type 4 (Latency-Loss)"),
#     ]
    
#     print(f"\n{'='*100}")
#     print(f"{'LATENCY (ms)':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     for ft_id, ft_name in flow_types:
#         ppo_data = read_log_file(os.path.join(ppo_log_dir, f"delay_type{ft_id}.log"))
#         mappo_data = read_log_file(os.path.join(mappo_log_dir, f"delay_type{ft_id}.log"))
        
#         if ppo_data and mappo_data:
#             ppo_avg = np.mean(ppo_data)
#             mappo_avg = np.mean(mappo_data)
#             improvement = (ppo_avg - mappo_avg) / ppo_avg * 100
#             print(f"{ft_name:<30} {ppo_avg:<20.2f} {mappo_avg:<20.2f} {improvement:+.2f}% ↓")
#         elif ppo_data:
#             print(f"{ft_name:<30} {np.mean(ppo_data):<20.2f} {'N/A':<20} {'N/A':<20}")
#         elif mappo_data:
#             print(f"{ft_name:<30} {'N/A':<20} {np.mean(mappo_data):<20.2f} {'N/A':<20}")
    
#     print(f"\n{'='*100}")
#     print(f"{'THROUGHPUT RATIO':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     for ft_id, ft_name in flow_types:
#         ppo_data = read_log_file(os.path.join(ppo_log_dir, f"throughput_type{ft_id}.log"))
#         mappo_data = read_log_file(os.path.join(mappo_log_dir, f"throughput_type{ft_id}.log"))
        
#         if ppo_data and mappo_data:
#             ppo_avg = np.mean(ppo_data)
#             mappo_avg = np.mean(mappo_data)
#             improvement = (mappo_avg - ppo_avg) / ppo_avg * 100
#             print(f"{ft_name:<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {improvement:+.2f}% ↑")
#         elif ppo_data:
#             print(f"{ft_name:<30} {np.mean(ppo_data):<20.4f} {'N/A':<20} {'N/A':<20}")
#         elif mappo_data:
#             print(f"{ft_name:<30} {'N/A':<20} {np.mean(mappo_data):<20.4f} {'N/A':<20}")
    
#     print(f"\n{'='*100}")
#     print(f"{'PACKET LOSS RATE':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     for ft_id, ft_name in flow_types:
#         ppo_data = read_log_file(os.path.join(ppo_log_dir, f"loss_type{ft_id}.log"))
#         mappo_data = read_log_file(os.path.join(mappo_log_dir, f"loss_type{ft_id}.log"))
        
#         if ppo_data and mappo_data:
#             ppo_avg = np.mean(ppo_data)
#             mappo_avg = np.mean(mappo_data)
#             if ppo_avg > 0:
#                 improvement = (ppo_avg - mappo_avg) / ppo_avg * 100
#                 print(f"{ft_name:<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {improvement:+.2f}% ↓")
#             else:
#                 print(f"{ft_name:<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {'N/A':<20}")
#         elif ppo_data:
#             print(f"{ft_name:<30} {np.mean(ppo_data):<20.4f} {'N/A':<20} {'N/A':<20}")
#         elif mappo_data:
#             print(f"{ft_name:<30} {'N/A':<20} {np.mean(mappo_data):<20.4f} {'N/A':<20}")
    
#     # Global metrics
#     print(f"\n{'='*100}")
#     print(f"{'GLOBAL METRICS':<30} {'PPO':<20} {'MAPPO':<20} {'Improvement':<20}")
#     print(f"{'='*100}")
    
#     ppo_reward = read_log_file(os.path.join(ppo_log_dir, "globalrwd.log"))
#     mappo_reward = read_log_file(os.path.join(mappo_log_dir, "globalrwd.log"))
    
#     if ppo_reward and mappo_reward:
#         ppo_avg = np.mean(ppo_reward)
#         mappo_avg = np.mean(mappo_reward)
#         improvement = (mappo_avg - ppo_avg) / abs(ppo_avg) * 100 if ppo_avg != 0 else 0
#         print(f"{'Global Reward':<30} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {improvement:+.2f}% ↑")
    
#     ppo_circle = read_log_file(os.path.join(ppo_log_dir, "circle.log"))
#     mappo_circle = read_log_file(os.path.join(mappo_log_dir, "circle.log"))
    
#     if ppo_circle and mappo_circle:
#         ppo_rate = np.mean(ppo_circle) * 100
#         mappo_rate = np.mean(mappo_circle) * 100
#         reduction = ppo_rate - mappo_rate
#         print(f"{'Unsafe Route Rate (%)':<30} {ppo_rate:<20.2f} {mappo_rate:<20.2f} {reduction:+.2f}% ↓")
    
#     print("="*100)
#     print("\nNote: ↓ means lower is better, ↑ means higher is better")
#     print("="*100)


# def main():
#     parser = argparse.ArgumentParser(
#         description='Compare PPO vs MAPPO performance in DRL-OR initialization scenario'
#     )
#     parser.add_argument(
#         '--ppo-log', 
#         type=str, 
#         default='./log/initialization',
#         help='Directory containing PPO logs (default: ./log/initialization)'
#     )
#     parser.add_argument(
#         '--mappo-log', 
#         type=str, 
#         default='./log/mappo_initialization',
#         help='Directory containing MAPPO logs (default: ./log/mappo_initialization)'
#     )
#     parser.add_argument(
#         '--output', 
#         type=str, 
#         default='comparison_ppo_mappo.png',
#         help='Output filename for the plot (default: comparison_ppo_mappo.png)'
#     )
#     parser.add_argument(
#         '--window', 
#         type=int, 
#         default=1000,
#         help='Smoothing window size (default: 1000)'
#     )
#     parser.add_argument(
#         '--max-steps', 
#         type=int, 
#         default=None,
#         help='Maximum timesteps to plot (default: auto)'
#     )
#     parser.add_argument(
#         '--table-only', 
#         action='store_true',
#         help='Print summary table without plotting'
#     )
    
#     args = parser.parse_args()
    
#     # Check directories
#     ppo_exists = os.path.exists(args.ppo_log)
#     mappo_exists = os.path.exists(args.mappo_log)
    
#     if not ppo_exists and not mappo_exists:
#         print(f"Error: Neither log directory exists!")
#         print(f"  PPO log: {args.ppo_log}")
#         print(f"  MAPPO log: {args.mappo_log}")
#         return
    
#     if not ppo_exists:
#         print(f"Warning: PPO log directory '{args.ppo_log}' not found!")
#     if not mappo_exists:
#         print(f"Warning: MAPPO log directory '{args.mappo_log}' not found!")
    
#     print(f"\n{'='*60}")
#     print(f"PPO logs:   {args.ppo_log} {'✓' if ppo_exists else '✗'}")
#     print(f"MAPPO logs: {args.mappo_log} {'✓' if mappo_exists else '✗'}")
#     print(f"{'='*60}")
    
#     if args.table_only:
#         print_summary_table(args.ppo_log, args.mappo_log)
#         return
    
#     # Create comparison plot
#     plot_comparison(
#         ppo_log_dir=args.ppo_log,
#         mappo_log_dir=args.mappo_log,
#         output_file=args.output,
#         window=args.window,
#         max_steps=args.max_steps
#     )
    
#     # Print summary table
#     print_summary_table(args.ppo_log, args.mappo_log)


# if __name__ == "__main__":
#     main()




















################## all scenarios ppo and mappo
"""
Compare PPO vs MAPPO Performance Across All 3 Scenarios
Based on DRL-OR Paper Figure 5

Scenarios:
1. Initialization
2. Link Failure  
3. Traffic Change

Usage:
    python3 compare_ppo_mappo.py \
        --log-dir ./drl-or/log \
        --output comparison_all_scenarios.png
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def read_log_file(filepath):
    """Read values from a log file"""
    values = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    values.append(float(line.strip()))
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return values


def smooth_curve(values, window=1000):
    """Apply moving average smoothing using fast numpy convolution"""
    if len(values) < window:
        return values
    
    values_array = np.array(values)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values_array, kernel, mode='same')
    return smoothed.tolist()


def plot_all_scenarios(log_dir, output_file='comparison_all_scenarios.png', window=1000):
    """
    Create comprehensive comparison plots for PPO vs MAPPO across all scenarios
    
    Layout: 4 rows × 3 columns
    Row 1: Latency (Type 1) for all 3 scenarios
    Row 2: Throughput (Type 2) for all 3 scenarios  
    Row 3: Global Reward for all 3 scenarios
    Row 4: Unsafe Route Rate for all 3 scenarios
    """
    
    # Create figure
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
    
    # Scenarios
    scenarios = [
        {'name': 'initialization', 'mappo_name': 'mappo_initialization', 
         'title': 'Initialization', 'max_steps': 300000},
        {'name': 'link_failure', 'mappo_name': 'mappo_link_failure',
         'title': 'Link Failure', 'max_steps': 180000},
        {'name': 'traffic_change', 'mappo_name': 'mappo_traffic_change',
         'title': 'Traffic Change', 'max_steps': 180000},
    ]
    
    # Colors
    PPO_COLOR = '#1f77b4'      # Blue
    MAPPO_COLOR = '#d62728'    # Red
    
    print("\n" + "="*100)
    print("COMPARING PPO vs MAPPO-CTDE - All Scenarios")
    print("="*100)
    
    # ========================================================================
    # Row 1: Latency Comparison (Type 0 - Latency-sensitive)
    # ========================================================================
    print("\n[ROW 1: LATENCY COMPARISON - Type 1 (Latency-sensitive)]")
    
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, col])
        
        ppo_dir = os.path.join(log_dir, scenario['name'])
        mappo_dir = os.path.join(log_dir, scenario['mappo_name'])
        
        # Read data
        ppo_data = read_log_file(os.path.join(ppo_dir, "delay_type0.log"))
        mappo_data = read_log_file(os.path.join(mappo_dir, "delay_type0.log"))
        
        print(f"\n  {scenario['title']}:")
        
        if ppo_data:
            ppo_smooth = smooth_curve(ppo_data, window=window)
            timesteps = np.arange(len(ppo_smooth)) / 1000.0
            ax.plot(timesteps, ppo_smooth, label='PPO', color=PPO_COLOR, linewidth=2, alpha=0.9)
            print(f"    PPO: avg={np.mean(ppo_data):.2f}ms, samples={len(ppo_data)}")
        
        if mappo_data:
            mappo_smooth = smooth_curve(mappo_data, window=window)
            timesteps = np.arange(len(mappo_smooth)) / 1000.0
            ax.plot(timesteps, mappo_smooth, label='MAPPO', color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
            print(f"    MAPPO: avg={np.mean(mappo_data):.2f}ms, samples={len(mappo_data)}")
            
            if ppo_data:
                improvement = (np.mean(ppo_data) - np.mean(mappo_data)) / np.mean(ppo_data) * 100
                print(f"    → Improvement: {improvement:+.2f}%")
        
        ax.set_xlabel('Timeslot (×10³)', fontsize=10)
        ax.set_ylabel('Latency (ms)', fontsize=10)
        ax.set_title(f'Latency - {scenario["title"]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
    
    # ========================================================================
    # Row 2: Throughput Comparison (Type 1 - Throughput-sensitive)
    # ========================================================================
    print("\n[ROW 2: THROUGHPUT COMPARISON - Type 2 (Throughput-sensitive)]")
    
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[1, col])
        
        ppo_dir = os.path.join(log_dir, scenario['name'])
        mappo_dir = os.path.join(log_dir, scenario['mappo_name'])
        
        ppo_data = read_log_file(os.path.join(ppo_dir, "throughput_type1.log"))
        mappo_data = read_log_file(os.path.join(mappo_dir, "throughput_type1.log"))
        
        print(f"\n  {scenario['title']}:")
        
        if ppo_data:
            ppo_smooth = smooth_curve(ppo_data, window=window)
            timesteps = np.arange(len(ppo_smooth)) / 1000.0
            ax.plot(timesteps, ppo_smooth, label='PPO', color=PPO_COLOR, linewidth=2, alpha=0.9)
            print(f"    PPO: avg={np.mean(ppo_data):.4f}, samples={len(ppo_data)}")
        
        if mappo_data:
            mappo_smooth = smooth_curve(mappo_data, window=window)
            timesteps = np.arange(len(mappo_smooth)) / 1000.0
            ax.plot(timesteps, mappo_smooth, label='MAPPO', color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
            print(f"    MAPPO: avg={np.mean(mappo_data):.4f}, samples={len(mappo_data)}")
            
            if ppo_data:
                improvement = (np.mean(mappo_data) - np.mean(ppo_data)) / np.mean(ppo_data) * 100
                print(f"    → Improvement: {improvement:+.2f}%")
        
        ax.set_xlabel('Timeslot (×10³)', fontsize=10)
        ax.set_ylabel('Throughput Ratio', fontsize=10)
        ax.set_title(f'Throughput - {scenario["title"]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim([0.85, 1.02])
    
    # ========================================================================
    # Row 3: Global Reward Comparison
    # ========================================================================
    print("\n[ROW 3: GLOBAL REWARD COMPARISON]")
    
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[2, col])
        
        ppo_dir = os.path.join(log_dir, scenario['name'])
        mappo_dir = os.path.join(log_dir, scenario['mappo_name'])
        
        ppo_data = read_log_file(os.path.join(ppo_dir, "globalrwd.log"))
        mappo_data = read_log_file(os.path.join(mappo_dir, "globalrwd.log"))
        
        print(f"\n  {scenario['title']}:")
        
        if ppo_data:
            ppo_smooth = smooth_curve(ppo_data, window=window)
            timesteps = np.arange(len(ppo_smooth)) / 1000.0
            ax.plot(timesteps, ppo_smooth, label='PPO', color=PPO_COLOR, linewidth=2, alpha=0.9)
            print(f"    PPO: avg={np.mean(ppo_data):.4f}")
        
        if mappo_data:
            mappo_smooth = smooth_curve(mappo_data, window=window)
            timesteps = np.arange(len(mappo_smooth)) / 1000.0
            ax.plot(timesteps, mappo_smooth, label='MAPPO', color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
            print(f"    MAPPO: avg={np.mean(mappo_data):.4f}")
        
        ax.set_xlabel('Timeslot (×10³)', fontsize=10)
        ax.set_ylabel('Global Reward', fontsize=10)
        ax.set_title(f'Global Reward - {scenario["title"]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
    
    # ========================================================================
    # Row 4: Unsafe Route Rate Comparison
    # ========================================================================
    print("\n[ROW 4: UNSAFE ROUTE RATE COMPARISON]")
    
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[3, col])
        
        ppo_dir = os.path.join(log_dir, scenario['name'])
        mappo_dir = os.path.join(log_dir, scenario['mappo_name'])
        
        print(f"\n  {scenario['title']}:")
        print("    Reading PPO circle.log...", end=" ", flush=True)
        ppo_data = read_log_file(os.path.join(ppo_dir, "circle.log"))
        print(f"done ({len(ppo_data)} samples)" if ppo_data else "not found")
        
        print("    Reading MAPPO circle.log...", end=" ", flush=True)
        mappo_data = read_log_file(os.path.join(mappo_dir, "circle.log"))
        print(f"done ({len(mappo_data)} samples)" if mappo_data else "not found")
        
        if ppo_data:
            ppo_array = np.array(ppo_data)
            ppo_cumulative = (np.cumsum(ppo_array) / np.arange(1, len(ppo_array) + 1)) * 100
            ppo_smooth = smooth_curve(ppo_cumulative.tolist(), window=window)
            timesteps = np.arange(len(ppo_smooth)) / 1000.0
            ax.plot(timesteps, ppo_smooth, label='PPO', color=PPO_COLOR, linewidth=2, alpha=0.9)
            print(f"    PPO: total unsafe rate = {np.mean(ppo_data)*100:.2f}%")
        
        if mappo_data:
            mappo_array = np.array(mappo_data)
            mappo_cumulative = (np.cumsum(mappo_array) / np.arange(1, len(mappo_array) + 1)) * 100
            mappo_smooth = smooth_curve(mappo_cumulative.tolist(), window=window)
            timesteps = np.arange(len(mappo_smooth)) / 1000.0
            ax.plot(timesteps, mappo_smooth, label='MAPPO', color=MAPPO_COLOR, linewidth=2, linestyle='--', alpha=0.9)
            print(f"    MAPPO: total unsafe rate = {np.mean(mappo_data)*100:.2f}%")
        
        ax.set_xlabel('Timeslot (×10³)', fontsize=10)
        ax.set_ylabel('Cumulative Unsafe Rate (%)', fontsize=10)
        ax.set_title(f'Unsafe Route Rate - {scenario["title"]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
    
    # Overall title
    fig.suptitle('PPO vs MAPPO-CTDE Performance Comparison\nAcross All Scenarios (Abilene Topology)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{'='*100}")
    print(f"✓ Figure saved to: {output_file}")
    print(f"{'='*100}")
    
    plt.close(fig)
    return fig


def plot_mappo_training_metrics(log_dir, output_file='mappo_training_metrics.png', window=100):
    """
    Plot MAPPO-specific training metrics (value_loss, action_loss, ratio)
    These are only available in MAPPO logs
    """
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    
    scenarios = [
        {'name': 'mappo_initialization', 'title': 'Initialization'},
        {'name': 'mappo_link_failure', 'title': 'Link Failure'},
        {'name': 'mappo_traffic_change', 'title': 'Traffic Change'},
    ]
    
    metrics = [
        {'file': 'value_loss.log', 'ylabel': 'Value Loss', 'row': 0, 'color': '#2ca02c'},
        {'file': 'action_loss.log', 'ylabel': 'Action Loss', 'row': 1, 'color': '#9467bd'},
        {'file': 'ratio.log', 'ylabel': 'Importance Ratio', 'row': 2, 'color': '#ff7f0e'},
    ]
    
    print("\n" + "="*100)
    print("MAPPO TRAINING METRICS (Only available in MAPPO)")
    print("="*100)
    
    for metric in metrics:
        print(f"\n[{metric['ylabel'].upper()}]")
        
        for col, scenario in enumerate(scenarios):
            ax = fig.add_subplot(gs[metric['row'], col])
            
            filepath = os.path.join(log_dir, scenario['name'], metric['file'])
            data = read_log_file(filepath)
            
            if data:
                data_smooth = smooth_curve(data, window=window)
                timesteps = np.arange(len(data_smooth))
                ax.plot(timesteps, data_smooth, color=metric['color'], linewidth=1.5, alpha=0.9)
                print(f"  {scenario['title']}: avg={np.mean(data):.4f}, final={np.mean(data[-100:]) if len(data) > 100 else np.mean(data):.4f}")
            else:
                print(f"  {scenario['title']}: NOT FOUND")
            
            ax.set_xlabel('Update Step', fontsize=10)
            ax.set_ylabel(metric['ylabel'], fontsize=10)
            ax.set_title(f'{metric["ylabel"]} - {scenario["title"]}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(left=0)
    
    fig.suptitle('MAPPO Training Metrics\n(Value Loss, Action Loss, Importance Sampling Ratio)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{'='*100}")
    print(f"✓ Figure saved to: {output_file}")
    print(f"{'='*100}")
    
    plt.close(fig)
    return fig


def print_summary_table(log_dir):
    """Print comprehensive summary table"""
    
    scenarios = [
        ('initialization', 'mappo_initialization', 'Initialization'),
        ('link_failure', 'mappo_link_failure', 'Link Failure'),
        ('traffic_change', 'mappo_traffic_change', 'Traffic Change'),
    ]
    
    flow_types = [(0, "Type 1"), (1, "Type 2"), (2, "Type 3"), (3, "Type 4")]
    
    print("\n" + "="*120)
    print("COMPREHENSIVE SUMMARY TABLE: PPO vs MAPPO-CTDE")
    print("="*120)
    
    for ppo_name, mappo_name, title in scenarios:
        ppo_dir = os.path.join(log_dir, ppo_name)
        mappo_dir = os.path.join(log_dir, mappo_name)
        
        print(f"\n{'─'*120}")
        print(f"  {title.upper()}")
        print(f"{'─'*120}")
        print(f"  {'Metric':<25} {'Flow Type':<12} {'PPO':<15} {'MAPPO':<15} {'Improvement':<15} {'Better':<10}")
        print(f"  {'-'*110}")
        
        # Latency
        for ft_id, ft_name in flow_types:
            ppo_data = read_log_file(os.path.join(ppo_dir, f"delay_type{ft_id}.log"))
            mappo_data = read_log_file(os.path.join(mappo_dir, f"delay_type{ft_id}.log"))
            
            if ppo_data and mappo_data:
                ppo_avg = np.mean(ppo_data)
                mappo_avg = np.mean(mappo_data)
                improvement = (ppo_avg - mappo_avg) / ppo_avg * 100
                better = "MAPPO ✓" if improvement > 0 else "PPO ✓"
                print(f"  {'Latency (ms)':<25} {ft_name:<12} {ppo_avg:<15.2f} {mappo_avg:<15.2f} {improvement:+.2f}%{'':<8} {better:<10}")
        
        # Throughput
        for ft_id, ft_name in flow_types:
            ppo_data = read_log_file(os.path.join(ppo_dir, f"throughput_type{ft_id}.log"))
            mappo_data = read_log_file(os.path.join(mappo_dir, f"throughput_type{ft_id}.log"))
            
            if ppo_data and mappo_data:
                ppo_avg = np.mean(ppo_data)
                mappo_avg = np.mean(mappo_data)
                improvement = (mappo_avg - ppo_avg) / ppo_avg * 100
                better = "MAPPO ✓" if improvement > 0 else "PPO ✓"
                print(f"  {'Throughput Ratio':<25} {ft_name:<12} {ppo_avg:<15.4f} {mappo_avg:<15.4f} {improvement:+.2f}%{'':<8} {better:<10}")
        
        # Global metrics
        ppo_reward = read_log_file(os.path.join(ppo_dir, "globalrwd.log"))
        mappo_reward = read_log_file(os.path.join(mappo_dir, "globalrwd.log"))
        
        if ppo_reward and mappo_reward:
            ppo_avg = np.mean(ppo_reward)
            mappo_avg = np.mean(mappo_reward)
            improvement = (mappo_avg - ppo_avg) / abs(ppo_avg) * 100 if ppo_avg != 0 else 0
            better = "MAPPO ✓" if improvement > 0 else "PPO ✓"
            print(f"  {'Global Reward':<25} {'All':<12} {ppo_avg:<15.4f} {mappo_avg:<15.4f} {improvement:+.2f}%{'':<8} {better:<10}")
        
        ppo_circle = read_log_file(os.path.join(ppo_dir, "circle.log"))
        mappo_circle = read_log_file(os.path.join(mappo_dir, "circle.log"))
        
        if ppo_circle and mappo_circle:
            ppo_rate = np.mean(ppo_circle) * 100
            mappo_rate = np.mean(mappo_circle) * 100
            reduction = ppo_rate - mappo_rate
            better = "MAPPO ✓" if reduction > 0 else "PPO ✓"
            print(f"  {'Unsafe Rate (%)':<25} {'All':<12} {ppo_rate:<15.2f} {mappo_rate:<15.2f} {reduction:+.2f}%{'':<8} {better:<10}")
    
    print("\n" + "="*120)
    print("Note: For Latency and Unsafe Rate, LOWER is better (positive improvement = MAPPO better)")
    print("      For Throughput and Global Reward, HIGHER is better (positive improvement = MAPPO better)")
    print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description='Compare PPO vs MAPPO across all scenarios'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='./log',
        help='Directory containing all log folders (default: ./log)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='comparison_all_scenarios.png',
        help='Output filename for main comparison plot'
    )
    parser.add_argument(
        '--output-training', 
        type=str, 
        default='mappo_training_metrics.png',
        help='Output filename for MAPPO training metrics plot'
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=1000,
        help='Smoothing window size (default: 1000)'
    )
    parser.add_argument(
        '--table-only', 
        action='store_true',
        help='Print summary table without plotting'
    )
    
    args = parser.parse_args()
    
    # Check directory
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' not found!")
        return
    
    # List available scenarios
    print(f"\nLog directory: {args.log_dir}")
    print("Available scenarios:")
    for item in sorted(os.listdir(args.log_dir)):
        if os.path.isdir(os.path.join(args.log_dir, item)):
            print(f"  ✓ {item}")
    
    if args.table_only:
        print_summary_table(args.log_dir)
        return
    
    # Create plots
    plot_all_scenarios(args.log_dir, args.output, args.window)
    plot_mappo_training_metrics(args.log_dir, args.output_training, window=100)
    
    # Print summary
    print_summary_table(args.log_dir)


if __name__ == "__main__":
    main()