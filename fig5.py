
######## it works
# """
# Plot Figure 5 from DRL-OR Paper
# Reproduces the 6 subplots showing latency and throughput ratio
# for three scenarios: initialization, link failure, and traffic change

# Usage:
#     python3 plot_figure5.py --log-dir ./log --output fig5_reproduction.png
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
#         print(f"Warning: {filepath} not found")
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


# def plot_figure5(log_dir='./log', output_file='figure5_reproduction.png', window=1000):
#     """
#     Create Figure 5 with 6 subplots (2 rows × 3 columns)
    
#     Row 1: Latency for (a) initialization, (b) link failure, (c) traffic change
#     Row 2: Throughput ratio for (d) initialization, (e) link failure, (f) traffic change
#     """
    
#     # Create figure with 2 rows, 3 columns
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
#     # Scenario configurations
#     scenarios = [
#         {
#             'name': 'initialization',
#             'max_steps': 300000,
#             'col': 0,
#             'title_lat': '(a) Latency under initialization',
#             'title_thr': '(d) Thrpt. ratio under initialization'
#         },
#         {
#             'name': 'link_failure',
#             'max_steps': 180000,
#             'col': 1,
#             'title_lat': '(b) Latency under link failure',
#             'title_thr': '(e) Thrpt. ratio under link failure'
#         },
#         {
#             'name': 'traffic_change',
#             'max_steps': 180000,
#             'col': 2,
#             'title_lat': '(c) Latency under traffic change',
#             'title_thr': '(f) Thrpt. ratio under traffic change'
#         }
#     ]
    
#     # Flow types and their colors
#     flow_types = [
#         {'id': 0, 'name': 'type1', 'color': 'blue', 'label': 'safe-type1'},
#         {'id': 1, 'name': 'type2', 'color': 'green', 'label': 'safe-type2'},
#         {'id': 2, 'name': 'type3', 'color': 'red', 'label': 'safe-type3'},
#         # {'id': 3, 'name': 'type4', 'color': 'orange', 'label': 'safe-type4'}  # Uncomment if you have type4
#     ]
    
#     # Process each scenario
#     for scenario in scenarios:
#         scenario_dir = os.path.join(log_dir, scenario['name'])
#         col = scenario['col']
        
#         # Get axes for this scenario
#         ax_lat = axes[0, col]  # Latency plot (top row)
#         ax_thr = axes[1, col]  # Throughput plot (bottom row)
        
#         print(f"\nProcessing scenario: {scenario['name']}")
        
#         # Plot each flow type
#         for flow_type in flow_types:
#             # ========== LATENCY ==========
#             delay_file = os.path.join(scenario_dir, f"delay_type{flow_type['id']}.log")
#             delays = read_log_file(delay_file)
            
#             if delays:
#                 # Smooth the data
#                 delays_smooth = smooth_curve(delays, window=window)
                
#                 # Create timestep axis (in thousands)
#                 timesteps = np.arange(len(delays_smooth)) / 1000.0
                
#                 # Limit to max_steps
#                 max_idx = min(len(delays_smooth), scenario['max_steps'])
                
#                 # Plot
#                 ax_lat.plot(
#                     timesteps[:max_idx], 
#                     delays_smooth[:max_idx],
#                     label=flow_type['label'],
#                     color=flow_type['color'],
#                     linewidth=1.5,
#                     alpha=0.8
#                 )
                
#                 print(f"  {flow_type['name']}: {len(delays)} delay samples, "
#                       f"avg={np.mean(delays):.2f}ms")
            
#             # ========== THROUGHPUT ==========
#             throughput_file = os.path.join(scenario_dir, f"throughput_type{flow_type['id']}.log")
#             throughputs = read_log_file(throughput_file)
            
#             if throughputs:
#                 # Smooth the data
#                 throughputs_smooth = smooth_curve(throughputs, window=window)
                
#                 # Create timestep axis (in thousands)
#                 timesteps = np.arange(len(throughputs_smooth)) / 1000.0
                
#                 # Limit to max_steps
#                 max_idx = min(len(throughputs_smooth), scenario['max_steps'])
                
#                 # Plot
#                 ax_thr.plot(
#                     timesteps[:max_idx], 
#                     throughputs_smooth[:max_idx],
#                     label=flow_type['label'],
#                     color=flow_type['color'],
#                     linewidth=1.5,
#                     alpha=0.8
#                 )
                
#                 print(f"  {flow_type['name']}: {len(throughputs)} throughput samples, "
#                       f"avg={np.mean(throughputs):.4f}")
        
#         # ========== FORMAT LATENCY PLOT ==========
#         ax_lat.set_xlabel('Timeslot (10³)', fontsize=11)
#         ax_lat.set_ylabel('Latency (ms)', fontsize=11)
#         ax_lat.set_title(scenario['title_lat'], fontsize=12, fontweight='bold')
#         ax_lat.legend(fontsize=9, loc='best')
#         ax_lat.grid(True, alpha=0.3, linestyle='--')
#         ax_lat.set_xlim(left=0)
        
#         # ========== FORMAT THROUGHPUT PLOT ==========
#         ax_thr.set_xlabel('Timeslot (10³)', fontsize=11)
#         ax_thr.set_ylabel('Throughput Ratio', fontsize=11)
#         ax_thr.set_title(scenario['title_thr'], fontsize=12, fontweight='bold')
#         ax_thr.legend(fontsize=9, loc='best')
#         ax_thr.grid(True, alpha=0.3, linestyle='--')
#         ax_thr.set_xlim(left=0)
#         ax_thr.set_ylim([0.95, 1.02])  # Typical range for good performance
    
#     # Overall title
#     fig.suptitle('Figure 5: Results of MAPPO on flow latency and throughput ratio under '
#                  'initialization, link failure and traffic change in Abilene',
#                  fontsize=14, fontweight='bold', y=0.995)
    
#     # Adjust layout
#     plt.tight_layout(rect=[0, 0, 1, 0.99])
    
#     # Save figure
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"\n✓ Figure saved to: {output_file}")
    
#     # Show figure
#     plt.show()
    
#     return fig


# def print_statistics(log_dir='./log'):
#     """Print summary statistics for all scenarios"""
    
#     print("\n" + "="*80)
#     print("SUMMARY STATISTICS")
#     print("="*80)
    
#     scenarios = ['initialization', 'link_failure', 'traffic_change']
#     flow_types = [0, 1, 2, 3]
    
#     for scenario in scenarios:
#         print(f"\n{scenario.upper().replace('_', ' ')}:")
#         print("-" * 80)
        
#         scenario_dir = os.path.join(log_dir, scenario)
        
#         # Print statistics for each flow type
#         for flow_type in flow_types:
#             delay_file = os.path.join(scenario_dir, f"delay_type{flow_type}.log")
#             throughput_file = os.path.join(scenario_dir, f"throughput_type{flow_type}.log")
#             loss_file = os.path.join(scenario_dir, f"loss_type{flow_type}.log")
            
#             delays = read_log_file(delay_file)
#             throughputs = read_log_file(throughput_file)
#             losses = read_log_file(loss_file)
            
#             if delays:
#                 print(f"  Type {flow_type}: "
#                       f"Delay={np.mean(delays):6.2f}±{np.std(delays):5.2f}ms, "
#                       f"Throughput={np.mean(throughputs):.4f}, "
#                       f"Loss={np.mean(losses):.4f}")
        
#         # Global reward
#         globalrwd_file = os.path.join(scenario_dir, "globalrwd.log")
#         rewards = read_log_file(globalrwd_file)
#         if rewards:
#             print(f"  Global Reward: {np.mean(rewards):.4f}±{np.std(rewards):.4f}")
        
#         # Circle flag (unsafe routes)
#         circle_file = os.path.join(scenario_dir, "circle.log")
#         circles = read_log_file(circle_file)
#         if circles:
#             unsafe_rate = np.mean(circles) * 100
#             print(f"  Unsafe Route Rate: {unsafe_rate:.2f}%")


# def main():
#     parser = argparse.ArgumentParser(
#         description='Plot Figure 5 from DRL-OR paper (MAPPO version)'
#     )
#     parser.add_argument(
#         '--log-dir', 
#         type=str, 
#         default='./log',
#         help='Directory containing log folders (default: ./log)'
#     )
#     parser.add_argument(
#         '--output', 
#         type=str, 
#         default='figure5_mappo_reproduction.png',
#         help='Output filename for the plot (default: figure5_mappo_reproduction.png)'
#     )
#     parser.add_argument(
#         '--window', 
#         type=int, 
#         default=1000,
#         help='Smoothing window size (default: 1000)'
#     )
#     parser.add_argument(
#         '--stats', 
#         action='store_true',
#         help='Print statistics without plotting'
#     )
    
#     args = parser.parse_args()
    
#     # Check if log directory exists
#     if not os.path.exists(args.log_dir):
#         print(f"Error: Log directory '{args.log_dir}' not found!")
#         print(f"Expected structure:")
#         print(f"  {args.log_dir}/")
#         print(f"  ├── initialization/")
#         print(f"  ├── link_failure/")
#         print(f"  └── traffic_change/")
#         return
    
#     # Check for required scenarios
#     scenarios = ['initialization', 'link_failure', 'traffic_change']
#     missing = []
#     for scenario in scenarios:
#         scenario_path = os.path.join(args.log_dir, scenario)
#         if not os.path.exists(scenario_path):
#             missing.append(scenario)
    
#     if missing:
#         print(f"Warning: Missing scenario directories: {', '.join(missing)}")
#         print(f"Will plot only available scenarios.")
    
#     # Print statistics if requested
#     if args.stats:
#         print_statistics(args.log_dir)
#         return
    
#     # Create the plot
#     print(f"Creating Figure 5 from logs in: {args.log_dir}")
#     print(f"Smoothing window: {args.window}")
#     print(f"Output file: {args.output}")
    
#     plot_figure5(
#         log_dir=args.log_dir,
#         output_file=args.output,
#         window=args.window
#     )
    
#     # Print statistics
#     print_statistics(args.log_dir)


# if __name__ == "__main__":
#     main()




















"""
Plot Figure 5 from DRL-OR Paper - PPO and MAPPO Comparison
Reproduces the 6 subplots showing latency and throughput ratio
for three scenarios: initialization, link failure, and traffic change

Now includes BOTH PPO (original) and MAPPO results for comparison.

Usage:
    python3 plot_figure5_comparison.py --log-dir ./log --output fig5_ppo_mappo.png
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


def plot_figure5_comparison(log_dir='./log', output_file='fig5_ppo_mappo_comparison.png', window=1000):
    """
    Create Figure 5 with 6 subplots (2 rows × 3 columns)
    Comparing PPO vs MAPPO
    
    Row 1: Latency for (a) initialization, (b) link failure, (c) traffic change
    Row 2: Throughput ratio for (d) initialization, (e) link failure, (f) traffic change
    """
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Scenario configurations
    scenarios = [
        {
            'ppo_dir': 'initialization',
            'mappo_dir': 'mappo_initialization',
            'col': 0,
            'title_lat': '(a) Latency under initialization',
            'title_thr': '(d) Thrpt. ratio under initialization'
        },
        {
            'ppo_dir': 'link_failure',
            'mappo_dir': 'mappo_link_failure',
            'col': 1,
            'title_lat': '(b) Latency under link failure',
            'title_thr': '(e) Thrpt. ratio under link failure'
        },
        {
            'ppo_dir': 'traffic_change',
            'mappo_dir': 'mappo_traffic_change',
            'col': 2,
            'title_lat': '(c) Latency under traffic change',
            'title_thr': '(f) Thrpt. ratio under traffic change'
        }
    ]
    
    # Flow types configuration - PPO uses solid lines, MAPPO uses dashed
    flow_types = [
        {'id': 0, 'name': 'type1', 'ppo_color': '#1f77b4', 'mappo_color': '#aec7e8',
         'ppo_label': 'PPO-type1', 'mappo_label': 'MAPPO-type1'},
        {'id': 1, 'name': 'type2', 'ppo_color': '#2ca02c', 'mappo_color': '#98df8a',
         'ppo_label': 'PPO-type2', 'mappo_label': 'MAPPO-type2'},
        {'id': 2, 'name': 'type3', 'ppo_color': '#d62728', 'mappo_color': '#ff9896',
         'ppo_label': 'PPO-type3', 'mappo_label': 'MAPPO-type3'},
    ]
    
    print("\n" + "="*80)
    print("PLOTTING FIGURE 5: PPO vs MAPPO Comparison")
    print("="*80)
    
    # Process each scenario
    for scenario in scenarios:
        ppo_path = os.path.join(log_dir, scenario['ppo_dir'])
        mappo_path = os.path.join(log_dir, scenario['mappo_dir'])
        col = scenario['col']
        
        ax_lat = axes[0, col]  # Latency plot (top row)
        ax_thr = axes[1, col]  # Throughput plot (bottom row)
        
        print(f"\nProcessing: {scenario['ppo_dir']} vs {scenario['mappo_dir']}")
        
        # Plot each flow type
        for flow_type in flow_types:
            # ========== LATENCY ==========
            # PPO
            ppo_delay_file = os.path.join(ppo_path, f"delay_type{flow_type['id']}.log")
            ppo_delays = read_log_file(ppo_delay_file)
            
            if ppo_delays:
                ppo_smooth = smooth_curve(ppo_delays, window=window)
                timesteps = np.arange(len(ppo_smooth)) / 1000.0
                ax_lat.plot(timesteps, ppo_smooth, label=flow_type['ppo_label'],
                           color=flow_type['ppo_color'], linewidth=2, alpha=0.9)
                print(f"  PPO {flow_type['name']}: delay avg={np.mean(ppo_delays):.2f}ms")
            
            # MAPPO
            mappo_delay_file = os.path.join(mappo_path, f"delay_type{flow_type['id']}.log")
            mappo_delays = read_log_file(mappo_delay_file)
            
            if mappo_delays:
                mappo_smooth = smooth_curve(mappo_delays, window=window)
                timesteps = np.arange(len(mappo_smooth)) / 1000.0
                ax_lat.plot(timesteps, mappo_smooth, label=flow_type['mappo_label'],
                           color=flow_type['mappo_color'], linewidth=2, linestyle='--', alpha=0.9)
                print(f"  MAPPO {flow_type['name']}: delay avg={np.mean(mappo_delays):.2f}ms")
            
            # ========== THROUGHPUT ==========
            # PPO
            ppo_thr_file = os.path.join(ppo_path, f"throughput_type{flow_type['id']}.log")
            ppo_throughput = read_log_file(ppo_thr_file)
            
            if ppo_throughput:
                ppo_smooth = smooth_curve(ppo_throughput, window=window)
                timesteps = np.arange(len(ppo_smooth)) / 1000.0
                ax_thr.plot(timesteps, ppo_smooth, label=flow_type['ppo_label'],
                           color=flow_type['ppo_color'], linewidth=2, alpha=0.9)
                print(f"  PPO {flow_type['name']}: throughput avg={np.mean(ppo_throughput):.4f}")
            
            # MAPPO
            mappo_thr_file = os.path.join(mappo_path, f"throughput_type{flow_type['id']}.log")
            mappo_throughput = read_log_file(mappo_thr_file)
            
            if mappo_throughput:
                mappo_smooth = smooth_curve(mappo_throughput, window=window)
                timesteps = np.arange(len(mappo_smooth)) / 1000.0
                ax_thr.plot(timesteps, mappo_smooth, label=flow_type['mappo_label'],
                           color=flow_type['mappo_color'], linewidth=2, linestyle='--', alpha=0.9)
                print(f"  MAPPO {flow_type['name']}: throughput avg={np.mean(mappo_throughput):.4f}")
        
        # Format latency plot
        ax_lat.set_xlabel('Timeslot (×10³)', fontsize=11)
        ax_lat.set_ylabel('Latency (ms)', fontsize=11)
        ax_lat.set_title(scenario['title_lat'], fontsize=12, fontweight='bold')
        ax_lat.legend(fontsize=8, loc='upper right', ncol=2)
        ax_lat.grid(True, alpha=0.3, linestyle='--')
        ax_lat.set_xlim(left=0)
        
        # Format throughput plot
        ax_thr.set_xlabel('Timeslot (×10³)', fontsize=11)
        ax_thr.set_ylabel('Throughput Ratio', fontsize=11)
        ax_thr.set_title(scenario['title_thr'], fontsize=12, fontweight='bold')
        ax_thr.legend(fontsize=8, loc='lower right', ncol=2)
        ax_thr.grid(True, alpha=0.3, linestyle='--')
        ax_thr.set_xlim(left=0)
        ax_thr.set_ylim([0.90, 1.02])
    
    # Overall title
    fig.suptitle('Figure 5: PPO vs MAPPO-CTDE Comparison\nLatency and Throughput under Initialization, Link Failure and Traffic Change (Abilene)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_file}")
    
    plt.close(fig)
    return fig


def plot_figure5_separate(log_dir='./log', output_prefix='fig5', window=1000):
    """
    Create separate Figure 5 plots for PPO and MAPPO (like original paper style)
    """
    
    algorithms = [
        {'name': 'PPO', 'dirs': ['initialization', 'link_failure', 'traffic_change'],
         'output': f'{output_prefix}_ppo.png'},
        {'name': 'MAPPO', 'dirs': ['mappo_initialization', 'mappo_link_failure', 'mappo_traffic_change'],
         'output': f'{output_prefix}_mappo.png'},
    ]
    
    for algo in algorithms:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        scenarios = [
            {'dir': algo['dirs'][0], 'col': 0,
             'title_lat': '(a) Latency under initialization',
             'title_thr': '(d) Thrpt. ratio under initialization'},
            {'dir': algo['dirs'][1], 'col': 1,
             'title_lat': '(b) Latency under link failure',
             'title_thr': '(e) Thrpt. ratio under link failure'},
            {'dir': algo['dirs'][2], 'col': 2,
             'title_lat': '(c) Latency under traffic change',
             'title_thr': '(f) Thrpt. ratio under traffic change'},
        ]
        
        flow_types = [
            {'id': 0, 'color': '#1f77b4', 'label': 'safe-type1'},
            {'id': 1, 'color': '#2ca02c', 'label': 'safe-type2'},
            {'id': 2, 'color': '#d62728', 'label': 'safe-type3'},
        ]
        
        print(f"\n{'='*60}")
        print(f"Plotting {algo['name']} results...")
        print(f"{'='*60}")
        
        for scenario in scenarios:
            scenario_path = os.path.join(log_dir, scenario['dir'])
            col = scenario['col']
            
            ax_lat = axes[0, col]
            ax_thr = axes[1, col]
            
            print(f"\n  {scenario['dir']}:")
            
            for flow_type in flow_types:
                # Latency
                delay_file = os.path.join(scenario_path, f"delay_type{flow_type['id']}.log")
                delays = read_log_file(delay_file)
                
                if delays:
                    delays_smooth = smooth_curve(delays, window=window)
                    timesteps = np.arange(len(delays_smooth)) / 1000.0
                    ax_lat.plot(timesteps, delays_smooth, label=flow_type['label'],
                               color=flow_type['color'], linewidth=1.5, alpha=0.8)
                    print(f"    {flow_type['label']}: delay={np.mean(delays):.2f}ms")
                
                # Throughput
                thr_file = os.path.join(scenario_path, f"throughput_type{flow_type['id']}.log")
                throughputs = read_log_file(thr_file)
                
                if throughputs:
                    thr_smooth = smooth_curve(throughputs, window=window)
                    timesteps = np.arange(len(thr_smooth)) / 1000.0
                    ax_thr.plot(timesteps, thr_smooth, label=flow_type['label'],
                               color=flow_type['color'], linewidth=1.5, alpha=0.8)
                    print(f"    {flow_type['label']}: throughput={np.mean(throughputs):.4f}")
            
            # Format plots
            ax_lat.set_xlabel('Timeslot (×10³)', fontsize=11)
            ax_lat.set_ylabel('Latency (ms)', fontsize=11)
            ax_lat.set_title(scenario['title_lat'], fontsize=12, fontweight='bold')
            ax_lat.legend(fontsize=9, loc='upper right')
            ax_lat.grid(True, alpha=0.3, linestyle='--')
            ax_lat.set_xlim(left=0)
            
            ax_thr.set_xlabel('Timeslot (×10³)', fontsize=11)
            ax_thr.set_ylabel('Throughput Ratio', fontsize=11)
            ax_thr.set_title(scenario['title_thr'], fontsize=12, fontweight='bold')
            ax_thr.legend(fontsize=9, loc='lower right')
            ax_thr.grid(True, alpha=0.3, linestyle='--')
            ax_thr.set_xlim(left=0)
            ax_thr.set_ylim([0.90, 1.02])
        
        fig.suptitle(f'Figure 5: {algo["name"]} Results on Flow Latency and Throughput Ratio\nunder Initialization, Link Failure and Traffic Change (Abilene)',
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(algo['output'], dpi=300, bbox_inches='tight')
        print(f"\n✓ {algo['name']} figure saved to: {algo['output']}")
        plt.close(fig)


def print_statistics(log_dir='./log'):
    """Print summary statistics for all scenarios - PPO vs MAPPO"""
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS: PPO vs MAPPO")
    print("="*100)
    
    comparisons = [
        ('initialization', 'mappo_initialization', 'INITIALIZATION'),
        ('link_failure', 'mappo_link_failure', 'LINK FAILURE'),
        ('traffic_change', 'mappo_traffic_change', 'TRAFFIC CHANGE'),
    ]
    
    flow_types = [0, 1, 2, 3]
    
    for ppo_name, mappo_name, title in comparisons:
        print(f"\n{title}")
        print("-" * 100)
        print(f"{'Metric':<20} {'Type':<8} {'PPO':<20} {'MAPPO':<20} {'Diff':<15} {'Winner':<10}")
        print("-" * 100)
        
        ppo_dir = os.path.join(log_dir, ppo_name)
        mappo_dir = os.path.join(log_dir, mappo_name)
        
        # Delay comparison
        for ft in flow_types:
            ppo_delay = read_log_file(os.path.join(ppo_dir, f"delay_type{ft}.log"))
            mappo_delay = read_log_file(os.path.join(mappo_dir, f"delay_type{ft}.log"))
            
            if ppo_delay and mappo_delay:
                ppo_avg = np.mean(ppo_delay)
                mappo_avg = np.mean(mappo_delay)
                diff = ppo_avg - mappo_avg
                winner = "MAPPO" if diff > 0 else "PPO"
                print(f"{'Delay (ms)':<20} {ft:<8} {ppo_avg:<20.2f} {mappo_avg:<20.2f} {diff:+.2f}ms{'':<6} {winner:<10}")
        
        # Throughput comparison
        for ft in flow_types:
            ppo_thr = read_log_file(os.path.join(ppo_dir, f"throughput_type{ft}.log"))
            mappo_thr = read_log_file(os.path.join(mappo_dir, f"throughput_type{ft}.log"))
            
            if ppo_thr and mappo_thr:
                ppo_avg = np.mean(ppo_thr)
                mappo_avg = np.mean(mappo_thr)
                diff = mappo_avg - ppo_avg
                winner = "MAPPO" if diff > 0 else "PPO"
                print(f"{'Throughput':<20} {ft:<8} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {diff:+.4f}{'':<6} {winner:<10}")
        
        # Global reward
        ppo_rwd = read_log_file(os.path.join(ppo_dir, "globalrwd.log"))
        mappo_rwd = read_log_file(os.path.join(mappo_dir, "globalrwd.log"))
        
        if ppo_rwd and mappo_rwd:
            ppo_avg = np.mean(ppo_rwd)
            mappo_avg = np.mean(mappo_rwd)
            diff = mappo_avg - ppo_avg
            winner = "MAPPO" if diff > 0 else "PPO"
            print(f"{'Global Reward':<20} {'All':<8} {ppo_avg:<20.4f} {mappo_avg:<20.4f} {diff:+.4f}{'':<6} {winner:<10}")
        
        # Unsafe rate
        ppo_circle = read_log_file(os.path.join(ppo_dir, "circle.log"))
        mappo_circle = read_log_file(os.path.join(mappo_dir, "circle.log"))
        
        if ppo_circle and mappo_circle:
            ppo_rate = np.mean(ppo_circle) * 100
            mappo_rate = np.mean(mappo_circle) * 100
            diff = ppo_rate - mappo_rate
            winner = "MAPPO" if diff > 0 else "PPO"
            print(f"{'Unsafe Rate (%)':<20} {'All':<8} {ppo_rate:<20.2f} {mappo_rate:<20.2f} {diff:+.2f}%{'':<6} {winner:<10}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Figure 5 comparing PPO vs MAPPO'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='./log',
        help='Directory containing log folders (default: ./log)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='fig5_ppo_mappo_comparison.png',
        help='Output filename for comparison plot'
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=1000,
        help='Smoothing window size (default: 1000)'
    )
    parser.add_argument(
        '--separate', 
        action='store_true',
        help='Generate separate figures for PPO and MAPPO (paper style)'
    )
    parser.add_argument(
        '--stats', 
        action='store_true',
        help='Print statistics only'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' not found!")
        return
    
    print(f"\nLog directory: {args.log_dir}")
    
    if args.stats:
        print_statistics(args.log_dir)
        return
    
    if args.separate:
        plot_figure5_separate(args.log_dir, output_prefix='fig5', window=args.window)
    else:
        plot_figure5_comparison(args.log_dir, args.output, args.window)
    
    print_statistics(args.log_dir)


if __name__ == "__main__":
    main()