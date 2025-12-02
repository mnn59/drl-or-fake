"""
Plot Figure 5 from DRL-OR Paper
Reproduces the 6 subplots showing latency and throughput ratio
for three scenarios: initialization, link failure, and traffic change

Usage:
    python3 plot_figure5.py --log-dir ./log --output fig5_reproduction.png
"""

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
        print(f"Warning: {filepath} not found")
        return []
    return values


def smooth_curve(values, window=1000):
    """Apply moving average smoothing"""
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2)
        smoothed.append(np.mean(values[start:end]))
    return smoothed


def plot_figure5(log_dir='./log', output_file='figure5_reproduction.png', window=1000):
    """
    Create Figure 5 with 6 subplots (2 rows × 3 columns)
    
    Row 1: Latency for (a) initialization, (b) link failure, (c) traffic change
    Row 2: Throughput ratio for (d) initialization, (e) link failure, (f) traffic change
    """
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Scenario configurations
    scenarios = [
        {
            'name': 'initialization',
            'max_steps': 300000,
            'col': 0,
            'title_lat': '(a) Latency under initialization',
            'title_thr': '(d) Thrpt. ratio under initialization'
        },
        {
            'name': 'link_failure',
            'max_steps': 180000,
            'col': 1,
            'title_lat': '(b) Latency under link failure',
            'title_thr': '(e) Thrpt. ratio under link failure'
        },
        {
            'name': 'traffic_change',
            'max_steps': 180000,
            'col': 2,
            'title_lat': '(c) Latency under traffic change',
            'title_thr': '(f) Thrpt. ratio under traffic change'
        }
    ]
    
    # Flow types and their colors
    flow_types = [
        {'id': 0, 'name': 'type1', 'color': 'blue', 'label': 'safe-type1'},
        {'id': 1, 'name': 'type2', 'color': 'green', 'label': 'safe-type2'},
        {'id': 2, 'name': 'type3', 'color': 'red', 'label': 'safe-type3'},
        # {'id': 3, 'name': 'type4', 'color': 'orange', 'label': 'safe-type4'}  # Uncomment if you have type4
    ]
    
    # Process each scenario
    for scenario in scenarios:
        scenario_dir = os.path.join(log_dir, scenario['name'])
        col = scenario['col']
        
        # Get axes for this scenario
        ax_lat = axes[0, col]  # Latency plot (top row)
        ax_thr = axes[1, col]  # Throughput plot (bottom row)
        
        print(f"\nProcessing scenario: {scenario['name']}")
        
        # Plot each flow type
        for flow_type in flow_types:
            # ========== LATENCY ==========
            delay_file = os.path.join(scenario_dir, f"delay_type{flow_type['id']}.log")
            delays = read_log_file(delay_file)
            
            if delays:
                # Smooth the data
                delays_smooth = smooth_curve(delays, window=window)
                
                # Create timestep axis (in thousands)
                timesteps = np.arange(len(delays_smooth)) / 1000.0
                
                # Limit to max_steps
                max_idx = min(len(delays_smooth), scenario['max_steps'])
                
                # Plot
                ax_lat.plot(
                    timesteps[:max_idx], 
                    delays_smooth[:max_idx],
                    label=flow_type['label'],
                    color=flow_type['color'],
                    linewidth=1.5,
                    alpha=0.8
                )
                
                print(f"  {flow_type['name']}: {len(delays)} delay samples, "
                      f"avg={np.mean(delays):.2f}ms")
            
            # ========== THROUGHPUT ==========
            throughput_file = os.path.join(scenario_dir, f"throughput_type{flow_type['id']}.log")
            throughputs = read_log_file(throughput_file)
            
            if throughputs:
                # Smooth the data
                throughputs_smooth = smooth_curve(throughputs, window=window)
                
                # Create timestep axis (in thousands)
                timesteps = np.arange(len(throughputs_smooth)) / 1000.0
                
                # Limit to max_steps
                max_idx = min(len(throughputs_smooth), scenario['max_steps'])
                
                # Plot
                ax_thr.plot(
                    timesteps[:max_idx], 
                    throughputs_smooth[:max_idx],
                    label=flow_type['label'],
                    color=flow_type['color'],
                    linewidth=1.5,
                    alpha=0.8
                )
                
                print(f"  {flow_type['name']}: {len(throughputs)} throughput samples, "
                      f"avg={np.mean(throughputs):.4f}")
        
        # ========== FORMAT LATENCY PLOT ==========
        ax_lat.set_xlabel('Timeslot (10³)', fontsize=11)
        ax_lat.set_ylabel('Latency (ms)', fontsize=11)
        ax_lat.set_title(scenario['title_lat'], fontsize=12, fontweight='bold')
        ax_lat.legend(fontsize=9, loc='best')
        ax_lat.grid(True, alpha=0.3, linestyle='--')
        ax_lat.set_xlim(left=0)
        
        # ========== FORMAT THROUGHPUT PLOT ==========
        ax_thr.set_xlabel('Timeslot (10³)', fontsize=11)
        ax_thr.set_ylabel('Throughput Ratio', fontsize=11)
        ax_thr.set_title(scenario['title_thr'], fontsize=12, fontweight='bold')
        ax_thr.legend(fontsize=9, loc='best')
        ax_thr.grid(True, alpha=0.3, linestyle='--')
        ax_thr.set_xlim(left=0)
        ax_thr.set_ylim([0.95, 1.02])  # Typical range for good performance
    
    # Overall title
    fig.suptitle('Figure 5: Results of MAPPO on flow latency and throughput ratio under '
                 'initialization, link failure and traffic change in Abilene',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_file}")
    
    # Show figure
    plt.show()
    
    return fig


def print_statistics(log_dir='./log'):
    """Print summary statistics for all scenarios"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    scenarios = ['initialization', 'link_failure', 'traffic_change']
    flow_types = [0, 1, 2, 3]
    
    for scenario in scenarios:
        print(f"\n{scenario.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        scenario_dir = os.path.join(log_dir, scenario)
        
        # Print statistics for each flow type
        for flow_type in flow_types:
            delay_file = os.path.join(scenario_dir, f"delay_type{flow_type}.log")
            throughput_file = os.path.join(scenario_dir, f"throughput_type{flow_type}.log")
            loss_file = os.path.join(scenario_dir, f"loss_type{flow_type}.log")
            
            delays = read_log_file(delay_file)
            throughputs = read_log_file(throughput_file)
            losses = read_log_file(loss_file)
            
            if delays:
                print(f"  Type {flow_type}: "
                      f"Delay={np.mean(delays):6.2f}±{np.std(delays):5.2f}ms, "
                      f"Throughput={np.mean(throughputs):.4f}, "
                      f"Loss={np.mean(losses):.4f}")
        
        # Global reward
        globalrwd_file = os.path.join(scenario_dir, "globalrwd.log")
        rewards = read_log_file(globalrwd_file)
        if rewards:
            print(f"  Global Reward: {np.mean(rewards):.4f}±{np.std(rewards):.4f}")
        
        # Circle flag (unsafe routes)
        circle_file = os.path.join(scenario_dir, "circle.log")
        circles = read_log_file(circle_file)
        if circles:
            unsafe_rate = np.mean(circles) * 100
            print(f"  Unsafe Route Rate: {unsafe_rate:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Figure 5 from DRL-OR paper (MAPPO version)'
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
        default='figure5_mappo_reproduction.png',
        help='Output filename for the plot (default: figure5_mappo_reproduction.png)'
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=1000,
        help='Smoothing window size (default: 1000)'
    )
    parser.add_argument(
        '--stats', 
        action='store_true',
        help='Print statistics without plotting'
    )
    
    args = parser.parse_args()
    
    # Check if log directory exists
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' not found!")
        print(f"Expected structure:")
        print(f"  {args.log_dir}/")
        print(f"  ├── initialization/")
        print(f"  ├── link_failure/")
        print(f"  └── traffic_change/")
        return
    
    # Check for required scenarios
    scenarios = ['initialization', 'link_failure', 'traffic_change']
    missing = []
    for scenario in scenarios:
        scenario_path = os.path.join(args.log_dir, scenario)
        if not os.path.exists(scenario_path):
            missing.append(scenario)
    
    if missing:
        print(f"Warning: Missing scenario directories: {', '.join(missing)}")
        print(f"Will plot only available scenarios.")
    
    # Print statistics if requested
    if args.stats:
        print_statistics(args.log_dir)
        return
    
    # Create the plot
    print(f"Creating Figure 5 from logs in: {args.log_dir}")
    print(f"Smoothing window: {args.window}")
    print(f"Output file: {args.output}")
    
    plot_figure5(
        log_dir=args.log_dir,
        output_file=args.output,
        window=args.window
    )
    
    # Print statistics
    print_statistics(args.log_dir)


if __name__ == "__main__":
    main()