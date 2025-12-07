"""
Diagnostic Script for MAPPO Training Analysis
Helps identify why MAPPO might underperform in certain scenarios
"""

import numpy as np
import os
import argparse


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


def analyze_mappo_training(log_dir, scenario_name):
    """Analyze MAPPO training metrics for a specific scenario"""
    
    scenario_dir = os.path.join(log_dir, scenario_name)
    
    print(f"\n{'='*80}")
    print(f"MAPPO TRAINING DIAGNOSTICS: {scenario_name}")
    print(f"{'='*80}")
    
    # 1. Importance Ratio Analysis
    print("\n[1] IMPORTANCE RATIO ANALYSIS")
    print("-" * 60)
    
    ratio = read_log_file(os.path.join(scenario_dir, "ratio.log"))
    if ratio:
        ratio_array = np.array(ratio)
        
        # Check for problematic ratios
        too_high = np.sum(ratio_array > 1.5) / len(ratio_array) * 100
        too_low = np.sum(ratio_array < 0.5) / len(ratio_array) * 100
        normal = np.sum((ratio_array >= 0.8) & (ratio_array <= 1.2)) / len(ratio_array) * 100
        
        print(f"  Total samples: {len(ratio)}")
        print(f"  Mean ratio: {np.mean(ratio):.4f} (should be ~1.0)")
        print(f"  Std ratio: {np.std(ratio):.4f} (should be small)")
        print(f"  Min/Max: {np.min(ratio):.4f} / {np.max(ratio):.4f}")
        print(f"  ")
        print(f"  Distribution:")
        print(f"    Normal [0.8-1.2]: {normal:.1f}%")
        print(f"    Too high (>1.5): {too_high:.1f}%")
        print(f"    Too low (<0.5): {too_low:.1f}%")
        
        if np.mean(ratio) > 1.5 or np.mean(ratio) < 0.7:
            print(f"\n  ⚠️  WARNING: Mean ratio is far from 1.0!")
            print(f"     This indicates PPO clipping may not be working correctly.")
            print(f"     Policy is changing too aggressively between updates.")
        
        if too_high > 10:
            print(f"\n  ⚠️  WARNING: {too_high:.1f}% of ratios are >1.5!")
            print(f"     This causes training instability.")
    else:
        print("  ❌ ratio.log not found")
    
    # 2. Value Loss Analysis
    print("\n[2] VALUE LOSS ANALYSIS")
    print("-" * 60)
    
    value_loss = read_log_file(os.path.join(scenario_dir, "value_loss.log"))
    if value_loss:
        vl_array = np.array(value_loss)
        
        # Check convergence
        first_quarter = np.mean(vl_array[:len(vl_array)//4])
        last_quarter = np.mean(vl_array[-len(vl_array)//4:])
        reduction = (first_quarter - last_quarter) / first_quarter * 100
        
        print(f"  Total updates: {len(value_loss)}")
        print(f"  Mean value loss: {np.mean(value_loss):.4f}")
        print(f"  First quarter avg: {first_quarter:.4f}")
        print(f"  Last quarter avg: {last_quarter:.4f}")
        print(f"  Reduction: {reduction:.1f}%")
        
        if last_quarter > 50:
            print(f"\n  ⚠️  WARNING: Value loss is still high ({last_quarter:.1f})!")
            print(f"     The centralized critic is not learning well.")
            print(f"     Consider:")
            print(f"       - Lower critic learning rate")
            print(f"       - More training steps")
            print(f"       - Check global state construction")
        
        if reduction < 20:
            print(f"\n  ⚠️  WARNING: Value loss reduction is only {reduction:.1f}%!")
            print(f"     Critic is not converging properly.")
    else:
        print("  ❌ value_loss.log not found")
    
    # 3. Action Loss Analysis
    print("\n[3] ACTION LOSS (POLICY LOSS) ANALYSIS")
    print("-" * 60)
    
    action_loss = read_log_file(os.path.join(scenario_dir, "action_loss.log"))
    if action_loss:
        al_array = np.array(action_loss)
        
        print(f"  Total updates: {len(action_loss)}")
        print(f"  Mean action loss: {np.mean(action_loss):.6f}")
        print(f"  Std: {np.std(action_loss):.6f}")
        print(f"  Min/Max: {np.min(action_loss):.6f} / {np.max(action_loss):.6f}")
        
        # Check for oscillation
        changes = np.diff(al_array)
        sign_changes = np.sum(np.diff(np.sign(changes)) != 0)
        oscillation_rate = sign_changes / len(changes) * 100
        
        print(f"  Oscillation rate: {oscillation_rate:.1f}%")
        
        if oscillation_rate > 80:
            print(f"\n  ⚠️  WARNING: High oscillation ({oscillation_rate:.1f}%)!")
            print(f"     Policy is unstable. Try lower learning rate.")
    else:
        print("  ❌ action_loss.log not found")
    
    # 4. Unsafe Route Analysis
    print("\n[4] UNSAFE ROUTE (CIRCLE) ANALYSIS")
    print("-" * 60)
    
    circle = read_log_file(os.path.join(scenario_dir, "circle.log"))
    if circle:
        circle_array = np.array(circle)
        
        # Analyze by phases
        total_samples = len(circle)
        phase_size = total_samples // 4
        
        phases = [
            ("Early (0-25%)", circle_array[:phase_size]),
            ("Mid-Early (25-50%)", circle_array[phase_size:2*phase_size]),
            ("Mid-Late (50-75%)", circle_array[2*phase_size:3*phase_size]),
            ("Late (75-100%)", circle_array[3*phase_size:]),
        ]
        
        print(f"  Total samples: {total_samples}")
        print(f"  Overall unsafe rate: {np.mean(circle)*100:.2f}%")
        print(f"\n  Unsafe rate by training phase:")
        
        for phase_name, phase_data in phases:
            rate = np.mean(phase_data) * 100
            print(f"    {phase_name}: {rate:.2f}%")
        
        # Check if improving
        early_rate = np.mean(circle_array[:phase_size])
        late_rate = np.mean(circle_array[-phase_size:])
        
        if late_rate < early_rate:
            improvement = (early_rate - late_rate) / early_rate * 100
            print(f"\n  ✅ Unsafe rate IMPROVING: {improvement:.1f}% reduction")
        else:
            degradation = (late_rate - early_rate) / early_rate * 100 if early_rate > 0 else 0
            print(f"\n  ⚠️  WARNING: Unsafe rate NOT improving (or getting worse)")
            print(f"     Early: {early_rate*100:.2f}% → Late: {late_rate*100:.2f}%")
    else:
        print("  ❌ circle.log not found")
    
    # 5. Global Reward Analysis
    print("\n[5] GLOBAL REWARD ANALYSIS")
    print("-" * 60)
    
    reward = read_log_file(os.path.join(scenario_dir, "globalrwd.log"))
    if reward:
        reward_array = np.array(reward)
        
        phase_size = len(reward) // 4
        early_reward = np.mean(reward_array[:phase_size])
        late_reward = np.mean(reward_array[-phase_size:])
        
        print(f"  Total samples: {len(reward)}")
        print(f"  Mean reward: {np.mean(reward):.4f}")
        print(f"  Early phase avg: {early_reward:.4f}")
        print(f"  Late phase avg: {late_reward:.4f}")
        
        if late_reward > early_reward:
            print(f"\n  ✅ Reward IMPROVING over training")
        else:
            print(f"\n  ⚠️  WARNING: Reward NOT improving")
    else:
        print("  ❌ globalrwd.log not found")
    
    # 6. Summary & Recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    
    if ratio and np.mean(ratio) > 1.5:
        issues.append("High importance ratio - clipping not effective")
    
    if value_loss and np.mean(value_loss[-len(value_loss)//4:]) > 50:
        issues.append("High value loss - critic not converging")
    
    if circle and np.mean(circle) > 0.1:
        issues.append("High unsafe rate (>10%)")
    
    if reward and np.mean(reward[-len(reward)//4:]) < np.mean(reward[:len(reward)//4]):
        issues.append("Reward not improving")
    
    if issues:
        print("\n  ❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
        
        print("\n  📋 Recommended fixes:")
        print("     1. Lower learning rates: --actor-lr 3e-4 --critic-lr 1e-4")
        print("     2. Reduce PPO epochs: --ppo-epoch 10")
        print("     3. Add gradient clipping: --max-grad-norm 0.5")
        print("     4. Pre-train critic separately before online training")
        print("     5. Use value normalization: --use-popart")
    else:
        print("\n  ✅ No major issues detected!")


def compare_scenarios(log_dir):
    """Compare MAPPO performance across all scenarios"""
    
    scenarios = [
        ('mappo_initialization', 'initialization'),
        ('mappo_link_failure', 'link_failure'),
        ('mappo_traffic_change', 'traffic_change'),
    ]
    
    print("\n" + "="*100)
    print("CROSS-SCENARIO COMPARISON")
    print("="*100)
    
    print(f"\n{'Scenario':<25} {'Ratio':<12} {'Value Loss':<15} {'Unsafe %':<12} {'Reward':<12}")
    print("-"*100)
    
    for mappo_name, ppo_name in scenarios:
        mappo_dir = os.path.join(log_dir, mappo_name)
        
        ratio = read_log_file(os.path.join(mappo_dir, "ratio.log"))
        value_loss = read_log_file(os.path.join(mappo_dir, "value_loss.log"))
        circle = read_log_file(os.path.join(mappo_dir, "circle.log"))
        reward = read_log_file(os.path.join(mappo_dir, "globalrwd.log"))
        
        ratio_str = f"{np.mean(ratio):.3f}" if ratio else "N/A"
        vl_str = f"{np.mean(value_loss):.2f}" if value_loss else "N/A"
        circle_str = f"{np.mean(circle)*100:.2f}%" if circle else "N/A"
        reward_str = f"{np.mean(reward):.4f}" if reward else "N/A"
        
        print(f"{mappo_name:<25} {ratio_str:<12} {vl_str:<15} {circle_str:<12} {reward_str:<12}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Diagnose MAPPO training issues')
    parser.add_argument('--log-dir', type=str, default='./log', help='Log directory')
    parser.add_argument('--scenario', type=str, default=None, 
                       help='Specific scenario to analyze (e.g., mappo_initialization)')
    parser.add_argument('--all', action='store_true', help='Analyze all scenarios')
    
    args = parser.parse_args()
    
    if args.all or args.scenario is None:
        # Analyze all MAPPO scenarios
        scenarios = ['mappo_initialization', 'mappo_link_failure', 'mappo_traffic_change']
        
        for scenario in scenarios:
            if os.path.exists(os.path.join(args.log_dir, scenario)):
                analyze_mappo_training(args.log_dir, scenario)
        
        compare_scenarios(args.log_dir)
    else:
        analyze_mappo_training(args.log_dir, args.scenario)


if __name__ == "__main__":
    main()