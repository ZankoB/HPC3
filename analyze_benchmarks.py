import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_benchmark_file(filepath="benchmark.md"):
    """
    Parses the benchmark.md file and returns a list of dictionaries.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if not line.strip().startswith('|') or "---" in line or "Num Particles" in line:
            continue

        parts = [p.strip() for p in line.strip().split('|') if p.strip()]
        
        if len(parts) != 4:
            continue
            
        try:
            num_particles = int(parts[0])
            method = parts[1]
            
            # Extract number from the configuration string
            config_str = parts[2]
            config_val_match = re.search(r'\d+', config_str)
            if not config_val_match:
                continue
            config_val = int(config_val_match.group())

            exec_time = float(parts[3])

            data.append({
                "Particles": num_particles,
                "Method": method,
                "Config": config_val,
                "Time": exec_time
            })
        except (ValueError, IndexError):
            # Skip lines that can't be parsed
            continue
            
    return data

def calculate_speedup(df):
    """
    Calculates the speedup relative to the sequential baseline.
    """
    # Find the sequential baseline times (CPU with 1 core)
    baseline_df = df[(df['Method'] == 'CPU') & (df['Config'] == 1)].set_index('Particles')['Time']
    
    # Create a dictionary for easy lookup: {1000: 17.320, 2000: 65.196, ...}
    ts_map = baseline_df.to_dict()

    # Calculate speedup S = ts / tp
    df['Baseline'] = df['Particles'].map(ts_map)
    df['Speedup'] = df['Baseline'] / df['Time']
    
    return df

def calculate_best_speedup(df):
    """
    Calculates:
    - CPU speedup vs CPU baseline
    - GPU speedup vs CPU baseline
    - GPU speedup vs best CPU
    """

    # 1. Baseline (CPU, 1 core)
    baseline_df = df[(df['Method'] == 'CPU') & (df['Config'] == 1)]
    baseline_map = baseline_df.set_index('Particles')['Time'].to_dict()

    # 2. Best CPU times (idxmin keeps structure intact)
    cpu_df = df[df['Method'] == 'CPU']
    best_cpu = cpu_df.loc[cpu_df.groupby('Particles')['Time'].idxmin()].copy()
    best_cpu = best_cpu.rename(columns={
        'Config': 'CPU_Best_Config',
        'Time': 'CPU_Min_Time'
    })

    # 3. Best GPU times
    gpu_df = df[df['Method'] == 'GPU']
    best_gpu = gpu_df.loc[gpu_df.groupby('Particles')['Time'].idxmin()].copy()
    best_gpu = best_gpu.rename(columns={
        'Config': 'GPU_Best_Config',
        'Time': 'GPU_Min_Time'
    })

    # DEBUG (optional, remove later)
    # print(best_cpu.columns)
    # print(best_gpu.columns)

    # 4. Merge safely
    merged = pd.merge(best_cpu, best_gpu, on='Particles', how='inner')

    # 5. Add baseline
    merged['Baseline_Time'] = merged['Particles'].map(baseline_map)

    # 6. Compute speedups
    merged['CPU_Speedup'] = merged['Baseline_Time'] / merged['CPU_Min_Time']
    merged['GPU_Speedup_vs_Baseline'] = merged['Baseline_Time'] / merged['GPU_Min_Time']
    merged['GPU_Speedup_vs_CPU'] = merged['CPU_Min_Time'] / merged['GPU_Min_Time']

    # 7. Final selection
    return merged[[
        'Particles',
        'CPU_Best_Config', 'CPU_Min_Time', 'CPU_Speedup',
        'GPU_Best_Config', 'GPU_Min_Time',
        'GPU_Speedup_vs_Baseline', 'GPU_Speedup_vs_CPU'
    ]].sort_values(by='Particles')

def plot_results(df):
    """
    Generates subplots while excluding baseline from the main hue 
    to avoid double-plotting and legend clutter.
    """
    sns.set_theme(style="whitegrid")
    
    # Define baseline criteria
    baseline_mask = (df['Method'] == 'CPU') & (df['Config'] == 1)
    baseline_data = df[baseline_mask]
    cpu_best = df[df['Method'] == 'CPU'].groupby('Particles')['Time'].min().reset_index()

    for method in ['CPU', 'GPU']:
        plot_df = df[(df['Method'] == method) & (~baseline_mask)].copy()
        
        config_label = 'Cores' if method == 'CPU' else 'Threads per Block'
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 7))
        
        for ax, scale in zip([ax_lin, ax_log], ['linear', 'log']):

            sns.lineplot(
                data=plot_df,
                x='Particles',
                y='Time',
                hue='Config',
                marker='o',
                palette='tab10',
                ax=ax
            )

            # Baseline (CPU 1 core)
            ax.plot(
                baseline_data['Particles'], 
                baseline_data['Time'], 
                'k--', 
                marker='s',
                label='CPU Baseline (1 Core)'
            )

            if method == 'GPU':
                ax.plot(
                    cpu_best['Particles'],
                    cpu_best['Time'],
                    color='blue',
                    linestyle='-.',
                    marker='D',    
                    linewidth=2,
                    label='Best CPU (Optimized)'
                )

            ax.set_yscale(scale)
            ax.set_title(f'{method} Performance ({scale.capitalize()} Scale)', fontsize=14)
            ax.set_xlabel('Number of Particles')
            ax.set_ylabel('Time (seconds)')
            ax.legend(title=config_label)

        plt.tight_layout()
        plt.savefig(f'{method.lower()}_subplots.png')
        print(f"Saved {method.lower()}_subplots.png")


def main():
    # 1. Parse the data
    benchmark_data = parse_benchmark_file()
    if not benchmark_data:
        print("Could not parse any data from benchmark.md. Please check the file format.")
        return
        
    df = pd.DataFrame(benchmark_data)

    # 2. Calculate speedup
    df = calculate_speedup(df)
    
    # Display the DataFrame with speedup calculations
    print("--- Benchmark Data with Speedup ---")
    print(df.to_string())
    
    # 3. Create plots
    plot_results(df)

    # print(calculate_best_speedup(df))

if __name__ == "__main__":
    main()