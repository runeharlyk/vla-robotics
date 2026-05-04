import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_paths, out_dir):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."

    dfs = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: Could not find file {csv_path}. Skipping.")
            continue
        dfs.append(pd.read_csv(csv_path))
    
    if len(dfs) == 0:
        print("Error: No valid CSV files found.")
        return
    
    df = pd.concat(dfs, ignore_index=True)

    # Plot 1: Severity vs L2 Distance & Relative L2 Distance
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.lineplot(data=df, x='noise_severity', y='l2_distance', hue='noise_type', marker='o', ax=axes[0])
    axes[0].set_title('Severity vs L2 Distance')
    axes[0].set_xlabel('Noise Severity')
    axes[0].set_ylabel('L2 Distance')

    sns.lineplot(data=df, x='noise_severity', y='rel_l2_distance', hue='noise_type', marker='o', ax=axes[1])
    axes[1].set_title('Severity vs Relative L2 Distance')
    axes[1].set_xlabel('Noise Severity')
    axes[1].set_ylabel('Relative L2 Distance')

    plt.tight_layout()
    severity_plot_path = os.path.join(out_dir, "severity_distance_plot.png")
    plt.savefig(severity_plot_path)
    print(f"Saved severity distance plot to '{severity_plot_path}'")
    # plt.show()

    # Plot 2: Individual Dimensions grouped by Noise Type
    dimensions = ['abs_err_x', 'abs_err_y', 'abs_err_z', 'abs_err_roll','abs_err_pitch','abs_err_yaw','abs_err_gripper']
    
    df_melt = df[['noise_type', 'noise_severity'] + dimensions].copy()
    df_melt = df_melt.melt(id_vars=['noise_type', 'noise_severity'], 
                           value_vars=dimensions, 
                           var_name='dimension', 
                           value_name='absolute_error')
    
    dim_name_mapping = {
        'abs_err_x': 'X',
        'abs_err_y': 'Y',
        'abs_err_z': 'Z',
        'abs_err_roll': 'Roll',
        'abs_err_pitch': 'Pitch',
        'abs_err_yaw': 'Yaw',
        'abs_err_gripper': 'Gripper'
    }
    df_melt['dimension'] = df_melt['dimension'].map(dim_name_mapping)
 
    g = sns.catplot(
        data=df_melt,
        kind="bar",
        x="dimension",
        y="absolute_error",
        col="noise_type",
        col_wrap=3, 
        height=4, 
        aspect=1.5,
        sharey=False 
    )
    
    g.set_axis_labels("Action Dimension", "Mean Absolute Error")
    g.set_titles("Noise Type: {col_name}")
    
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', labelbottom=True, rotation=45)
        
    plt.tight_layout()
    dims_plot_path = os.path.join(out_dir, "dimension_errors_by_noise.png")
    plt.savefig(dims_plot_path)
    print(f"Saved dimensional error plot to '{dims_plot_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for smolvla visual pilot experiment results.")
    parser.add_argument("--csv", type=str, nargs='+', required=True, help="One or more paths to results.csv files")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the generated plots")
    
    args = parser.parse_args()
    plot_results(args.csv, args.out_dir)
