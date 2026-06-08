import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your hyperparameter optimization logs
df = pd.read_csv('SK_bcm_results_35000_datapoints.csv')

# 2. Extract and print the ultimate best result for Simulation stability
best_simulation = df.loc[df['SIM'].idxmin()]
print("==================================================")
print("     OPTIMAL PERFORMANCE CONFIGURATION FOUND       ")
print("==================================================")
print(f"Past Output 'u' (na):        {int(best_simulation['na'])}")
print(f"Past Input 'th' (nb):        {int(best_simulation['nb'])}")
print(f"# of Experts:                {int(best_simulation['n_experts'])}")
print(f"Simulation RMS Error:        {best_simulation['SIM']:.6f}")
print(f"One-Step Prediction Error:   {best_simulation['PRED']:.6f}")
print(f"Wall-Time Runtime (seconds): {best_simulation['time_taken']:.2f}")
print("==================================================")

# 3. Print top 5 configuration variations for comparison
print("\nTop 5 Architectures ordered by Simulation Accuracy:")
print(df.sort_values(by='SIM').head(5).to_string(index=False))

# 4. Generate the sequential heatmaps (na vs nb) per n_experts value
unique_experts = sorted(df['n_experts'].unique())

for exp in unique_experts:
    # Filter dataset for the specific expert committee tier
    df_subset = df[df['n_experts'] == exp]
    
    # Restructure into matrix format: rows=nb (y-axis), columns=na (x-axis)
    pivot_matrix = df_subset.pivot(index='nb', columns='na', values='SIM')
    
    # Instantiate figure plot
    plt.figure(figsize=(7, 5))
    
    # Plot using a sequential, easy-to-read inverted colormap (lower is darker/better)
    sns.heatmap(
        pivot_matrix, 
        annot=True, 
        fmt=".4f", 
        cmap="rocket_r", 
        cbar_kws={'label': 'Simulation RMS Error'},
        linewidths=0.5, 
        linecolor='#e0e0e0'
    )
    
    plt.title(f'Simulation Error Heatmap Matrix (n_experts = {exp})', fontsize=11, fontweight='bold')
    plt.xlabel('na (Past Output History Windows)', fontsize=10)
    plt.ylabel('nb (Past Control Input History Windows)', fontsize=10)
    
    # Invert the y-axis so small nb lengths sit at the standard origin position
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save each individual matrix view to disk
    plt.savefig(f'heatmap_experts_{exp}.png', dpi=300)
    plt.show()