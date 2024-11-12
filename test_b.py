import pickle
from tqdm import tqdm  # For displaying a progress bar
from joblib import Parallel, delayed  # For parallel processing
from LikelihoodRatioBiasTest import LikelihoodRatioBiasTest  # Import the Likelihood Ratio Bias Test class
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2  # For chi-squared distribution

# Attempt to set the max CPU count to the number of physical cores (helps with parallel processing)
try:
    import os
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    os.environ['LOKY_MAX_CPU_COUNT'] = str(physical_cores)
except:
    pass

# Load data from the file 'h1.pkl' into a list of datasets
with open('h1.pkl', 'rb') as file:
    data_list = pickle.load(file)

# Optional: Sample a subset of data if needed for testing
# from random import sample
# data_list = sample(data_list, 1000)

# Use joblib to apply the LikelihoodRatioBiasTest in parallel to each dataset in `data_list`
h1_analyses = Parallel(n_jobs=-1)(
    delayed(LikelihoodRatioBiasTest)(dataset['data']) for dataset in tqdm(data_list, desc="Processing datasets")
)

# Calculate the 95th percentile for a chi-squared distribution with 0.5 degrees of freedom
quantile_95 = chi2.ppf(0.95, df=0.5)

# Create a list of dictionaries, each containing bias level, sample size, and likelihood ratio test statistic (LRTS)
data = [{'bias': dataset['bias'], 'size': dataset['size'], 'lrts': analysis.lrts} for dataset, analysis in zip(data_list, h1_analyses)]

# Convert data to a DataFrame for easier manipulation and plotting
df = pd.DataFrame(data)


# Function to calculate the proportion of LRTS values above a given threshold
def proportion_above_threshold(series, threshold):
    return (series > threshold).sum() / len(series)

# Group the DataFrame by 'bias' and 'size', then calculate the proportion of LRTS values above the 95th quantile
grouped = df.groupby(['bias', 'size'])['lrts'].apply(proportion_above_threshold, threshold=quantile_95).reset_index()

# Rename the resulting column to 'Power' for clarity
grouped.rename(columns={'lrts': 'Power'}, inplace=True)

# Set up a plot to display the power curves
plt.figure(figsize=(10, 4))

# Loop through each unique sample size and plot 'bias' vs. 'Power' as separate lines
for size in grouped['size'].unique():
    subset = grouped[grouped['size'] == size]
    plt.plot(subset['bias'], subset['Power'], marker='o', label=f'Number of p-values: {size}')

# Add labels, title, and grid to the plot
plt.xlabel('Bias ("%" of unreported non-significant results)')
plt.ylabel('Power')
plt.title('Power Plot for Different Sample Sizes')
plt.legend()
plt.grid(True)
# Save the power plot in both PNG and SVG formats
plt.savefig('plots/b_test_power.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/b_test_power.svg', format='svg', bbox_inches='tight')
plt.show()


# Extract true and estimated rates of unreported tests for scatter plot
true = [x['missing'] for x in data_list]
estimated = [x.missing_estimation / (1 + x.missing_estimation) for x in h1_analyses]

# Identify significant results based on the 95th quantile threshold
signif = [x.lrts > quantile_95 for x in h1_analyses]

# Define colors for scatter plot based on significance
colors = ['green' if s else 'darkblue' for s in signif]

# Create scatter plot to compare true vs. estimated unreported test rates
plt.figure(figsize=(8, 6))
plt.scatter(true, estimated, c=colors, alpha=0.4, s=1)
plt.scatter([], [], color='green', label='Significant results')
plt.scatter([], [], color='darkblue', label='Nonsignificant results')
plt.title('True vs Estimated Rate of Unreported Tests')
plt.xlabel('True Rate of Unreported Tests')
plt.ylabel('Estimated Rate of Unreported Tests')
plt.ylim(0, 1)
plt.xlim(0, 0.9)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Line indicating ideal fit
# Set grid and tick intervals
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Save scatter plot in both PNG and SVG formats
plt.savefig('plots/b_test_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/b_test_scatter.svg', format='svg', bbox_inches='tight')
plt.show()


from scipy.stats import spearmanr  # For calculating Spearman correlation

# Specify sample sizes to filter and plot by
sizes = [100, 200, 400, 800]
fs = 20  # Font size for plot labels and text

# Set up a 2x2 grid of subplots to show scatter plots for different sample sizes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

# Loop over each sample size and plot in corresponding subplot
for i, size in enumerate(sizes):
    # Filter data for the current sample size
    true = [x['missing'] for x in data_list if x['size'] == size]
    estimated = [x.missing_estimation / (1 + x.missing_estimation) for x, y in zip(h1_analyses, data_list) if y['size'] == size]
    signif = [x.lrts > quantile_95 for x, y in zip(h1_analyses, data_list) if y['size'] == size]
    
    # Calculate power as the percentage of significant results
    power = 100 * sum(signif) / len(signif) if signif else 0  # Avoid division by zero
    
    # Calculate Spearman correlation between true and estimated rates
    if len(true) > 1:  # Ensure there is more than one data point to calculate correlation
        spearman_corr, _ = spearmanr(true, estimated)
    else:
        spearman_corr = float('nan')  # Set to NaN if insufficient points for correlation
    
    # Define colors for points based on significance
    colors = ['green' if s else 'darkblue' for s in signif]

    # Plot true vs. estimated values in the current subplot
    ax = axes[i]
    ax.scatter(true, estimated, c=colors, alpha=0.6, s=2)
    ax.set_title(f'Sample Size = {size}', fontsize=fs)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 0.9)
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')  # Ideal fit line
    
    # Display power and Spearman correlation on the plot
    ax.text(0.05, 0.85, f"Power: {power:.2f}%\nSpearman r: {spearman_corr:.2f}", 
            transform=ax.transAxes, fontsize=fs, color='black', 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.6))
    
    # Set grid with intervals of 0.1
    ax.set_xticks([i * 0.1 for i in range(10)])
    ax.set_yticks([i * 0.1 for i in range(11)])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add legend to last subplot
    if i == len(sizes) - 1:
        ax.scatter([], [], color='green', label='Significant results')
        ax.scatter([], [], color='darkblue', label='Nonsignificant results')
        ax.legend(loc='lower right', fontsize=fs)

# Common x and y labels for all subplots
plt.figtext(0.5, 0.01, 'True Rate of Unreported Tests', ha='center', fontsize=fs)
plt.figtext(0.01, 0.5, 'Estimated Rate of Unreported Tests', va='center', rotation='vertical', fontsize=fs)

plt.tight_layout(rect=[0.02, 0.02, 1, 1])  # Adjust layout for labels

# Save the subplots to files
plt.savefig('plots/b_test_scatter_size.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/b_test_scatter_size.svg', format='svg', bbox_inches='tight')
plt.show()
