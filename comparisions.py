import pickle
from tqdm import tqdm  # Progress bar for iteration
from joblib import Parallel, delayed  # Parallel computation for faster processing
from LikelihoodRatioBiasTest import LikelihoodRatioBiasTest  # Main class for bias testing
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr  # Spearman correlation calculation
import csv

# Minor configuration for parallel processing based on physical CPU cores
try:
    import os
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    os.environ['LOKY_MAX_CPU_COUNT'] = str(physical_cores)
except:
    pass

# Load test data from 'sample_to_tests.pkl' to ensure consistent data across tests
with open('sample_to_tests.pkl', 'rb') as file:
    data_list = pickle.load(file)

# Define the Caliper Test function to compare z-scores around a critical value
def caliper_test(z_scores, critical_z=1.96, caliper_width=0.5, plot=False):
    # Define z-score ranges immediately below and above the critical value
    caliper_below = (critical_z - caliper_width, critical_z)
    caliper_above = (critical_z, critical_z + caliper_width)
    
    # Count observations within each range
    obs_below = np.sum((z_scores >= caliper_below[0]) & (z_scores < caliper_below[1]))
    obs_above = np.sum((z_scores >= caliper_above[0]) & (z_scores < caliper_above[1]))
    return (obs_below, obs_above)

# Compute results for Likelihood Ratio Bias Test (LRBT) and Caliper Test in parallel
LikelihoodRatioBiasTest_results = Parallel(n_jobs=-1)(
    delayed(LikelihoodRatioBiasTest)(dataset['data']) for dataset in tqdm(data_list)
)
caliper_test_results = Parallel(n_jobs=-1)(
    delayed(caliper_test)(dataset['data']) for dataset in tqdm(data_list)
)

# Load precomputed Z-Curve 2 results from a CSV file
z_curve_results = []
with open('results_r.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    for row in reader:
        value = row[0]
        if value == 'missings':  # Skip header
            continue
        z_curve_results.append(float(value))  # Convert valid values to float

# Calculate the estimated missing rates for each method
LikelihoodRatioBiasTest_results = [x.missing_estimation / (1 + x.missing_estimation) for x in LikelihoodRatioBiasTest_results]
caliper_test_results = [(x[1] - x[0]) / (x[1] + x[0]) for x in caliper_test_results]

# Extract true values of missing rates from the dataset
true = [x['missing'] for x in data_list]

# Compute differences (errors) between estimated and true values for each method
dif_LRBT = [est - true_val for est, true_val in zip(LikelihoodRatioBiasTest_results, true)]
dif_CT = [est - true_val for est, true_val in zip(caliper_test_results, true)]
dif_Z2 = [est - true_val for est, true_val in zip(z_curve_results, true)]

# Calculate mean bias and standard deviation for each method
dif_LRBT_bias = np.mean(dif_LRBT)
dif_CT_bias = np.mean(dif_CT)
dif_Z2_bias = np.mean(dif_Z2)

dif_LRBT_std = np.std(dif_LRBT)
dif_CT_std = np.std(dif_CT)
dif_Z2_std = np.std(dif_Z2)

# Calculate Spearman correlation between estimated and true values for each method
LRBT_cor, _ = spearmanr(LikelihoodRatioBiasTest_results, true)
CT_cor, _ = spearmanr(caliper_test_results, true)
Z2_cor, _ = spearmanr(z_curve_results, true)

# Calculate Mean Squared Error (MSE) for each method
mse_LRBT = np.mean([(est - true_val) ** 2 for est, true_val in zip(LikelihoodRatioBiasTest_results, true)])
mse_CT = np.mean([(est - true_val) ** 2 for est, true_val in zip(caliper_test_results, true)])
mse_Z2 = np.mean([(est - true_val) ** 2 for est, true_val in zip(z_curve_results, true)])


print("Mean Squared Error values: (lower values = better)")
print(f"Likelihood Ratio Bias Test (LRBT): MSE = {mse_LRBT:.4f}")
print(f"Caliper Test: MSE = {mse_CT:.4f}")
print(f"Z-curve 2: MSE = {mse_Z2:.4f}")

print(f"Mean and Standard Deviation values: (closer to 0 = better)")

print(f"LikelihoodRatioBiasTest_results: Bias = {dif_LRBT_bias:.4f}, Error(std) = {dif_LRBT_std:.4f}")
print(f"Caliper Test: Bias = {dif_CT_bias:.4f}, Error(std) = {dif_CT_std:.4f}")
print(f"Z-curve 2: Bias = {dif_Z2_bias:.4f}, Error(std) = {dif_Z2_std:.4f}")

print(f"Correlation values: (closer to 1 = better)")

print(f"LikelihoodRatioBiasTest_results: Corrlation with true = {LRBT_cor:.4f}")
print(f"Caliper Test: Corrlation with true = {CT_cor:.4f}")
print(f"Z-curve 2: Corrlation with true = {Z2_cor:.4f}")


fs = 15

plt.figure(figsize=(16, 5))

# LRBT Scatter Plot
plt.subplot(1, 3, 1)
plt.scatter(true, LikelihoodRatioBiasTest_results, alpha=0.6, color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.ylim(-1, 1)
plt.xlim(0, 1)
plt.title('LRBT')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 - 1 for i in range(21)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.text(0.95, 0.05, f"MSE: {mse_LRBT:.4f}\nBias: {dif_LRBT_bias:.4f}\nStd Dev: {dif_LRBT_std:.4f}\nCorrelation: {LRBT_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='lightblue'))

# Caliper Test Scatter Plot
plt.subplot(1, 3, 2)
plt.scatter(true, caliper_test_results, alpha=0.6, color='orange')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.ylim(-1, 1)
plt.xlim(0, 1)
plt.title('Caliper Test')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 - 1 for i in range(21)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.text(0.95, 0.05, f"MSE: {mse_CT:.4f}\nBias: {dif_CT_bias:.4f}\nStd Dev: {dif_CT_std:.4f}\nCorrelation: {CT_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='moccasin'))

# Z-curve 2 Scatter Plot
plt.subplot(1, 3, 3)
plt.scatter(true, z_curve_results, alpha=0.6, color='green')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.ylim(-1, 1)
plt.xlim(0, 1)
plt.title(' Z-curve2')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 - 1 for i in range(21)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.text(0.95, 0.05, f"MSE: {mse_Z2:.4f}\nBias: {dif_Z2_bias:.4f}\nStd Dev: {dif_Z2_std:.4f}\nCorrelation: {Z2_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='lightgreen'))

# Set common labels for x and y axes
plt.figtext(0.5, 0.01, 'True rate of unreported tests', ha='center', fontsize=12)
plt.figtext(0.01, 0.5, 'Estimated rate of unreported tests', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 1, 1])  # Adjust layout to make room for common axis labels

# Save the plot to a file
plt.savefig('plots/comparisions1.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/comparisions1.svg', format='svg', bbox_inches='tight')
plt.show()






# Replace negative values with 0
LikelihoodRatioBiasTest_results_non_neg = np.maximum(LikelihoodRatioBiasTest_results, 0)
caliper_test_results_non_neg = np.maximum(caliper_test_results, 0)
z_curve_results_non_neg = np.maximum(z_curve_results, 0)

# Calculate differences for bias and standard deviation
dif_LRBT_non_neg = [est - true_val for est, true_val in zip(LikelihoodRatioBiasTest_results_non_neg, true)]
dif_CT_non_neg = [est - true_val for est, true_val in zip(caliper_test_results_non_neg, true)]
dif_Z2_non_neg = [est - true_val for est, true_val in zip(z_curve_results_non_neg, true)]

# Recalculate bias and standard deviation for non-negative versions
dif_LRBT_bias_non_neg = np.mean(dif_LRBT_non_neg)
dif_CT_bias_non_neg = np.mean(dif_CT_non_neg)
dif_Z2_bias_non_neg = np.mean(dif_Z2_non_neg)

dif_LRBT_std_non_neg = np.std(dif_LRBT_non_neg)
dif_CT_std_non_neg = np.std(dif_CT_non_neg)
dif_Z2_std_non_neg = np.std(dif_Z2_non_neg)

# Recalculate Spearman correlation for non-negative versions
LRBT_cor_non_neg, _ = spearmanr(LikelihoodRatioBiasTest_results_non_neg, true)
CT_cor_non_neg, _ = spearmanr(caliper_test_results_non_neg, true)
Z2_cor_non_neg, _ = spearmanr(z_curve_results_non_neg, true)

# Recalculate MSE for non-negative versions
mse_LRBT_non_neg = np.mean([(est - true_val) ** 2 for est, true_val in zip(LikelihoodRatioBiasTest_results_non_neg, true)])
mse_CT_non_neg = np.mean([(est - true_val) ** 2 for est, true_val in zip(caliper_test_results_non_neg, true)])
mse_Z2_non_neg = np.mean([(est - true_val) ** 2 for est, true_val in zip(z_curve_results_non_neg, true)])

# Print the recalculated values
print("Non-negative Mean and Standard Deviation values: (closer to 0 = better)")
print(f"Likelihood Ratio Bias Test (LRBT): Bias = {dif_LRBT_bias_non_neg:.4f}, Error (std) = {dif_LRBT_std_non_neg:.4f}")
print(f"Caliper Test: Bias = {dif_CT_bias_non_neg:.4f}, Error (std) = {dif_CT_std_non_neg:.4f}")
print(f"Z-curve 2: Bias = {dif_Z2_bias_non_neg:.4f}, Error (std) = {dif_Z2_std_non_neg:.4f}")

print("\nNon-negative Correlation values: (closer to 1 = better)")
print(f"Likelihood Ratio Bias Test (LRBT): Correlation with true = {LRBT_cor_non_neg:.4f}")
print(f"Caliper Test: Correlation with true = {CT_cor_non_neg:.4f}")
print(f"Z-curve 2: Correlation with true = {Z2_cor_non_neg:.4f}")

print("\nNon-negative Mean Squared Error values: (lower values = better)")
print(f"Likelihood Ratio Bias Test (LRBT): MSE = {mse_LRBT_non_neg:.4f}")
print(f"Caliper Test: MSE = {mse_CT_non_neg:.4f}")
print(f"Z-curve 2: MSE = {mse_Z2_non_neg:.4f}")


# Replace negative values with 0
LikelihoodRatioBiasTest_results_non_neg = np.maximum(LikelihoodRatioBiasTest_results, 0)
caliper_test_results_non_neg = np.maximum(caliper_test_results, 0)
z_curve_results_non_neg = np.maximum(z_curve_results, 0)

# Recalculate MSE for the modified arrays
mse_LRBT_non_neg = np.mean([(est - true_val) ** 2 for est, true_val in zip(LikelihoodRatioBiasTest_results_non_neg, true)])
mse_CT_non_neg = np.mean([(est - true_val) ** 2 for est, true_val in zip(caliper_test_results_non_neg, true)])
mse_Z2_non_neg = np.mean([(est - true_val) ** 2 for est, true_val in zip(z_curve_results_non_neg, true)])

plt.figure(figsize=(16, 5))

# LRBT Scatter Plot
plt.subplot(1, 3, 1)
plt.scatter(true, LikelihoodRatioBiasTest_results_non_neg, alpha=0.6, color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # diagonal line
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title('LRBT')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Annotate metrics in the bottom-right corner
plt.text(0.95, 0.05, f"MSE: {mse_LRBT_non_neg:.4f}\nBias: {dif_LRBT_bias_non_neg:.4f}\nStd Dev: {dif_LRBT_std_non_neg:.4f}\nCorrelation: {LRBT_cor_non_neg:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='lightblue'))

# Caliper Test Scatter Plot
plt.subplot(1, 3, 2)
plt.scatter(true, caliper_test_results_non_neg, alpha=0.6, color='orange')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # diagonal line
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title('Caliper Test (Non-negative)')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Annotate metrics in the bottom-right corner
plt.text(0.95, 0.05, f"MSE: {mse_CT_non_neg:.4f}\nBias: {dif_CT_bias_non_neg:.4f}\nStd Dev: {dif_CT_std_non_neg:.4f}\nCorrelation: {CT_cor_non_neg:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='moccasin'))

# Z-curve 2 Scatter Plot
plt.subplot(1, 3, 3)
plt.scatter(true, z_curve_results_non_neg, alpha=0.6, color='green')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # diagonal line
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title('Z-curve2 (Non-negative)')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Annotate metrics in the bottom-right corner
plt.text(0.95, 0.05, f"MSE: {mse_Z2_non_neg:.4f}\nBias: {dif_Z2_bias_non_neg:.4f}\nStd Dev: {dif_Z2_std_non_neg:.4f}\nCorrelation: {Z2_cor_non_neg:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='lightgreen'))

# Set common labels for x and y axes
plt.figtext(0.5, 0.01, 'True Missing Values', ha='center', fontsize=12)
plt.figtext(0.01, 0.5, 'Estimated Missing Values', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 1, 1])  # Adjust layout to make room for common axis labels
plt.savefig('plots/comparisions2.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/comparisions2.svg', format='svg', bbox_inches='tight')
plt.show()


# Compare 2
true_bias = [x['bias'] for x in data_list]
n_sig = [len([x for x in y['data'] if x>1.96])/y['size']  for y in data_list]
n_nsig = [len([x for x in y['data'] if x<=1.96])/y['size']  for y in data_list]


Bias_LRBT_EST = [100*x*(1 + x/(1-x))/(y+x/(1-x)) for x,y in zip(LikelihoodRatioBiasTest_results,n_nsig)]
Bias_Caliper_EST = [
    100 if x == 1 else 100*x * (1 + x / (1 - x)) / (y + x / (1 - x))
    for x, y in zip(caliper_test_results, n_nsig)
]
Bias_Z2_EST = [100*x*(1 + x/(1-x))/(y+x/(1-x)) for x,y in zip(z_curve_results,n_nsig)]


# Compute different error statistics
bdif_LRBT = [est-true for est, true in zip(Bias_LRBT_EST, true_bias)]
bdif_CT = [est-true for est, true in zip(Bias_Caliper_EST, true_bias)]
bdif_Z2 = [est-true for est, true in zip(Bias_Z2_EST, true_bias)]



bdif_LRBT_bias = np.mean(bdif_LRBT)
bdif_CT_bias = np.mean(bdif_CT)
bdif_Z2_bias = np.mean(bdif_Z2)

bdif_LRBT_std = np.std(bdif_LRBT)
bdif_CT_std = np.std(bdif_CT)
bdif_Z2_std = np.std(bdif_Z2)

bLRBT_cor, pv = spearmanr(Bias_LRBT_EST, true)
bCT_cor, pv = spearmanr(Bias_Caliper_EST, true)
bZ2_cor, pv = spearmanr(Bias_Z2_EST, true)

bmse_LRBT = np.mean([(est - true_val) ** 2 for est, true_val in zip(Bias_LRBT_EST, true_bias)])
bmse_CT = np.mean([(est - true_val) ** 2 for est, true_val in zip(Bias_Caliper_EST, true_bias)])
bmse_Z2 = np.mean([(est - true_val) ** 2 for est, true_val in zip(Bias_Z2_EST, true_bias)])

print("Mean Squared Error values: (lower values = better)")
print(f"Likelihood Ratio Bias Test (LRBT): MSE = {bmse_LRBT:.4f}")
print(f"Caliper Test: MSE = {bmse_CT:.4f}")
print(f"Z-curve 2: MSE = {bmse_Z2:.4f}")

print(f"Mean and Standard Deviation values: (closer to 0 = better)")

print(f"Likelihood Ratio Bias Test: Bias = {bdif_LRBT_bias:.4f}, Error(std) = {bdif_LRBT_std:.4f}")
print(f"Caliper Test: Bias = {bdif_CT_bias:.4f}, Error(std) = {bdif_CT_std:.4f}")
print(f"Z-curve 2: Bias = {bdif_Z2_bias:.4f}, Error(std) = {bdif_Z2_std:.4f}")

print(f"Correlation values: (closer to 1 = better)")

print(f"Likelihood Ratio Bias Test: Corrlation with true = {bLRBT_cor:.4f}")
print(f"Caliper Test: Corrlation with true = {bCT_cor:.4f}")
print(f"Z-curve 2: Corrlation with true = {bZ2_cor:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns# Set the figure size
plt.figure(figsize=(16, 5))

# LRBT Boxplot
plt.subplot(1, 3, 1)
sns.boxplot(x=true_bias, y=Bias_LRBT_EST, color='blue')
plt.ylim(-1, 1)
plt.title('LRBT')
plt.yticks([i * 10 - 100 for i in range(21)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.text(0.95, 0.05, f"MSE: {bmse_LRBT:.4f}\nBias: {bdif_LRBT_bias:.4f}\nStd Dev: {bdif_LRBT_std:.4f}\nCorrelation: {bLRBT_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='lightblue'))

# Caliper Test Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(x=true_bias, y=Bias_Caliper_EST, color='orange')
plt.ylim(-1, 1)
plt.title('Caliper Test')
plt.yticks([i * 10 - 100 for i in range(21)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.text(0.95, 0.05, f"MSE: {bmse_CT:.4f}\nBias: {bdif_CT_bias:.4f}\nStd Dev: {bdif_CT_std:.4f}\nCorrelation: {bCT_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='moccasin'))

# Z-curve 2 Boxplot
plt.subplot(1, 3, 3)
sns.boxplot(x=true_bias, y=Bias_Z2_EST, color='green')
plt.ylim(-1, 1)
plt.title('Z-curve2')
plt.yticks([i * 10 - 100 for i in range(21)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.text(0.95, 0.05, f"MSE: {bmse_Z2:.4f}\nBias: {bdif_Z2_bias:.4f}\nStd Dev: {bdif_Z2_std:.4f}\nCorrelation: {bZ2_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='lightgreen'))

# Set common labels for x and y axes
plt.figtext(0.5, 0.01, 'True rate of unreported tests', ha='center', fontsize=fs)
plt.figtext(0.01, 0.5, 'Bias Estimate', va='center', rotation='vertical', fontsize=fs)

plt.tight_layout(rect=[0.02, 0.02, 1, 1])  # Adjust layout to make room for common axis labels
plt.savefig('plots/comparisions3.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/comparisions3.svg', format='svg', bbox_inches='tight')
plt.show()


