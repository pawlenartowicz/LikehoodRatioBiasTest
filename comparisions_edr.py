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
with open('sample_to_tests3.pkl', 'rb') as file:
    data_list = pickle.load(file)

# Compute results for Likelihood Ratio Bias Test (LRBT) and Caliper Test in parallel
LikelihoodRatioBiasTest_results = Parallel(n_jobs=-1)(
    delayed(LikelihoodRatioBiasTest)(dataset['data']) for dataset in tqdm(data_list)
)
LikelihoodRatioBiasTest_results = [x.estimated_discovery_rate for x in LikelihoodRatioBiasTest_results]

# Compute results for Likelihood Ratio Bias Test (LRBT) and Caliper Test in parallel
LikelihoodRatioBiasTest_ZCurve_results = Parallel(n_jobs=-1)(
    delayed(LikelihoodRatioBiasTest)(dataset['data'], method = "Z_Curve2", tolerance = 0.01) for dataset in tqdm(data_list)
)
LikelihoodRatioBiasTest_ZCurve_results = [x.estimated_discovery_rate for x in LikelihoodRatioBiasTest_ZCurve_results]


# Load precomputed Z-Curve 2 results from a CSV file
z_curve_results = []
with open('results_edr_4000.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    for row in reader:
        value = row[0]
        if value == 'estimated_discovery_rate':  # Skip header
            continue
        z_curve_results.append(float(value))  # Convert valid values to float

# Calculate the estimated missing rates for each method


# Extract true values of missing rates from the dataset
true = [x['edr'] for x in data_list]

# Compute differences (errors) between estimated and true values for each method
dif_LRBT = [est - true_val for est, true_val in zip(LikelihoodRatioBiasTest_results, true)]
dif_CT = [est - true_val for est, true_val in zip(LikelihoodRatioBiasTest_ZCurve_results, true)]
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
CT_cor, _ = spearmanr(LikelihoodRatioBiasTest_ZCurve_results, true)
Z2_cor, _ = spearmanr(z_curve_results, true)

# Calculate Mean Squared Error (MSE) for each method
mse_LRBT = np.mean([(est - true_val) ** 2 for est, true_val in zip(LikelihoodRatioBiasTest_results, true)])
mse_CT = np.mean([(est - true_val) ** 2 for est, true_val in zip(LikelihoodRatioBiasTest_ZCurve_results, true)])
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
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title('LRBT')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.text(0.95, 0.05, f"MSE: {mse_LRBT:.4f}\nBias: {dif_LRBT_bias:.4f}\nStd Dev: {dif_LRBT_std:.4f}\nCorrelation: {LRBT_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='lightblue'))

# Caliper Test Scatter Plot
plt.subplot(1, 3, 2)
plt.scatter(true, LikelihoodRatioBiasTest_ZCurve_results, alpha=0.6, color='orange')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title('LRBT ZCurve method')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.text(0.95, 0.05, f"MSE: {mse_CT:.4f}\nBias: {dif_CT_bias:.4f}\nStd Dev: {dif_CT_std:.4f}\nCorrelation: {CT_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='moccasin'))

# Z-curve 2 Scatter Plot
plt.subplot(1, 3, 3)
plt.scatter(true, z_curve_results, alpha=0.6, color='green')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title(' Z-curve2')
plt.xticks([i * 0.1 for i in range(10)])
plt.yticks([i * 0.1 for i in range(11)])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.text(0.95, 0.05, f"MSE: {mse_Z2:.4f}\nBias: {dif_Z2_bias:.4f}\nStd Dev: {dif_Z2_std:.4f}\nCorrelation: {Z2_cor:.4f}",
         transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=fs,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='lightgreen'))

# Set common labels for x and y axes
plt.figtext(0.5, 0.01, 'True discovery rate', ha='center', fontsize=12)
plt.figtext(0.01, 0.5, 'Estimated discovery rate', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 1, 1])  # Adjust layout to make room for common axis labels

# # Save the plot to a file
# plt.savefig('plots/edr1.png', dpi=300, bbox_inches='tight')
# plt.savefig('plots/edr1.svg', format='svg', bbox_inches='tight')
plt.show()
