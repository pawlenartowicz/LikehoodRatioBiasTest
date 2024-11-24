import pickle
from tqdm import tqdm  # For progress bar during dataset processing
from joblib import Parallel, delayed  # For parallel processing of datasets
from LikelihoodRatioBiasTest import LikelihoodRatioBiasTest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, probplot  # Statistical functions for chi-squared and Q-Q plotting

# Attempt to set the maximum CPU count to the number of physical cores for parallel processing
try:
    import os
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    os.environ['LOKY_MAX_CPU_COUNT'] = str(physical_cores)
except:
    pass

# Load the data from a pickle file containing preprocessed datasets
with open('h0.pkl', 'rb') as file:
    data_list = pickle.load(file)

# Optionally sample a subset of data if needed
# from random import sample
# data_list = sample(data_list, 100)

# Apply the Likelihood Ratio Bias Test (LRBT) in parallel across all datasets in `data_list`
h0_analyses = Parallel(n_jobs=-1, batch_size='auto')(
    delayed(LikelihoodRatioBiasTest)(dataset['data']) for dataset in tqdm(data_list, desc="Processing datasets")
)

# Extract the likelihood ratio test statistics (LRTS) from each analysis result
ll = [x.lrts for x in h0_analyses]

# problems = [x.data for x in h0_analyses if x.lrts >150]
# with open('problems.pkl', 'wb') as file:
#      pickle.dump(problems, file)


# Calculate chi-squared quantiles for the 95% and 99% thresholds based on degrees of freedom (df)
df = 0.6
quantile_95 = chi2.ppf(0.95, df=df)
quantile_99 = chi2.ppf(0.99, df=df)


# Filter the LRTS data to count how many values exceed the 95% and 99% quantiles
l95 = [x for x in ll if x > quantile_95]
l99 = [x for x in ll if x > quantile_99]


# Generate theoretical and sample quantiles for chi-squared distribution Q-Q plot (without plotting yet)
res = probplot(ll, dist="chi2", sparams=(df,))
theoretical_quantiles, sample_quantiles = res[0]

# Determine x-axis limits for the Q-Q plot with a margin for better visual fit
x_min = theoretical_quantiles.min() - 1
x_max = theoretical_quantiles.max() + 1

# Create a Q-Q plot to compare the sample LRTS values with the chi-squared distribution
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, label="Data Points")

# Plot the ideal fit line for a chi-squared distribution (slope = 1, intercept = 0)
min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Theoretical fit")

# Set x-axis limits to include margin space
plt.xlim(x_min, x_max)

# Add horizontal lines for 95% and 99% quantiles to indicate thresholds
plt.axhline(y=quantile_95, color='blue', linestyle='-.', label=f'95% threshold: {100 * len(l95) / len(ll):.4f}% above')
plt.axhline(y=quantile_99, color='green', linestyle='-.', label=f'99% threshold: {100 * len(l99) / len(ll):.4f}% above')

# Configure plot labels, title, and legend
plt.title(f'Q-Q Plot - Chi-squared Distribution with df = {df}')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.legend()

# # Save the Q-Q plot in both PNG and SVG formats with specified quality settings
plt.savefig('plots/a_test_ll.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/a_test_ll.svg', format='svg', bbox_inches='tight')
plt.show()

ll0 = [x.lrts for x in h0_analyses if x.lrts<0]
print(len(ll0))