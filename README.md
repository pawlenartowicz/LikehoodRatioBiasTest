This repository hosts the **Likelihood Ratio Test for Publication Bias** ([see more about this method](https://freestylerscientist.pl/lrbt-proof-of-concept/)) by Paweł Lenartowicz.

### How to Use LRBT (Note: Work in Progress)

```python
# Import the class/function
from source.LikelihoodRatioBiasTest import LikelihoodRatioBiasTest

# Save data as a list or numpy array of z-values
data = YourDataAsZValues

# Alternatively, convert p-values to z-values:
from scipy.stats import norm
data = [norm.ppf(1 - p_value / 2) for p_value in YourDataAsPValues]

# Calculate LRBT results
results = LikelihoodRatioBiasTest(data)

# Optionally, calculate bootstrapped confidence intervals
results.bootstrap()

# Visualize the results
results.visualize()

# Access specific results:
results.missing_estimation  # Estimated bias as % of unreported studies
results.lrts                # Likelihood ratio comparing biased and unbiased estimates
```



`