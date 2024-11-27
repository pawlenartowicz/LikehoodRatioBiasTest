import numpy as np
from scipy.stats import truncnorm, norm
from functools import lru_cache
from scipy.special import erf

# Calculate estimated number of missing studies due to publication bias
def missing_studies(distributions,pi):
    ms= 0
    for i, dist in enumerate(distributions):
        if dist['name'] == 'P_hacked':
            mu = dist['params']['mu']
            a = dist['params']['a']
            # Calculate missing studies based on bias in publication
            prob_above = probability_Y_greater_than_a(mu, a)
            ms += pi[i] * ((1 - prob_above) / prob_above)
    ms = ms/(1+ms)
    return ms

def estimated_discovery_rate(distributions,pi):
    edr = 0
    denom = 0
    for i, dist in enumerate(distributions):
        mu = distributions[i]['params']['mu']
        wage = pi[i]

        if dist['name'] == 'P_hacked':
            prob_above = probability_Y_greater_than_a(mu, dist['params']['a'])
            wage = wage / prob_above

        edr += probability_Y_greater_than_a(mu,1.96-mu) * wage
        denom += wage
    edr /= denom
    return edr

def precompute_folded_normal_means(mu_values):
    """Precompute the means of the Folded Normal distribution for sigma=1."""
    # Use NumPy's vectorized operations
    mu_values = np.asarray(mu_values)  # Ensure mu_values is a NumPy array
    means = np.sqrt(2 / np.pi) * np.exp(-mu_values**2 / 2) + mu_values * erf(mu_values / np.sqrt(2))
    return means

# Define the range of mu values (0 to 6, step size 0.005 for high resolution)
mu_values = np.linspace(0, 6, 1201)
precomputed_means = precompute_folded_normal_means(mu_values)

def find_mu_fast(observed_mean, mu_values, precomputed_means):
    """
    Quickly find the mu corresponding to the observed mean using interpolation.
    """
    return np.interp(observed_mean, precomputed_means, mu_values)

# Calculate the probability that a Gaussian variable Y with mean `mu` is greater than a threshold `a`
def probability_Y_greater_than_a(mu, a):
    # Ensure a is non-negative since we're dealing with absolute values
    a = np.abs(mu + a)
    
    # Calculate probability using the corrected formula
    prob_below_a = norm.cdf(a - mu) - norm.cdf(-a - mu)
    prob_above_a = 1 - prob_below_a
    
    return prob_above_a

# A helper function to create a unique hash from numpy array data for caching purposes
def array_hash(data):
    return hash(data.tobytes())

# Cache results for truncated Gaussian calculations for performance improvement
@lru_cache(maxsize=None)
def truncated_gaussian_cached(a, b, mu, sigma, data_hash):
    # Convert hash back to numpy array for PDF computation
    data = np.frombuffer(data_hash, dtype=np.float64)
    return truncnorm.pdf(data, a=a, b=b, loc=mu, scale=sigma)

# Wrapper to compute truncated Gaussian PDF with caching
def truncated_gaussian_pdf_wrapper(data, a, b, mu, sigma):
    data_hash = data.tobytes()  # Convert data to bytes for hashing
    return truncated_gaussian_cached(a, b, mu, sigma, data_hash)

# Cache results for folded normal distribution calculations
@lru_cache(maxsize=None)
def folded_normal_cached(sigma, mu, data_hash):
    # Convert hash back to numpy array for PDF computation
    data = np.frombuffer(data_hash, dtype=np.float64)
    coeff = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    two_sigma_sq = 2 * sigma ** 2
    # Compute both positive and negative components of the folded normal distribution
    exp_component1 = np.exp(-((data - mu) ** 2) / two_sigma_sq)
    exp_component2 = np.exp(-((data + mu) ** 2) / two_sigma_sq)
    return coeff * (exp_component1 + exp_component2)

# Wrapper for folded normal PDF calculation with caching
def folded_normal_pdf_wrapper(data, sigma, mu):
    data_hash = data.tobytes()  # Convert data to bytes for hashing
    return folded_normal_cached(sigma, mu, data_hash)


def calculate_log_likelihood(data, distributions, pi):
    n_data = len(data)
    total_likelihood = np.zeros(n_data)

    # Calculate total likelihood
    for k, dist in enumerate(distributions):
        if dist['name'] in ['TruncatedGaussian', 'P_hacked']:
            pdf_values = truncated_gaussian_pdf_wrapper(data, **dist['params'])
        elif dist['name'] == 'FoldedNormal':
            pdf_values = folded_normal_pdf_wrapper(data, **dist['params'])

        pdf_values = np.maximum(pdf_values, 1e-10)  # Avoid log(0) errors
        total_likelihood += pi[k] * pdf_values

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(np.maximum(total_likelihood, 1e-10)))
    return log_likelihood