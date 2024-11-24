import numpy as np
from scipy.stats import truncnorm, norm
from functools import lru_cache
from scipy.special import erf

def precompute_folded_normal_means(mu_values):
    """Precompute the means of the Folded Normal distribution for sigma=1."""
    means = []
    for mu in mu_values:
        mean = np.sqrt(2 / np.pi) * np.exp(-mu**2 / 2) + mu * erf(mu / np.sqrt(2))
        means.append(mean)
    return np.array(means)

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


# E-step: calculate responsibilities in the EM algorithm
def e_step(data, distributions, pi):
    n_data = len(data)
    n_components = len(distributions)
    responsibilities = np.zeros((n_data, n_components))

    # Compute responsibilities for each component in the mixture model
    for k, dist in enumerate(distributions):
        if dist['name'] in ['TruncatedGaussian', 'P_hacked']:
            pdf_values = truncated_gaussian_pdf_wrapper(data, **dist['params'])
        elif dist['name'] == 'FoldedNormal':
            pdf_values = folded_normal_pdf_wrapper(data, **dist['params'])

        pdf_values = np.maximum(pdf_values, 1e-10)  # Avoid log(0) errors
        responsibilities[:, k] = pi[k] * pdf_values

    # Normalize responsibilities across all components for each data point
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= row_sums

    return responsibilities


# M-step: update parameters for each distribution component based on responsibilities
def m_step(data, responsibilities, distributions):
    n_components = responsibilities.shape[1]
    pis = responsibilities.mean(axis=0)  # Update mixture component weights

    # Update parameters for each distribution component
    for i in range(n_components):
        dist = distributions[i]
        r_i = responsibilities[:, i]
        r_sum = r_i.sum()

        if r_sum > 1e-10:  # Avoid division by zero
            observed_mean = np.sum(r_i * data) / r_sum

            if dist['name'] == 'FoldedNormal':
                # dist['params']['mu'] = observed_mean            
                dist['params']['mu'] = find_mu_fast(observed_mean, mu_values, precomputed_means)

    return pis

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

# EM-algorithm: iteratively apply E-step and M-step until convergence
def em_algorithm(data, distributions, pi, tolerance=0.01, max_iter=1000):    

    prev_log_likelihood = -np.inf  # Initialize previous log-likelihood to a very low value

    for i in range(max_iter):
        # E-step
        responsibilities = e_step(data, distributions, pi)

        # M-step
        pi = m_step(data, responsibilities, distributions)

        # Check for convergence
        if i > 10:
            current_log_likelihood = calculate_log_likelihood(data, distributions, pi)
            if (current_log_likelihood - prev_log_likelihood) < tolerance: break
            prev_log_likelihood = current_log_likelihood

    # Calculate estimated number of missing studies due to publication bias
    missing_studies = 0
    for i, dist in enumerate(distributions):
        if dist['name'] == 'P_hacked':
            mu = distributions[i]['params']['mu']
            a = distributions[i]['params']['a']
            # Calculate missing studies based on bias in publication
            prob_above = probability_Y_greater_than_a(mu, a)
            missing_studies += pi[i] * ((1 - prob_above) / prob_above)
    missing_studies = missing_studies/(1+missing_studies)

    return {"pi": pi, "distributions": distributions, "loglikelihood": current_log_likelihood, "missing_studies": missing_studies}
