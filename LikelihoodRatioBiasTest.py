from em_functions import *
from helper_functions import *
import numpy as np
from copy import deepcopy
from scipy.stats import chi2
import pandas as pd


# Attempt to set the max CPU count to physical cores only (relevant if using parallel computation in bootstrap)
try:
    import os
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    os.environ['LOKY_MAX_CPU_COUNT'] = str(physical_cores)
except:
    pass


# Define the main class for the Likelihood Ratio Bias Test (LRBT)
class LikelihoodRatioBiasTest:
    initialized_bootstrap = False
    used_method = None

    # Define bias models used for bias distributions in the LRBT
    default_bias_functions_lrbt =[(0,1.96),(1,1.96),(1.96,1.96),(3,1.96)]
    default_distributions_z_curve =[(0,1.96),(1,1.96),(1.96,1.96),(3,1.96),(4,1.96),(5,1.96)]
    default_tolerance_lrbt = 1e-2
    default_tolerance_z_curve = 1e-2
    default_distro_lrbt = "FoldedNormal"
    degrees_of_freedom = 0.6
    p_value = None
    # Initialize the test with data and parameters
    def __init__(self, data, format="guess", bias_functions='default', n_clusters='default', method = "LRBT", tolerance = 'default', fixed_mu = "default"):
        
        self.input = deepcopy(data)
        self.len_data = len(self.input)
        self.used_method = method
        # Check for custom bias functions; use default if not specified or incorrect


        if format == "guess":
            if isinstance(data, (list, np.ndarray)):
                data = np.array(data)
                if np.all((data >= 0) & (data <= 1)):
                    format = "P-values"
                else:
                    format == "Z-values"
            else: return "Can't recognise, it should be numpy array or list, with z-values or p-values"

        if format == "P-values":
            from scipy.stats import norm
            data = [norm.ppf(1 - p_value / 2) for p_value in data]
        elif format == "Z-values":
            data = np.abs(data)

        # filter z_scores bigger than 6
        self.observed_discovery_rate = len(data[data>1.96])/len(data)
        data = data[data < 6]
        self.z_bigger_than_6 = (self.len_data - len(data))/self.len_data
        self.data = deepcopy(data)  # Store data


        if method == "LRBT":

            # load default bias functions
            if bias_functions == 'default': bias_functions = deepcopy(self.default_bias_functions_lrbt)
            if tolerance == 'default': tolerance = deepcopy(self.default_tolerance_lrbt)

            # init bias function
            self.bias = [*[
            {
                'name': 'P_hacked',
                'pdf': truncated_gaussian_pdf_wrapper,
                'params': {'mu': i[0], 'sigma': 1, 'a': i[1]- i[0], 'b': np.inf}
            } for i in bias_functions
            ]]

            # Generating initial cluster centers
            if n_clusters=='default':   n_clusters = int(np.max([5, np.power(len(data), 1/3) - len(self.bias)]))

            percentile_centers  = np.linspace(100 / (2 * n_clusters), 100 - 100 / (2 * n_clusters), n_clusters)
            centers             = np.percentile(data, percentile_centers)

            if   fixed_mu == "default" :   distro_name = self.default_distro_lrbt
            elif fixed_mu: distro_name = 'FoldedNormalFixed'
            else:          distro_name = 'FoldedNormal'

            # Initialize distributions with quantile-based centers
            distributions = [
                {
                    'name': "TruncatedGaussian",
                    'pdf': truncated_gaussian_pdf_wrapper,
                    'params': {'mu': 0, 'sigma': 1, 'a': 0, 'b': np.inf}
                },
                *[
                    {
                        'name': distro_name,
                        'pdf': folded_normal_pdf_wrapper,
                        'params': {'mu': find_mu_fast(center, mu_values, precomputed_means), 'sigma': 1}
                    } for center in centers[1:n_clusters]
                ],
            ]

            # Save initialized quantile-based distributions for later use
            self.quantile_distributions = deepcopy(distributions)

            # Fit the "no-bias" model using the EM algorithm
            pi = np.full(len(distributions), 1.0 / len(distributions))
            self.no_b = em_algorithm(data, distributions, pi, tolerance=tolerance)

            # Calculate "bias" model, starting from no bias estimation
            pi = np.full(len(distributions), 1.0 / len(distributions))
            pi = np.concatenate((self.no_b['pi'], np.full(len(self.bias), 0.1)))
            pi /= pi.sum()  
            distributions = deepcopy(self.quantile_distributions + self.bias)

            # Start algorithm
            bias1 = em_algorithm(data, distributions, pi, tolerance=tolerance)

            # Calculate "bias" model, starting from no specified distribution
            pi = np.concatenate((self.no_b['pi'], np.full(len(self.bias), 0.1)))
            pi /= pi.sum()
            distributions = deepcopy(self.no_b['distributions'] + self.bias)

            # Start algorithm
            bias2 = em_algorithm(data, distributions, pi, tolerance=tolerance)

            if bias1['loglikelihood'] > bias2['loglikelihood']:
                self.ex_b = bias1

            else:
                self.ex_b = bias2

            self.lrts = 2 * ( self.ex_b['loglikelihood'] - self.no_b['loglikelihood'])

            self.missing_estimation = missing_studies(self.ex_b['distributions'],self.ex_b['pi'], z_over = self.z_bigger_than_6)
            self.estimated_discovery_rate = estimated_discovery_rate(self.ex_b['distributions'],self.ex_b['pi'], z_over = self.z_bigger_than_6)
            self.p_value = chi2.sf(self.lrts, self.degrees_of_freedom)

        if method == "Gradient":
            print("")
            # Place for gradient based algorithm

        if method == "Z_Curve":
            if bias_functions == 'default': bias_functions = deepcopy(self.default_distributions_z_curve)
            if tolerance == 'default': tolerance = deepcopy(self.default_tolerance_z_curve)
            distributions =  [*[
            {
                'name': "P_hacked",
                'pdf': folded_normal_pdf_wrapper,
                'params': {'mu': i[0], 'sigma': 1, 'a': i[1]- i[0], 'b': np.inf}
            } for i in bias_functions
            ]]

            pi = np.full(len(distributions), 1.0 / len(distributions))


            data = data[data>1.96]
            self.z_lower_than_1_96 = len(self.data) - len(data)

            self.z_curve2 = em_algorithm(data, distributions, pi, tolerance=tolerance)
            self.estimated_discovery_rate = estimated_discovery_rate(self.z_curve2['distributions'],self.z_curve2['pi'], z_over = self.z_bigger_than_6)
            self.missing_estimation = 1 -  self.estimated_discovery_rate / self.observed_discovery_rate





    def bootstrap(self, n_samples=500, parallel_computing = True):
        from tqdm import tqdm
        self.initialized_bootstrap = True
        data = deepcopy(self.data)

        if self.used_method == "Z_Curve":
            data = data[data>1.96]

        n_data = len(data)

        if self.used_method == "LRBT":
            subsample_size = int(np.ceil(n_data**0.7)) # Ma, Y., Leng, C., & Wang, H. (2024). Optimal subsampling bootstrap for massive data. Journal of Business & Economic Statistics, 42(1), 174-186.
        elif self.used_method == "Z_Curve":
            subsample_size = int(np.ceil(n_data**0.7))


        bootstrap_indices = np.array([np.random.choice(n_data, subsample_size, replace=False) for _ in range(n_samples)])
        bootstrap_data = np.array([data[indices] for indices in bootstrap_indices])



        # Define a single bootstrap iteration
        def single_iteration(data, used_method):

            if used_method == "LRBT":
                pi = deepcopy(self.ex_b['pi'])
                distributions = deepcopy(self.ex_b['distributions'])
            elif used_method == "Z_Curve":
                pi = deepcopy(self.z_curve2['pi'])
                distributions = deepcopy(self.z_curve2['distributions'])

            return em_algorithm(data, distributions, pi)

        if parallel_computing:
            # Parallelize bootstrap iterations

            from joblib import Parallel, delayed

            bootstrap_results = Parallel(n_jobs=-1)(
                delayed(single_iteration)(bootstrap_data[i], self.used_method) for i in tqdm(range(n_samples))
            )
        else:
            bootstrap_results = [single_iteration(bootstrap_data[i], self.used_method) for i in range(n_samples)]

        # Extract missing studies estimates
        edr_list = [estimated_discovery_rate(result['distributions'],result['pi']) for result in bootstrap_results]
        if self.used_method == "LRBT":
            missing_values_list = [missing_studies(result['distributions'],result['pi']) for result in bootstrap_results]
        elif self.used_method == "Z_Curve":
            edr_array = np.array(edr_list)
            missing_values_list = 1 - edr_array / self.observed_discovery_rate



        # Compute confidence intervals
        l_missing = np.percentile(missing_values_list, 2.5)
        u_missing = np.percentile(missing_values_list, 97.5)
        l_edr = np.percentile(edr_list, 2.5)
        u_edr= np.percentile(edr_list, 97.5)

        # Helper function to find the closest value to the percentile target
        def find_closest_value(value_list, target_value):
            closest_index = (np.abs(np.array(value_list) - target_value)).argmin()
            return closest_index

        # Retrieve models for lower and upper confidence intervals
        lower_missings  = find_closest_value(missing_values_list, l_missing)
        upper_missings  = find_closest_value(missing_values_list, u_missing)

        self.bootstraped_ci_models = {
            "lower": bootstrap_results[lower_missings],
            "upper": bootstrap_results[upper_missings]
        }

        self.missing_values_ci = {
            "lower": l_missing,
            "upper": u_missing
        }

        self.estimated_discovery_rate_ci = {
            "lower": l_edr,
            "upper": u_edr
        }


        return self


    # Visualization method to plot model distributions
    def visualize(self, title = 'Likelihood Ratio Bias Test'):
        import matplotlib.pyplot as plt
        data = self.data

        # Define a range of x values over which to evaluate the PDFs
        x_values = np.linspace(0, 6, 1000)
        plt.figure(figsize=(8, 8), dpi=100)

        # Define histogram bins starting at 0 with a bin width of 0.196
        bins = np.arange(0, max(data) + 0.196, 0.196)

        # Plot the histogram of the data
        plt.hist(data, bins=bins, density=True, alpha=0.8, color='gray', label='Data Histogram')

        if self.used_method in ["LRBT"]:
            # Display bootstrap confidence interval if available
            if self.initialized_bootstrap:
                text = f"Unreported tests {round(100 * self.missing_estimation, 2)}%, 95%CI [{round(100 * self.missing_values_ci['lower'], 2)}, {round(100 * self.missing_values_ci['upper'], 2)}]"
                models = {
                    'no_b': {'distributions': self.no_b['distributions'], 'pi': self.no_b['pi'], 'linestyle': '-', 'color': 'blue', 'label': '<No Bias> Estimated true distribution.'},
                    'ex_b': {'distributions': self.ex_b['distributions'], 'pi': self.ex_b['pi'], 'linestyle': '-', 'color': 'red', 'label': '<Ex Bias> Estimated true distribution.'},
                    'low_ci': {'distributions': self.bootstraped_ci_models['lower']['distributions'], 'linestyle': '--', 'pi': self.bootstraped_ci_models['lower']['pi'], 'color': 'red', 'label': '<Ex Bias> Lower CI.'},
                    'high_ci': {'distributions': self.bootstraped_ci_models['upper']['distributions'], 'linestyle': '--', 'pi': self.bootstraped_ci_models['upper']['pi'], 'color': 'red', 'label': '<Ex Bias> Upper CI.'}
                }
            else:
                text =  f"Unreported tests {round(100 * self.missing_estimation, 2)}%"
                models = {
                    'no_b': {'distributions': self.no_b['distributions'], 'pi': self.no_b['pi'], 'linestyle': '-', 'color': 'blue', 'label': '<No Bias> Estimated true distribution.'},
                    'ex_b': {'distributions': self.ex_b['distributions'], 'pi': self.ex_b['pi'], 'linestyle': '-', 'color': 'red', 'label': '<Ex Bias> Estimated true distribution.'}
                }
        elif self.used_method in ["Z_Curve"]:
                text =  f"Unreported tests {round(100 * self.missing_estimation, 2)}%"
                models = {
                    'z_curve2': {'distributions': self.z_curve2['distributions'], 'pi': self.z_curve2['pi'], 'linestyle': '-', 'color': 'blue', 'label': '<No Bias> Estimated true distribution.'}
                }

        # Plot overall mixture density for each model
        for model_key, model_details in models.items():
            mixture_density = np.zeros_like(x_values)

            for i, dist in enumerate(model_details['distributions']):
                params = dist['params']
                pdf_values = folded_normal_pdf_wrapper(x_values, mu=params['mu'], sigma=params['sigma'])
                weighted_pdf_values = model_details['pi'][i] * pdf_values

                if dist['name'] == "P_hacked":  
                    weighted_pdf_values = weighted_pdf_values / probability_Y_greater_than_a(params['mu'], params['a'])
                if self.used_method in ["Z_Curve"]:
                    weighted_pdf_values *= self.observed_discovery_rate

                mixture_density += weighted_pdf_values

            # Plot the calculated mixture density
            plt.plot(x_values, mixture_density, label=model_details['label'], color=model_details['color'], linestyle=model_details['linestyle'])

        # Add a vertical line at x = 1.96 (significance threshold)
        plt.axvline(x=1.96, color='black', linestyle='--', label='Critical Value (for p = 0.05)')

        plt.plot([], [], ' ', label=text)

        # Labels and legend
        plt.xlabel('Z-scores')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend(loc='upper right')

        # Set x-axis limits
        plt.xlim(0, 6)
        plt.show()

    def __repr__(self):

        missings = f"{round(self.missing_estimation * 100, 2)}%"
        edr = f"{round(self.estimated_discovery_rate * 100, 2)}%"

        p_value = f"{self.p_value:.4f}" if self.p_value is not None else "N/A"

        # Add confidence intervals if bootstrap is initialized
        if self.initialized_bootstrap:
            missings_ci = (
                f" 95% CI [{round(self.missing_values_ci['lower'] * 100, 2)}%, "
                f"{round(self.missing_values_ci['upper'] * 100, 2)}%]"
            )
            missings += missings_ci

            edr_ci = (
                f" 95% CI [{round(self.estimated_discovery_rate_ci['lower'] * 100, 2)}%, "
                f"{round(self.estimated_discovery_rate_ci['upper'] * 100, 2)}%]"
            )
            edr += edr_ci

        # Determine the maximum width for alignment
        col1_width = 10  # Width for "Metric"
        col2_width = 50  # Adjusted width for "Value" to accommodate CIs

        # Format the table with aligned columns
        table = (
            f"{'Metric':<{col1_width}} | {'Value':<{col2_width}}\n"
            f"{'-' * (col1_width + col2_width + 3)}\n"
            f"{'Missings':<{col1_width}} | {missings:<{col2_width}}\n"
            f"{'EDR':<{col1_width}} | {edr:<{col2_width}}\n"
            f"{'P_value':<{col1_width}} | {p_value:<{col2_width}}"
        )
        return table
