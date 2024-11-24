from em_functions import *  # Import all necessary functions from the EM algorithm helper module
import numpy as np
from copy import deepcopy

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
    initialized = False  # A flag to track initialization status

    # Define bias models used for bias distributions in the LRBT
    bias = [
        # P-hacked "no effect" distribution
        {
            'name': 'P_hacked',
            'pdf': truncated_gaussian_pdf_wrapper,
            'params': {'mu': 0, 'sigma': 1, 'a': 1.96, 'b': np.inf}
        },
        # P-hacked distribution with 50% power
        {
            'name': 'P_hacked',
            'pdf': truncated_gaussian_pdf_wrapper,
            'params': {'mu': 1.96, 'sigma': 1, 'a': 0, 'b': np.inf}
        },
        # P-hacked distribution with ~32% power
        {
            'name': 'P_hacked',
            'pdf': truncated_gaussian_pdf_wrapper,
            'params': {'mu': 1, 'sigma': 1, 'a': 0.96, 'b': np.inf}
        },
        {
            'name': 'P_hacked',
            'pdf': truncated_gaussian_pdf_wrapper,
            'params': {'mu': 3, 'sigma': 1, 'a': -1.04, 'b': np.inf}
        }
    ]

    # Initialize the test with data and parameters
    def __init__(self, data, format="Z-values", bias_functions='default', n_clusters='default'):
        self.initialized = True

        # Check for custom bias functions; use default if not specified or incorrect
        if bias_functions != 'default':
            try:
                self.bias = bias_functions
            except:
                print("bias_functions should be in specified format. Set to default instead")

        # Convert data format if specified as "Z-values"
        if format == "Z-values":
            data = np.array(data)

        self.data = data  # Store data for use in test

        # Quantile-based initialization to generate initial cluster centers
        n_clusters = int(np.max([5, np.power(len(data), 1/3) - len(self.bias)]))
        quantiles = np.percentile(data, np.linspace(0, 100, n_clusters + 2))

        # Initialize distributions with quantile-based centers
        distributions = [
            {
                'name': 'TruncatedGaussian',
                'pdf': truncated_gaussian_pdf_wrapper,
                'params': {'mu': 0, 'sigma': 1, 'a': 0, 'b': np.inf}
            },
            *[
                {
                    'name': 'FoldedNormal',
                    'pdf': folded_normal_pdf_wrapper,
                    'params': {'mu': find_mu_fast(center, mu_values, precomputed_means), 'sigma': 1}
                } for center in quantiles[1:n_clusters+1]
            ],
        ]

        # Save initialized quantile-based distributions for later use
        self.quantile_distributions = deepcopy(distributions)

        # Fit the "no-bias" model using the EM algorithm
        pi = np.full(len(distributions), 1.0 / len(distributions))
        self.no_b = em_algorithm(data, distributions, pi)

        # Calculate "bias" model, starting from no bias estimation
        pi = np.full(len(distributions), 1.0 / len(distributions))
        pi = np.concatenate((self.no_b['pi'], np.full(len(self.bias), 0.1)))
        pi /= pi.sum()  
        distributions = deepcopy(self.quantile_distributions + self.bias)
        bias1 = em_algorithm(data, distributions, pi)

        # Calculate "bias" model, starting from no specified distribution
        pi = np.concatenate((self.no_b['pi'], np.full(len(self.bias), 0.1)))
        pi /= pi.sum()
        distributions = deepcopy(self.no_b['distributions'] + self.bias)
        bias2 = em_algorithm(data, distributions, pi)

        if bias1['loglikelihood'] > bias2['loglikelihood']:
            self.ex_b = bias1

        else:
            self.ex_b = bias2

        self.lrts = 2 * ( self.ex_b['loglikelihood'] - self.no_b['loglikelihood'])

        self.missing_estimation = self.ex_b['missing_studies']


    def bootstrap(self, n_steps=500, parallel_computing = True):
        from tqdm import tqdm
        n_data = len(self.data)
        # Vectorized sampling of bootstrap datasets
        bootstrap_indices = np.random.randint(0, n_data, (n_steps, n_data))
        bootstrap_data = self.data[bootstrap_indices]

        # Define a single bootstrap iteration
        def single_iteration(data):
            distributions = deepcopy(self.ex_b['distributions'])
            pi = deepcopy(self.ex_b['pi'])
            return em_algorithm(data, distributions, pi)

        if parallel_computing:
            from joblib import Parallel, delayed

            # Parallelize bootstrap iterations
            bootstrap_results = Parallel(n_jobs=-1)(
                delayed(single_iteration)(bootstrap_data[i]) for i in tqdm(range(n_steps))
            )
        else:
            bootstrap_result = [single_iteration(bootstrap_data[i]) for i in tqdm(range(n_steps))]

        # Extract missing studies estimates
        missing_values_list = [result['missing_studies'] for result in bootstrap_results]

        # Compute confidence intervals
        lower_percentile_value = np.percentile(missing_values_list, 2.5)
        upper_percentile_value = np.percentile(missing_values_list, 97.5)

        # Helper function to find the closest value to the percentile target
        def find_closest_value(value_list, target_value):
            closest_index = (np.abs(np.array(value_list) - target_value)).argmin()
            return closest_index

        # Retrieve models for lower and upper confidence intervals
        lower_closest_model = find_closest_value(missing_values_list, lower_percentile_value)
        upper_closest_model = find_closest_value(missing_values_list, upper_percentile_value)

        self.bootstraped_ci_models = {
            "lower": bootstrap_results[lower_closest_model],
            "upper": bootstrap_results[upper_closest_model]
        }

        self.missing_values_ci = {
            "lower": lower_percentile_value,
            "upper": upper_percentile_value
        }

        return self


    # Visualization method to plot model distributions
    def visualize(self, title = 'Likelihood Ratio Bias Test'):
        import numpy as np  # Ensure numpy is imported
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        data = self.data

        # Define a range of x values over which to evaluate the PDFs
        x_values = np.linspace(0, 6, 1000)
        plt.figure(figsize=(8, 8), dpi=100)

        # Define histogram bins starting at 0 with a bin width of 0.196
        bins = np.arange(0, max(data) + 0.196, 0.196)

        # Plot the histogram of the data
        plt.hist(data, bins=bins, density=True, alpha=0.8, color='gray', label='Data Histogram')

        # Display bootstrap confidence interval if available
        if hasattr(self, 'bootstraped_ci_models') and self.bootstraped_ci_models:
            text = f"Unreported tests {round(100 * self.missing_estimation, 2)}%, 95%CI [{round(100 * self.missing_values_ci['lower'], 2)}, {round(100 * self.missing_values_ci['upper'], 2)}]"
            models = {
                'no_b': {'distributions': self.no_b['distributions'], 'pi': self.no_b['pi'], 'linestyle': '-', 'color': 'blue', 'label': '<No Bias> Maximum Likelihood Distr.'},
                'ex_b': {'distributions': self.ex_b['distributions'], 'pi': self.ex_b['pi'], 'linestyle': '-', 'color': 'red', 'label': '<Ex Bias> Maximum Likelihood Distr.'},
                'low_ci': {'distributions': self.bootstraped_ci_models['lower']['distributions'], 'linestyle': '--', 'pi': self.bootstraped_ci_models['lower']['pi'], 'color': 'red', 'label': '<Ex Bias> Lower Confidence Interval.'},
                'high_ci': {'distributions': self.bootstraped_ci_models['upper']['distributions'], 'linestyle': '--', 'pi': self.bootstraped_ci_models['upper']['pi'], 'color': 'red', 'label': '<Ex Bias> Upper Confidence Interval.'}
            }
        else:
            text = ""
            models = {
                'no_b': {'distributions': self.no_b['distributions'], 'pi': self.no_b['pi'], 'linestyle': '-', 'color': 'blue', 'label': '<No Bias> Maximum Likelihood Distr.'},
                'ex_b': {'distributions': self.ex_b['distributions'], 'pi': self.ex_b['pi'], 'linestyle': '-', 'color': 'red', 'label': '<Ex Bias> Maximum Likelihood Distr.'}
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
