from em_functions import *  # Import all necessary functions from the EM algorithm helper module
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

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
        }
    ]

    # Initialize the test with data and parameters
    def __init__(self, data, format="Z-values", start_bias_from='no_bias', bias_functions='default'):
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
        n_clusters = 5
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
                    'params': {'mu': center, 'sigma': 1}
                } for center in quantiles[1:n_clusters+1]
            ],
        ]

        # Save initialized quantile-based distributions for later use
        self.quantile_distributions = deepcopy(distributions)

        # Fit the "no-bias" model using the EM algorithm
        pi = np.full(len(distributions), 1.0 / len(distributions))
        self.no_b = em_algorithm(data, distributions, pi)

        # Fit the "biased" model
        if start_bias_from == 'quantiles':
            distributions = deepcopy(self.quantile_distributions)
            pi = np.full(len(distributions) + len(self.bias), 1.0 / (len(distributions) + len(self.bias)))

        elif start_bias_from == 'no_bias':
            # Start with parameters from the no-bias model and add the bias components
            pi = np.concatenate((self.no_b['pi'], np.full(len(self.bias), 0.1)))
            pi /= pi.sum()
            distributions = deepcopy(self.no_b['distributions'] + self.bias)

        # Apply the EM algorithm for the biased model and calculate test statistics
        self.ex_b = em_algorithm(data, distributions, pi)
        self.missing_estimation = self.ex_b['missing_studies']
        self.lrts = -2 * (self.no_b['loglikelihood'] - self.ex_b['loglikelihood'])

    # Bootstrap method for calculating confidence intervals
    def bootstrap(self, n_steps=500, parallel=False):
        bootstrap = []

        if parallel:
            # Set CPU count for parallel computation if available
            try:
                import os
                import psutil
                physical_cores = psutil.cpu_count(logical=False)
                os.environ['LOKY_MAX_CPU_COUNT'] = str(physical_cores)
            except:
                pass
        else:
            distributions = deepcopy(self.ex_b['distributions'])
            pi = deepcopy(self.ex_b['pi'])

            # Perform bootstrap sampling and store results
            for _ in range(n_steps):
                resampled_data = np.random.choice(self.data, size=len(self.data), replace=True)
                boostraped_single = em_algorithm(resampled_data, distributions, pi)
                bootstrap.append(boostraped_single)

        # Extract 'missing_studies' estimates for confidence interval calculation
        missing_values_list = [result['missing_studies'] for result in bootstrap]

        # Compute confidence intervals (2.5th and 97.5th percentiles)
        lower_percentile_value = np.percentile(missing_values_list, 2.5)
        upper_percentile_value = np.percentile(missing_values_list, 97.5)

        # Helper function to find the closest value to the percentile target
        def find_closest_value(value_list, target_value):
            closest_value = (np.abs(np.array(value_list) - target_value)).argmin()
            return closest_value

        # Retrieve models for lower and upper confidence intervals
        lower_closest_model = find_closest_value(missing_values_list, lower_percentile_value)
        upper_closest_model = find_closest_value(missing_values_list, upper_percentile_value)
        self.bootstraped_ci_models = {"lower": bootstrap[lower_closest_model], "upper": bootstrap[upper_closest_model]}

        # Store confidence interval values for missing studies
        self.missing_values_ci = {"lower": lower_percentile_value, "upper": upper_percentile_value}

        return self

    # Visualization method to plot model distributions
    def visualize(self):
        import numpy as np  # Ensure numpy is imported
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        data = self.data

        # Define a range of x values over which to evaluate the PDFs
        x_values = np.linspace(0, 6, 1000)
        plt.figure(figsize=(10, 6), dpi=100)

        # Define histogram bins starting at 0 with a bin width of 0.196
        bins = np.arange(0, max(data) + 0.196, 0.196)

        # Plot the histogram of the data
        plt.hist(data, bins=bins, density=True, alpha=0.5, color='lightgray', label='Data Histogram')

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
        plt.title('Comparison of Fitted Mixture Models')
        plt.legend(loc='upper right')

        # Set x-axis limits
        plt.xlim(0, 6)

        plt.show()
