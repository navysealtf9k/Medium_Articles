import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 


#------------------------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------------------------
def plot_kde_hist(sample: np.array, plt_title: str, x_title: str,
                  save_fig: bool=False, file_name: str=None, **kwargs):

    # Plot its empirical distribution
    sns.set_style("darkgrid")
    axes = sns.histplot(sample, color=kwargs.get('hist_col', 'deepskyblue'), stat='density')
    axes = sns.kdeplot(sample, color=kwargs.get('kde_col', 'red'))
    axes.set_title(plt_title)
    axes.set_xlabel(x_title)

    if save_fig:
        plt.savefig(file_name)
    plt.show()
#-------------------------------------------------------------------------------------------------

# Initializations
np.random.seed(100)
n_obs = 1000
B = 1000

# Draw random numbers from a non-normal distribution
obs_sample = np.random.normal(loc=50, scale=15, size=n_obs)

# Visualize shape of random sample's distribution
plot_kde_hist(sample=obs_sample, plt_title='Thumbnail A: View Time Distribution',
              x_title='Time (minutes)', save_fig=False)

# Compute standard error of sample mean using theoretical formula
obs_mean = np.mean(obs_sample)
obs_se_mean_hat = (np.var(obs_sample) / (n_obs - 1))**(0.5)
print(f'Observed mean is - {obs_mean}')
print(f'SE of observed mean statistic is - {obs_se_mean_hat}')


########################################################################################
# The Bootstrap Algorithm
########################################################################################
boot_means = []


for iteration in range(1, B+1):
    boot_sample = np.random.choice(obs_sample, size=n_obs, replace=True)

    # Observe the shape of two bootsrap samples
    if iteration % 250 == 0:
        plot_kde_hist(sample=boot_sample, plt_title=f'Bootstrap sample - {iteration}',
                      x_title='Time (minutes)', save_fig=False, hist_col='green')

    # Compute sample mean
    boot_means.append(np.mean(boot_sample))

# Compute bootstrap se of mean statistic
boot_se_mean_hat = (np.sum((boot_means - np.mean(boot_means))**2) / (B - 1))**(1/2)
print(f'SE of bootstap mean statistic is - {boot_se_mean_hat}')

# # Plot distribution of means
plot_kde_hist(sample=boot_means, plt_title='Distribution of Bootstrapped Means',
              x_title='Mean Time (minutes)', save_fig=False, hist_col='orange')
