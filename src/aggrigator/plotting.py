import numpy as np
import matplotlib.pyplot as plt

def plot_discrete_distribution(array):
    """
    Plots the relative frequency distribution of unique values in the given array.

    Parameters
    ----------
    array : array_like
        Input array to be plotted.

    Returns
    -------
    None
    """

    flat_array = array.flatten()
    N = len(flat_array)

    # Get unique values and their occurrences
    unique_values, counts = np.unique(flat_array, return_counts=True)

    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot a vertical line for each unique value at its occurrence count
    for value, count in zip(unique_values, counts):
        plt.vlines(value, 0, count/N, colors='b', linestyles='solid', label=f'Value {value}' if count == counts[0] else "")

    assert sum([count for count in counts]) == N

    # Adding labels and title
    plt.axhline(0, color='black', linewidth=0.8)  # Draw the x-axis (y=0 line)
    plt.title('Relative frequency distribution of pixelwise uncertainty values')
    plt.xlabel('Uncertainty value')
    plt.ylabel('Relative frequency')
    plt.xticks(unique_values)
    #plt.grid(axis='y', linestyle='solid', alpha=0.7)
    # Show the plot
    plt.show()


def plot_binned_distribution(array, bin_size):
    """
    Plots a binned histogram of the given array, with the bin edges chosen to be equally spaced by the given bin size.

    Parameters
    ----------
    array : array_like
        Input array to be plotted.
    bin_size : float
        The size of each bin in the histogram.

    Returns
    -------
    None

    Notes
    -----
    The histogram is normalized to show relative frequencies of the pixelwise uncertainty values.
    """
    # Calculate bin edges
    flat_array = array.flatten()
    N = len(flat_array)
    min_val, max_val = flat_array.min(), flat_array.max()
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Plot the binned histogram
    plt.hist(flat_array, bins=bins, edgecolor='black', alpha=0.7, weights=np.ones_like(flat_array) / N)
    plt.title(f'Relative frequency distribution of pixelwise uncertainty values\nBin size = {bin_size}')
    plt.xlabel('Uncertainty value')
    plt.ylabel('Frequency')
    plt.xticks(bins)  # Ensure the bin edges are shown as ticks
    plt.show()