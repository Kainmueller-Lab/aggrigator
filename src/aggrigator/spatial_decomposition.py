import numpy as np

from scipy.ndimage import sobel
from skimage.util import view_as_windows

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.spatial import fast_morans_I



def entropy_map_sliding_window(array, window_size=3, param=None):
    """
    Computes local entropy at each pixel using a sliding window, fully vectorized.

    Args:
        array (np.ndarray): 2D array with values in [0, 1].
        window_size (int): Window size.
        bins (int): Number of bins for histogram.

    Returns:
        np.ndarray: 2D entropy map, values in [0, 1]
    """
    pad = window_size // 2
    padded = np.pad(array, pad, mode='reflect')
    
    # shape: (H, W, w, w)
    windows = view_as_windows(padded, (window_size, window_size))
    H, W, _, _ = windows.shape
    flattened = windows.reshape(H * W, -1)  # shape: (N, window_size^2)

    # Compute histograms for each window
    try:
        bins = param["bins"]
    except:
        bins = 4
    bin_edges = np.linspace(0, 1, bins + 1)
    hist = np.apply_along_axis(
        lambda row: np.histogram(row, bins=bin_edges, density=False)[0],
        axis=1,
        arr=flattened
    )  # shape: (N, bins)

    hist = hist.astype(np.float32)
    hist_sum = hist.sum(axis=1, keepdims=True)
    nonzero_mask = hist_sum > 0
    probs = np.zeros_like(hist)
    probs[nonzero_mask.squeeze()] = hist[nonzero_mask.squeeze()] / hist_sum[nonzero_mask].reshape(-1, 1)

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-12), axis=1)
    entropy /= np.log2(bins)  # Normalize to [0,1]

    # Reshape back to array
    return entropy.reshape(H, W)    


def eds_map_sliding_window(array, window_size=3, param=None):
    """
    Vectorized version of local_eds: compute edge density per sliding window (matching pixelwise logic).
    """
    assert array.ndim == 2, "Input must be 2D"
    pad = window_size // 2
    padded = np.pad(array, pad, mode='reflect')
    windows = view_as_windows(padded, (window_size, window_size))
    H, W, _, _ = windows.shape
    flat_windows = windows.reshape(-1, window_size, window_size)  # shape: (H*W, w, w)

    # Compute Sobel gradients per window
    gx = np.array([sobel(w, axis=0, mode='reflect') for w in flat_windows])
    gy = np.array([sobel(w, axis=1, mode='reflect') for w in flat_windows])
    grad_mag = np.hypot(gx, gy)

    # Apply threshold
    try:
        threshold = param["threshold"]
    except:
        threshold = 0.2
    edge_pixels = grad_mag > threshold
    eds = edge_pixels.sum(axis=(1, 2)) / (window_size * window_size)

    return eds.reshape(H, W)


def moran_map_sliding_window(array, window_size=3, param=None):
    """
    Computes a local Moran's I map using the fast_morans_I method for each sliding window.

    Args:
        array (np.ndarray): 2D input array.
        window_size (int): Size of the local window (odd integer, e.g. 3, 5, 7).

    Returns:
        np.ndarray: 2D array of same shape with local Moran's I values (clipped to [0, 1]).
    """
    assert array.ndim == 2, "Input array must be 2D"
    pad = window_size // 2
    padded = np.pad(array, pad, mode='reflect')

    H, W = array.shape
    output = np.zeros_like(array, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            patch = padded[y:y + window_size, x:x + window_size]
            output[y, x] = max(0, fast_morans_I(patch))

    return output


def spatial_decomposition(unc_map, window_size, spatial_measure, param=None):
    """
    Perform spatial decomposition of an uncertainty map using a specified local spatial measure.

    This method computes a spatial weighting map over the input uncertainty map based on a 
    local spatial measure (e.g., edge density, entropy, or Moran's I). The uncertainty map 
    is then decomposed into two components:
    
    - A "high spatial coherence" map, where each pixel is weighted by the spatial measure value.
    - A complementary "low spatial coherence" map, weighted by (1 - measure value).

    The decomposition allows analysis of how much uncertainty mass is concentrated in spatially
    coherent vs. incoherent regions.

    Args:
        unc_map (UncertaintyMap): Input uncertainty map (must have `.array` and `.mask` attributes).
        window_size (int): Size of the local window (odd integer, e.g., 3 or 5) used to compute spatial measures.
        spatial_measure (str): One of {"eds", "moran", "entropy"} indicating the spatial weighting function.
        param (dict, optional): Optional parameters passed to the spatial measure function (e.g., thresholds, bins).

    Returns:
        tuple:
            - UncertaintyMap: Weighted map using the spatial measure (high-coherence component).
            - UncertaintyMap: Inverse-weighted map (low-coherence component).
            - np.ndarray: 2D weight map with values in [0, 1] representing local spatial coherence.
            - float: Mass ratio = (sum of weighted uncertainty) / (sum of total uncertainty).

    Raises:
        ValueError: If an invalid spatial_measure name is given.
    """

    # Map spatial measure name to function
    spatial_funcs = {
        'eds': eds_map_sliding_window,
        'moran': moran_map_sliding_window,
        'entropy': entropy_map_sliding_window
    }

    if spatial_measure not in spatial_funcs.keys():
        raise ValueError(f"Invalid spatial measure: {spatial_measure}")

    # Compute local spatial measure array
    weight_map = spatial_funcs[spatial_measure](unc_map.array, window_size, param)

    # Decomposition of uncertainty map by weighted spatial measure
    weighted = unc_map.array * weight_map
    inv_weighted = unc_map.array * (1 - weight_map)

    # Uncertainty mass ratio
    ratio = np.sum(weighted) / np.sum(unc_map.array)

    return (
        UncertaintyMap(array=weighted, mask=unc_map.mask, name=f"high_{spatial_measure}_filter_size_{window_size}"),
        UncertaintyMap(array=inv_weighted, mask=unc_map.mask, name=f"low_{spatial_measure}_filter_size_{window_size}"),
        weight_map,
        ratio
    )
