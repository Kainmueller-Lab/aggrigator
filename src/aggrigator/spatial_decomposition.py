import numpy as np
import warnings

from aggrigator.spatial import local_entropy, local_eds, local_moran
from aggrigator.uncertainty_maps import UncertaintyMap


def spatial_decomposition(unc_map, window_size, spatial_measure, param=None):
    '''
    Decomposes the uncertainty map into two filtered maps based on a spatial measure.
    Each pixelwise uncertainty value is weighted with a pixelwise local spatial measure value v and 1-v.
    This results in two maps, one where the uncertainty with high spatial measure values is kept and one where it is removed.

    :param unc_map: An object containing the uncertainty map.
    :param window_size: The size of the window used for filtering.
    :param spatial_measure: The spatial measure to be used for filtering. Should have values between 0 and 1. Current options are 'eds', 'moran' and 'entropy'.
    :param param: Optional parameter to be passed to the spatial measure.
    :return: A tuple containing the following:
        - Two uncertainty maps, one with high spatial measure values and one with low spatial measure values.
        - The weight map.
        - The mass ratio of the uncertainty mass in the high spatial map compared to the full uncertainty mass.
    '''
    # Extract uncertainty array
    unc_array = unc_map.array

    # Pad uncertainty array
    pad_size = window_size // 2
    padded = np.pad(unc_array, pad_size, mode='reflect')

    # Prepare output
    weight_map = np.zeros_like(unc_array)
    weighted_map = np.zeros_like(unc_array)
    inv_weighted_map = np.zeros_like(unc_array)
    H, W = unc_array.shape

    # Select spatial measure
    if spatial_measure == 'eds':
        local_measure = local_eds
    elif spatial_measure == 'moran':
        local_measure = local_moran
    elif spatial_measure == 'entropy':
        local_measure = local_entropy
    else:
        warnings.warn(f"Invalid spatial measure {spatial_measure}. Using Moran's as default.")
        local_measure = local_moran

    # Sliding window loop
    for y in range(H):
        for x in range(W):
            window = padded[y:y + window_size, x:x + window_size]
            weight_map[y, x] = local_measure(window, param)
            weighted_map[y, x] = unc_array[y, x] * weight_map[y, x]
            inv_weighted_map[y, x] = unc_array[y, x] * (1.0 - weight_map[y, x])

    high_spatial_mass_ratio = np.sum(weighted_map) / np.sum(unc_array)

    return (UncertaintyMap(array=weighted_map, mask=unc_map.mask, name=f"high_{spatial_measure}_filter_size_{window_size}"),
            UncertaintyMap(array=inv_weighted_map, mask=unc_map.mask, name=f"low_{spatial_measure}_filter_size_{window_size}"),
            weight_map,
            high_spatial_mass_ratio)
    