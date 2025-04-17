import numpy as np
import warnings

from scipy.signal import convolve
from scipy.stats import gmean, hmean

from aggrigator.util import get_id_mask, get_id_mask_boundary, get_id_mask_interior, get_fg_ratio
from aggrigator.optimized_gearys import fast_gearys_C
from aggrigator.optimized_morans import fast_morans_I


class AggregationMethods:
    """
    A collection of static aggregation strategies to summarize uncertainty maps.
    These methods operate on pixel-wise uncertainty arrays and optionally use segmentation masks.
    """

    # ------------------------- General Methods -------------------------

    @staticmethod
    def mean(unc_map, param=None):
        """
        Compute the mean uncertainty over the entire array.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Mean uncertainty value.
        """
        return np.mean(unc_map.array)

    @staticmethod
    def sum(unc_map, param=None):
        """
        Compute the sum of all uncertainty values.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Sum of uncertainty values.
        """
        return np.sum(unc_map.array, dtype=np.float64)

    @staticmethod
    def max(unc_map, param=None):
        """
        Compute the maximum uncertainty value.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Maximum uncertainty value.
        """
        return np.max(unc_map.array)

    @staticmethod
    def quantile(unc_map, param):
        """
        Compute the specified quantile of the uncertainty values.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Quantile to compute (between 0 and 1).

        Returns
        -------
        float
            The computed quantile.
        """
        return np.quantile(unc_map.array, param)

    @staticmethod
    def std_dev(unc_map, param=None):
        """
        Compute the standard deviation of the uncertainty values.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Standard deviation of uncertainty values.
        """
        return np.std(unc_map.array)

    @staticmethod
    def geometric_mean(unc_map, param=None):
        """
        Compute the geometric mean of all uncertainty values.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Geometric mean or NaN if values < 0 are present.
        """
        if np.any(unc_map.array < 0):
            return np.nan
        return gmean(unc_map.array.flatten())

    @staticmethod
    def harmonic_mean(unc_map, param=None):
        """
        Compute the harmonic mean of all uncertainty values.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Harmonic mean or NaN if values < 0 are present.
        """
        if np.any(unc_map.array < 0):
            return np.nan
        return hmean(unc_map.array.flatten())

    @staticmethod
    def patch_aggregation(unc_map, param):
        """
        Patch-level aggregation: average uncertainty over patches of a given size.
        Returns the highest mean uncertainty over all patches.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : int or list[int]
            Size of the patch (int or per-dimension).

        Returns
        -------
        float
            Maximum average patch uncertainty.
        """
        patch_size = param
        if type(patch_size) == int:
            patch_size = len(unc_map.array.shape) * [patch_size]
        kernel = np.ones(patch_size)
        patch_aggragated = convolve(unc_map.array, kernel, mode="valid")
        patch_aggragated = patch_aggragated / np.prod(patch_size)
        return float(np.max(patch_aggragated))

    # ------------------------- Above Threshold Methods -------------------------

    @staticmethod
    def above_threshold_mean(unc_map, param):
        """
        Mean of all uncertainty values above a given threshold.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Threshold value.

        Returns
        -------
        float
            Mean of values above the threshold, or 0 if none.
        """
        uncertainty_sum = unc_map.array[unc_map.array >= param].sum(dtype=np.float64)
        count = (unc_map.array >= param).sum(dtype=np.float64)
        return uncertainty_sum / count if count > 0 else 0

    @staticmethod
    def above_threshold_sum(unc_map, param):
        """
        Sum of uncertainty values above a given threshold.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Threshold value.

        Returns
        -------
        float
            Sum of values above the threshold.
        """
        return unc_map.array[unc_map.array >= param].sum(dtype=np.float64)

    @staticmethod
    def above_threshold_volume(unc_map, param):
        """
        Normalized count of values above a threshold.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Threshold value.

        Returns
        -------
        float
            Fraction of values above the threshold.
        """
        arr = unc_map.array
        return np.count_nonzero(arr >= param) / arr.size

    @staticmethod
    def above_threshold_std_dev(unc_map, param):
        """
        Standard deviation of uncertainty values above a given threshold.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Threshold value.

        Returns
        -------
        float
            Standard deviation of values above the threshold.
        """
        above_threshold_values = unc_map.array[unc_map.array >= param]
        return np.std(above_threshold_values)

    # ------------------------- Above Quantile Methods -------------------------

    @staticmethod
    def above_quantile_mean(unc_map, param):
        """
        Mean of uncertainty values above a given quantile threshold.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Quantile threshold (0–1).

        Returns
        -------
        float
            Mean of values above the quantile, or NaN if none exist.
        """
        threshold = np.quantile(unc_map.array, param)
        above_threshold_values = unc_map.array[unc_map.array >= threshold]
        return np.mean(above_threshold_values) if above_threshold_values.size > 0 else np.nan

    @staticmethod
    def above_quantile_std_dev(unc_map, param):
        """
        Standard deviation of uncertainty values above a given quantile threshold.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : float
            Quantile threshold (0–1).

        Returns
        -------
        float
            Standard deviation of values above the quantile, or NaN if none exist.
        """
        threshold = np.quantile(unc_map.array, param)
        above_threshold_values = unc_map.array[unc_map.array >= threshold]
        return np.std(above_threshold_values)

    @staticmethod
    def above_quantile_mean_fg_ratio(unc_map, param=None):
        """
        Foreground-based quantile mean.
        Computes the quantile (1 - FG ratio) and returns the mean uncertainty of all higher values.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map (must have a mask).

        Returns
        -------
        float
            Mean of uncertainty values above the (1 - FG ratio) quantile.
        """
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        fg_ratio = get_fg_ratio(unc_map.mask)
        threshold = np.quantile(unc_map.array, 1 - fg_ratio)
        above_threshold_values = unc_map.array[unc_map.array >= threshold]
        return np.mean(above_threshold_values) if above_threshold_values.size > 0 else np.nan

    # ------------------------- Spatial Methods -------------------------

    @staticmethod
    def morans_I(unc_map, param=None):
        """
        Compute Moran's I statistic for spatial autocorrelation.

        Moran's I measures how similar uncertainty values are across neighboring pixels.
        Values close to +1 indicate strong positive spatial autocorrelation (clustering),
        while values near -1 suggest dispersion.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Moran's I value for the uncertainty array.
        """
        return fast_morans_I(unc_map)

    @staticmethod
    def gearys_C(unc_map, param=None):
        """
        Compute Geary's C statistic for spatial autocorrelation.

        Geary's C is a local indicator of spatial association, complementary to Moran’s I.
        Values closer to 0 indicate strong positive spatial correlation,
        and values near 1 suggest spatial randomness.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The input uncertainty map.
        param : None
            Not used for this method.

        Returns
        -------
        float
            Geary's C value for the uncertainty array.
        """
        return fast_gearys_C(unc_map)


    # ------------------------- Class Based Methods -------------------------
    # These methods use additional prediction mask and return a single score for each class where the class is a parameter to be passed to the method.
    # Useful for plotting but not for analyses.
    @staticmethod
    def class_mean(unc_map, param):
        """
        Computes the mean uncertainty value for a specific class label.

        This method uses the segmentation mask provided in the `UncertaintyMap` to isolate
        all pixels belonging to a given class (e.g., foreground or background) and then computes
        the average uncertainty over those pixels.

        Parameters
        ----------
        unc_map : UncertaintyMap
            The uncertainty map with an associated segmentation mask.
        param : int
            The class label to compute the mean uncertainty for.

        Returns
        -------
        float
            The mean uncertainty for the specified class.

        Raises
        ------
        AssertionError
            If no mask is provided or if the class label is invalid.
        """
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        assert param in unc_map.class_indices, f"Invalid class label {param} for uncertainty map {unc_map.name}"
        return np.sum(unc_map.array[unc_map.mask == param], dtype=np.float64) / unc_map.class_volumes[param]

    @staticmethod
    def class_boundary_mean(unc_map, param):
        """
        Computes the mean uncertainty for the boundary of a specified class.

        Args:
            unc_map: An object containing the uncertainty map and associated mask.
            param: A dictionary containing:
                - 'id': The class ID.
                - 'thickness': The boundary thickness.
                - 'nbhood': Neighborhood definition ('full' or 'star').
        Returns: Mean uncertainty within the specified boundary region.
        """
        id = param.get('id', 0)
        thickness = param.get('thickness', 1)
        nbhood = param.get('nbhood', 'full')

        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        assert id in unc_map.class_indices, f"Invalid class label {id} for uncertainty map {unc_map.name}"

        boundary_mask = get_id_mask_boundary(unc_map.mask, id, nbhood=nbhood, thickness=thickness)
        return np.mean(unc_map.array[boundary_mask == 1])
    
    @staticmethod
    def class_interior_mean(unc_map, param):
        """
        Computes the mean uncertainty for the interior of a specified class.

        Args:
            unc_map: An object containing the uncertainty map and associated mask.
            param: A dictionary containing:
                - 'id': The class ID.
                - 'thickness': The boundary thickness.
                - 'nbhood': Neighborhood definition ('full' or 'star').
        Returns: Mean uncertainty within the specified boundary region.
        """
        id = param.get('id', 0)
        thickness = param.get('thickness', 1)
        nbhood = param.get('nbhood', 'full')
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        assert id in unc_map.class_indices, f"Invalid class label {param} for uncertainty map {unc_map.name}"
        interior_mask = get_id_mask_interior(unc_map.mask, id, nbhood=nbhood, thickness=thickness)
        # Warn if interior is empty.
        if not np.any(interior_mask == 1):
            warnings.warn(
                f"No interior pixels found for class {id} in map '{unc_map.name}'. "
                "Returning NaN. Consider adjusting `thickness` or `nbhood`.",
                UserWarning
            )
        return np.mean(unc_map.array[interior_mask == 1])
    
    ######## WEIGHTED CLASS BASED METHODS ########
    @staticmethod
    def class_mean_w_custom_weights(unc_map, param, return_weights=False): # param = weights: A dict of weights for each class you want to include.
        """
        Compute the weighted average of class means, allowing for custom weights.
        Parameters:
        - unc_map: An object containing class indices and a method to compute class means.
        - param (dict, optional): A dictionary specifying custom weights for each class.
        Returns:
        - Weighted average of class means.
        """
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        weights = param
        class_ids = list(weights.keys())
        # Compute class means
        class_means = {class_id: AggregationMethods.class_mean(unc_map, class_id)
                       for class_id in class_ids}
        # Ensure provided weights sum to 1
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, "Weights must sum to 1."
        # Compute the weighted average
        if return_weights:
            return sum(class_means[id] * weights[id] for id in class_ids), weights
        return sum(class_means[id] * weights[id] for id in class_ids)

    def class_mean_w_equal_weights(unc_map, param=None, return_weights=False):
        # NOTE: We exclude BG class 0
        # TODO: Add inlcude BG option?
        fg_classes = [class_id for class_id in unc_map.class_indices if not class_id == 0]
        # Use equal weights for all classes
        weights = {id: 1 / len(fg_classes) for id in fg_classes}
        return AggregationMethods.class_mean_w_custom_weights(unc_map, weights, return_weights)

    def class_mean_weighted_by_occurrence(unc_map, param=None, return_weights=False):
        # NOTE: We exclude BG class 0
        # TODO: Add inlcude BG option?
        fg_classes = [class_id for class_id in unc_map.class_indices if not class_id == 0]
        # Count class pixels 
        class_pixel_counts = {class_id: get_id_mask(unc_map.mask, class_id).sum()
                              for class_id in fg_classes}
        fg_pixel_count = np.sum(list(class_pixel_counts.values()))
        # Use weights proportional to the number of pixels in each class
        weights = {id: class_pixel_counts[id] / fg_pixel_count for id in fg_classes}
        return AggregationMethods.class_mean_w_custom_weights(unc_map, weights, return_weights)


    ## EXTRA METHODS ###############################################
    
    # NOTE: These return not single scores but a dictionary of scores for each class.
    # Useful for analyses but not for plotting.
    @staticmethod
    def class_mean_dict(unc_map, param=None):
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        result_list = {param : AggregationMethods.class_mean(unc_map, param) for param in unc_map.class_indices}
        return result_list
    
    @staticmethod
    def class_boundary_mean_dict(unc_map, param={}):
        """
        Computes the mean uncertainty for the boundary of each class.

        Args:
            unc_map: Object containing the uncertainty map and associated mask.
            param (dict): Optional dictionary with keys:
                - 'thickness': Integer specifying boundary thickness.
                - 'nbhood': String specifying neighborhood definition ('full' or 'star').

        Returns:
            dict: Mapping from class ID to mean uncertainty in the boundary region.
        """
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        result_list = {
            i: AggregationMethods.class_boundary_mean(
            unc_map, {
                'id': i,
                'thickness': param.get('thickness', 1),
                'nbhood': param.get('nbhood', 'full')}) for i in unc_map.class_indices}
        return result_list
    
    @staticmethod
    def class_interior_mean_dict(unc_map, param={}):
        """
        Computes the mean uncertainty for the boundary of all classes.

        Args:
            unc_map: An object containing the uncertainty map and associated mask.
            param: A dictionary containing:
                - 'thickness': The boundary thickness.
                - 'nbhood': Neighborhood definition ('full' or 'star').
        Returns: Mean uncertainty within the specified boundary region.
        """
        assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
        result_list = {i : AggregationMethods.class_interior_mean(
            unc_map, {
                'id': i,
                'thickness': param.get('thickness', 1),
                'nbhood': param.get('nbhood', 'full')}) for i in unc_map.class_indices}
        return result_list