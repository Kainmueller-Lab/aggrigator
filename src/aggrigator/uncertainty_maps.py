import numpy as np

from aggrigator.plotting import plot_discrete_distribution, plot_binned_distribution

class UncertaintyMap:
    """Base class for uncertainty maps which may include segmentation masks."""
    def __init__(self, array, mask=None, name=""):
        self.array = array.astype(np.float64) # Uncertainty heatmap array
        self.name = name
        self.shape = array.shape

        self.mask_provided = False if mask is None else True
        self.mask = mask # Corresponding segmentation mask, if provided
        if self.mask_provided:
            assert self.mask.shape == self.array.shape, "Uncertainty array and segmentation mask need to have same dimensions, you ingenious fool!"
        self.class_indices = np.unique(self.mask) if self.mask_provided else None
        self.class_pixels = self.compute_class_pixels() if self.mask_provided else None
        self.class_volumes = self.compute_class_volumes() if self.mask_provided else None


    def compute_class_pixels(self):
        """
        Computes the indices of pixels corresponding to each class in the mask.
        Returns a dictionary where keys are class indices and values are arrays of pixel indices.
        """
        return {idx : np.argwhere(self.mask == idx) for idx in self.class_indices}
    
    def compute_class_volumes(self):
        """
        Computes the number of pixels (volume) corresponding to each class in the mask.
        Returns a dictionary where keys are class indices and values are volumes.
        """
        return {idx : len(self.class_pixels[idx]) for idx in self.class_indices}
    
    def plot_distribution(self):
        return plot_discrete_distribution(self.array)
    
    def plot_binned_distribution(self, bin_size):
        return plot_binned_distribution(self.array, bin_size)