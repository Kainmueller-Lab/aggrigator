import numpy as np
from numba import jit
import libpysal
from scipy.ndimage import sobel


@jit(nopython=True)
def compute_morans_I_numba(x, weights_i, weights_j, weights_data):
    """
    Compute Moran's I using Numba with libpysal weights:
    - Speeds up computation by compiling core operations to machine code.
    - Works with a sparse weight representation.
    """
    n = len(x)
    mean_x = x.mean()
    
    # Compute denominator
    denom = np.sum((x - mean_x) ** 2)
    if denom == 0:
        return 1.0  # Undefined case, return neutral Moran's I
    
    # Compute numerator
    num = 0.0
    for idx in range(len(weights_i)):
        i = weights_i[idx]
        j = weights_j[idx]
        w = weights_data[idx]
        num += w * (x[i] - mean_x) * (x[j] - mean_x)
    
    # Get sum of weights
    s0 = np.sum(weights_data)
    
    # Compute Moran's I
    return (n / s0) * (num / denom)

def fast_morans_I(array, param=None):
    """
    Optimized Moran's I computation using libpysal weights:
    - Caches weights to avoid recomputation for same image size.
    - Uses Numba for accelerated computation.
    - Uses a raveled 1D image vector for fast indexing.
    """
    h, w = array.shape
    image_vector = array.ravel().astype(np.float64)

    # Create or get cached weights using a shape-based key
    if not hasattr(fast_morans_I, 'weights_cache'):
        fast_morans_I.weights_cache = {}
    
    shape_key = (h, w)
    if shape_key not in fast_morans_I.weights_cache:
        # Generate spatial weight matrix using libpysal
        w = libpysal.weights.lat2W(h, w)
        
        # Convert to arrays for Numba processing
        weights_i, weights_j, weights_data = [], [], []
        
        for i, neighbors in w.neighbors.items():
            for j in neighbors:
                weights_i.append(i)
                weights_j.append(j)
                weights_data.append(w.weights[i][w.neighbors[i].index(j)])  # Use actual weight
        
        fast_morans_I.weights_cache[shape_key] = (
            np.array(weights_i),
            np.array(weights_j),
            np.array(weights_data)
        )

    # Retrieve cached weights
    weights_i, weights_j, weights_data = fast_morans_I.weights_cache[shape_key]

    # Compute Moran's I
    return compute_morans_I_numba(image_vector, weights_i, weights_j, weights_data)


@jit(nopython=True)
def compute_gearys_c_numba(x, weights_i, weights_j, weights_data):
    """
    Compute Geary's C using Numba with weights from libpysal:
    - compiles the core computation to machine code; 
    - eliminates Python's interpretation overhead (important for nested for loops).
    """
    n = len(x)
    mean = x.mean()
    
    # Compute denominator
    denom = np.sum((x - mean) ** 2)
    if denom == 0:
        return 1.0
    
    # Compute numerator
    num = 0.0
    for idx in range(len(weights_i)): #Converts libpysal's weights matrix to three separate arrays (weights_i, weights_j, weights_data)
        i = weights_i[idx]
        j = weights_j[idx]
        w = weights_data[idx]
        num += w * ((x[i] - x[j]) ** 2)
    
    # Get sum of weights
    s0 = np.sum(weights_data)
    
    # Compute Geary's C
    return ((n - 1) * num) / (2 * s0 * denom)

def fast_gearys_C(array, param=None):
    """
    Optimized version of Geary's C computation using libpysal weights:
    - avoids recreating the weights for images of the same size using a cached version of them;
    - manually implements the Geary's C formula using Numba JIT compilation;
    - uses contiguous arrays and direct indexing;
    results in a dramatic speed improvement compared to standard library calculations (33.4 > 0.0 sec per image 512x512).
    """
    h, w = array.shape
    image_vector = array.ravel().astype(np.float64)
    
    # Create or get cached weights using a shape-based key
    if not hasattr(fast_gearys_C, 'weights_cache'):
        fast_gearys_C.weights_cache = {}
    
    shape_key = (h, w)
    if shape_key not in fast_gearys_C.weights_cache:
        # Use libpysal's weights
        w = libpysal.weights.lat2W(h, w)
        
        # Convert to arrays for Numba
        weights_i = []
        weights_j = []
        weights_data = []
        
        for i, neighbors in w.neighbors.items():
            for j in neighbors:
                weights_i.append(i)
                weights_j.append(j)
                weights_data.append(1.0)  # Binary weights
                
        fast_gearys_C.weights_cache[shape_key] = (
            np.array(weights_i),
            np.array(weights_j),
            np.array(weights_data)
        )
    
    weights_i, weights_j, weights_data = fast_gearys_C.weights_cache[shape_key]
    
    return compute_gearys_c_numba(
        image_vector,
        weights_i,
        weights_j,
        weights_data
    )


########## LOCAL MEASURES ##########
# These methods are used to compute local spatial measures of image patches centered on each pixel.
# They are designed to return a score between 0 and 1 for each pixel.

def local_entropy(window, param=None):
    """
    Compute normalized entropy of a 2D window using histogram binning.
    A value of 1 indicates maximum entropy (uniform distribution), and 0 indicates minimum entropy (all values in one bin).
    By default we use 4 bins, but this can be adjusted using the `bins` key in param.

    Args:
        window (np.ndarray): 2D window array.
        bins (int): Number of histogram bins.
    
    Returns:
        float: Entropy in [0, 1], representing the randomness of pixel distribution in the window.
    """
    vmin, vmax = 0, 1

    if np.isclose(vmin, vmax):
        return 0.0  # No variation → zero entropy

    bins = param["bins"] if param else 4
    hist, _ = np.histogram(window, bins=bins, range=(vmin, vmax), density=False)
    hist = hist.astype(np.float32)

    if hist.sum() == 0:
        return 0.0

    probs = hist / hist.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    return entropy / np.log2(bins)  # Normalize to [0, 1]


def local_eds(window, param=None):
    """
    Compute local edge density score (EDS) for a 2D window.
    A valued of 1 indicates maximum edge density (all pixels are edges), and 0 indicates minimum edge density (no edges).
    
    Args:
        window (np.ndarray): 2D window array.
        threshold (float): Edge threshold (relative to max gradient).
    
    Returns:
        float: EDS in [0, 1], ratio of edge pixels in the window.
    """
    # Compute Sobel gradients
    gx = sobel(window, axis=0, mode='reflect')
    gy = sobel(window, axis=1, mode='reflect')
    grad_mag = np.hypot(gx, gy)

    # # Normalize gradient (optional, makes threshold relative)
    # max_val = grad_mag.max()
    # if max_val == 0:
    #     return 0.0  # flat window, no edges

    # grad_mag /= max_val

    # Threshold to detect edges
    threshold = param["threshold"] if param else 0.2
    edge_pixels = grad_mag > threshold

    # Compute density
    eds = np.sum(edge_pixels) / edge_pixels.size
    return eds

def local_moran(window, param=None):
    """
    Compute local Moran's I score for a 2D array.
    A value of 1 indicates maximum spatial autocorrelation (clustering), and 0 indicates minimum spatial autocorrelation (random noise).
    Moran'S I can also be negative (up to -1, where -1 indicates perfect negative spatial autocorrelation, i.e. checkerboard-like pattern).
    We cap negative values to 0 to ensure a measure between 0 and 1.

    Args:
        window (np.ndarray): 2D window patch.
        param (dict): Optional parameters.
    
    Returns:
        float: Moran's I score in [0, 1].
    """
    return max(0, fast_morans_I(window))
    



