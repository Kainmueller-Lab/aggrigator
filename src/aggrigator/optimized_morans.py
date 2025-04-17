import numpy as np
from numba import jit
import libpysal

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

def fast_morans_I(unc_map, param=None):
    """
    Optimized Moran's I computation using libpysal weights:
    - Caches weights to avoid recomputation for same image size.
    - Uses Numba for accelerated computation.
    - Uses a raveled 1D image vector for fast indexing.
    """
    h, w = unc_map.array.shape
    image_vector = unc_map.array.ravel().astype(np.float64)

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
