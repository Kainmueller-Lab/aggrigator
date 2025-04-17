import numpy as np
from numba import jit
import libpysal

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

def fast_gearys_C(unc_map, param=None):
    """
    Optimized version of Geary's C computation using libpysal weights:
    - avoids recreating the weights for images of the same size using a cached version of them;
    - manually implements the Geary's C formula using Numba JIT compilation;
    - uses contiguous arrays and direct indexing;
    results in a dramatic speed improvement compared to standard library calculations (33.4 > 0.0 sec per image 512x512).
    """
    h, w = unc_map.array.shape
    image_vector = unc_map.array.ravel().astype(np.float64)
    
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
