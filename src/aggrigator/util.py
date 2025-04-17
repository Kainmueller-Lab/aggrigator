import numpy as np
import os
from scipy.ndimage import convolve, binary_erosion

def get_id_mask(mask, id):
    return np.where(mask==id, 1, 0)

def get_id_mask_interior(mask, id, nbhood='full', thickness=1):
    """
    Returns a binary mask of the interior pixels belonging to a given 'id' in 'mask'. 
    The interior pixels are eroded by 'thickness' using the specified neighborhood shape.
    
    Args:
        mask: A numpy array representing labeled regions.
        id: The label whose interior we are extracting.
        nbhood: Neighborhood shape, either 'full' or 'cross'.
        thickness: Number of times to erode (shrink) the mask.
    Returns: 
        A numpy array (binary mask) indicating the interior pixels for the given 'id'.
    """
    # Define structure for neighborhood
    if nbhood == 'full':
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=bool)

    id_mask = get_id_mask(mask, id)
    if thickness == 0:
        return id_mask
    eroded = binary_erosion(id_mask, structure=structure, iterations=thickness, border_value=0)
    return eroded.astype(np.uint8)

def get_id_mask_boundary(mask, id, nbhood='full', thickness=1):
    id_mask = get_id_mask(mask, id)
    interior_mask = get_id_mask_interior(mask, id, nbhood, thickness=thickness)
    return id_mask - interior_mask

def get_fg_ratio(mask):
    """
    Computes the ratio of nonzero entries to total entries in the mask.
    """
    return np.count_nonzero(mask) / mask.size

def generate_disk_mask(array_size, radius, center):
    """
    Generates an array with a disk of white pixels (1s) centered at the specified position.
    
    :param array_size: Tuple of the shape of the array (height, width).
    :param radius: Radius of the disk.
    :param center: Tuple of the disk's center (y, x).
    :return: A numpy array of the specified size with a disk of 1s and 0s elsewhere.
    """
    # Create a coordinate grid
    y, x = np.ogrid[:array_size[0], :array_size[1]]

    # Calculate the squared distance from the center
    distance_squared = (y - center[0])**2 + (x - center[1])**2

    # Create the mask for the disk
    mask = distance_squared <= radius**2 

    # Create the result array with 1s inside the disk and 0s elsewhere
    result = np.zeros(array_size, dtype=np.uint8)
    result[mask] = 1

    return result

def generate_disk_bdry_mask(array_size, radius, center, thickness):
    """
    Generates an array with a disk of white pixels (1s) centered at the specified position.
    
    :param array_size: Tuple of the shape of the array (height, width).
    :param radius: Radius of the disk.
    :param center: Tuple of the disk's center (y, x).
    :return: A numpy array of the specified size with a disk of 1s and 0s elsewhere.
    """
    # Create a coordinate grid
    y, x = np.ogrid[:array_size[0], :array_size[1]]

    # Calculate the squared distance from the center
    distance_squared = (y - center[0])**2 + (x - center[1])**2

    # Create the mask for the disk
    mask = (distance_squared <= radius**2 ) & (distance_squared >= (radius-thickness)**2)

    # Create the result array with 1s inside the disk and 0s elsewhere
    result = np.zeros(array_size, dtype=np.uint8)
    result[mask] = 1

    return result

def get_num_cpus():
    try:
        return os.cpu_count()
    except NotImplementedError:
        return 1