import unittest
import numpy as np

from aggrigator.util import get_id_mask, get_id_mask_interior, get_id_mask_boundary

class TestUtil(unittest.TestCase):
    def setUp(self):
        self.unc_map_array = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ])

    def test_full_boundary(self):
        nbhood = 'full'
        id_mask = get_id_mask(self.unc_map_array, 1)
        interior = get_id_mask_interior(self.unc_map_array, 1, nbhood)
        boundary_mask = get_id_mask_boundary(self.unc_map_array, 1, nbhood)

        self.assertEqual(np.all(id_mask == self.unc_map_array), True)
        self.assertEqual(np.all(interior == np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])), True)
        self.assertEqual(np.all(boundary_mask == np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ])), True)

        # Test with thickness=2
        interior_2 = get_id_mask_interior(self.unc_map_array, 1, nbhood, thickness=2)
        boundary_2 = get_id_mask_boundary(self.unc_map_array, 1, nbhood, thickness=2)

        self.assertEqual(np.all(interior_2 == np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])), True)
        self.assertEqual(np.all(boundary_2 == np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ])), True)

    def test_star_boundary(self):
        nbhood = 'star'
        id_mask = get_id_mask(self.unc_map_array, 1)
        interior = get_id_mask_interior(self.unc_map_array, 1, nbhood)
        boundary_mask = get_id_mask_boundary(self.unc_map_array, 1, nbhood)

        self.assertEqual(np.all(id_mask == self.unc_map_array), True)
        self.assertEqual(np.all(interior == np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ])), True)
        self.assertEqual(np.all(boundary_mask == np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ])), True)

        # Test with thickness=2
        interior_2 = get_id_mask_interior(self.unc_map_array, 1, nbhood, thickness=2)
        boundary_2 = get_id_mask_boundary(self.unc_map_array, 1, nbhood, thickness=2)

        self.assertEqual(np.all(interior_2 == np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])), True)
        self.assertEqual(np.all(boundary_2 == np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ])), True)


if __name__ == "__main__":
    unittest.main()
