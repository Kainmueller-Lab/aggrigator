import unittest
import numpy as np

from aggrigator.spatial import local_entropy, local_eds, local_moran, fast_morans_I
from aggrigator.datasets import generate_binary_quadrant_array
from aggrigator.methods import AggregationMethods as am
from aggrigator.uncertainty_maps import UncertaintyMap

np.random.seed(123)


class TestSpatialCorrelationMethods(unittest.TestCase):
    def setUp(self):
        N = 100
        random_array = np.random.random((N, N))
        checkerboard_array = np.indices((N, N)).sum(axis=0) % 2
        clustered_array = generate_binary_quadrant_array(N)
        self.random_uc_map = UncertaintyMap(array=random_array, mask=None, name="")
        self.checkerboard_uc_map = UncertaintyMap(array=checkerboard_array, mask=None, name="")
        self.clustered_uc_map = UncertaintyMap(array=clustered_array, mask=None, name="")

    def test_morans_I(self):
        '''
        I > 0 → Positive spatial autocorrelation (clusters of similar intensity)
        I < 0 → Negative autocorrelation (checkerboard-like pattern)
        I ≈ 0 → No spatial correlation (random noise)
        '''
        result = am.morans_I(self.random_uc_map)
        self.assertAlmostEqual(result, 0.0, places=1)
        result = am.morans_I(self.checkerboard_uc_map)
        self.assertAlmostEqual(result, -1.0, places=1)
        result = am.morans_I(self.clustered_uc_map)
        self.assertAlmostEqual(result, 0.98, places=1)

    def test_gearys_C(self):
        '''
        C=1 → No spatial autocorrelation (random pattern).
        C<1 → Positive spatial autocorrelation (clusters of similar values).
        C>1 → Negative spatial autocorrelation (checkerboard-like pattern).
        '''
        result = am.gearys_C(self.random_uc_map)
        self.assertAlmostEqual(result, 1.0, places=1)
        result = am.gearys_C(self.checkerboard_uc_map)
        self.assertAlmostEqual(result, 2.0, places=1)
        result = am.gearys_C(self.clustered_uc_map)
        self.assertAlmostEqual(result, 0.0, places=1)


class TestLocalSpatialMeasures(unittest.TestCase):
    def setUp(self):
        # Create example windows for testing
        self.uniform_window = np.ones((5, 5))  # All same values → entropy = 0
        self.checkerboard_window = np.array([[0, 1, 0, 1, 0],
                                             [1, 0, 1, 0, 1],
                                             [0, 1, 0, 1, 0],
                                             [1, 0, 1, 0, 1],
                                             [0, 1, 0, 1, 0]])
        self.all_bins_window = np.array([[0.1, 0.1, 0.1, 0.1, 0.1],
                                         [0.3, 0.3, 0.3, 0.3, 0.3],
                                         [0.5, 0.5, 0.5, 0.5, 0.5],
                                         [0.7, 0.7, 0.7, 0.7, 0.7],
                                         [0.9, 0.9, 0.9, 0.9, 0.9]])
        self.random_window = np.random.rand(5, 5)  # High entropy expected
        self.edge_window = np.zeros((5, 5))
        self.edge_window[2:] = 1  # Sharp horizontal edge

    # ENTROPY
    def test_local_entropy_uniform(self):
        result = local_entropy(self.uniform_window)
        self.assertAlmostEqual(result, 0.0, msg="Entropy of uniform window should be 0")

    def test_local_entropy_random(self):
        result = local_entropy(self.random_window)
        self.assertGreater(result, 0.5, msg="Entropy of random window should be high")

    def test_local_entropy_bins(self):
        result = local_entropy(self.all_bins_window, {"bins": 5})
        self.assertAlmostEqual(result, 1.0, msg="Entropy of uniform window should be 1")

    # EDGE DENSITY SCORE
    def test_local_eds_no_edges(self):
        result = local_eds(self.uniform_window)
        self.assertAlmostEqual(result, 0.0, msg="EDS of flat window should be 0")

    def test_local_eds_edge_present(self):
        result = local_eds(self.edge_window, {"threshold": 0.1})
        self.assertGreater(result, 0.0, msg="EDS should detect edges")

    # MORAN'S I
    def test_local_moran_random(self):
        result = local_moran(self.random_window)
        self.assertAlmostEqual(result, 0.0, msg="Moran's I of random window should be 0")

    def test_local_moran_uniform(self):
        result = local_moran(self.uniform_window)
        self.assertAlmostEqual(result, 1.0, msg="Moran's I of uniform window should be 1")

    def test_local_moran_checkerboard(self):
        true_morans = fast_morans_I(self.checkerboard_window)
        self.assertAlmostEqual(true_morans, -1.0, msg="Actual Moran's I of checkerboard window should be -1")
        adapted_morans = local_moran(self.checkerboard_window)
        self.assertAlmostEqual(adapted_morans, 0.0, msg="Adapted Moran's I of checkerboard window should be 0")


if __name__ == "__main__":
    unittest.main()
