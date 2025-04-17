import unittest
import numpy as np
import numpy.testing as npt

from aggrigator.util import get_fg_ratio
from aggrigator.methods import AggregationMethods as am
from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.datasets import generate_binary_quadrant_array

class TestImageLevelMethods(unittest.TestCase):
    def setUp(self):
        self.array = np.array([1, 2, 3, 4, 5])
        self.array_2d = np.array([[1,2,3,4],
                                  [1,2,3,4],
                                  [1,2,3,4],
                                  [1,2,3,4]])
        self.mask = None
        self.name = "TestMap"
        self.unc_map = UncertaintyMap(array=self.array, mask=self.mask, name=self.name)
        self.unc_map_2d = UncertaintyMap(array=self.array_2d, mask=self.mask, name=self.name)

    def test_mean_aggregation(self):
        """Test mean aggregation method."""
        result = am.mean(self.unc_map)
        self.assertEqual(result, 3.0)

    def test_sum_aggregation(self):
        """Test sum aggregation method."""
        result = am.sum(self.unc_map)
        self.assertEqual(result, 15)

    def test_max_aggregation(self):
        """Test max aggregation method."""
        result = am.max(self.unc_map)
        self.assertEqual(result, 5)
    
    def test_geometric_mean_aggregation(self):
        """Test geometric mean aggregation method."""
        result = am.geometric_mean(self.unc_map)
        self.assertAlmostEqual(result, 120**(1/5), places=8)
        result = am.geometric_mean(self.unc_map_2d)
        self.assertAlmostEqual(result, 24**(1/4), places=8)
    
    def test_harmonic_mean_aggregation(self):
        """Test harmonic mean aggregation method."""
        result = am.harmonic_mean(self.unc_map)
        self.assertAlmostEqual(result, 300/137, places=8)
        result = am.harmonic_mean(self.unc_map_2d)
        self.assertAlmostEqual(result, 48/25, places=8)

    def test_threshold_aggregation(self):
        """Test threshold aggregation method."""
        result = am.above_threshold_mean(self.unc_map, 3.5)
        self.assertEqual(result, 4.5)

    def test_threshold_aggregation(self):
        """Test above threshold sum method."""
        result = am.above_threshold_sum(self.unc_map, 3.5)
        self.assertEqual(result, 9)

    def test_threshold_volume(self):
        """Test threshold volume method."""
        result = am.above_threshold_volume(self.unc_map_2d, 3.5)
        self.assertEqual(result, 4/16)
        result = am.above_threshold_volume(self.unc_map_2d, 2.5)
        self.assertEqual(result, 8/16)

    def test_patch_aggregation(self):
        result = am.patch_aggregation(self.unc_map_2d, 2)
        self.assertEqual(result, 3.5)
        result = am.patch_aggregation(self.unc_map_2d, 3)
        self.assertEqual(result, 3.0)
        result = am.patch_aggregation(self.unc_map_2d, 4)
        self.assertEqual(result, 2.5)


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

class TestClassLevelMethods(unittest.TestCase):
    def setUp(self):
        self.unc_values = np.array([
            [0.4, 0.4, 0.4, 0, 0],
            [0.4, 4, 0.4, 0, 0],
            [0.4, 0.4, 0.4, 1.2, 1.2],
            [0, 0, 1.2, 10.8, 1.2],
            [0, 0, 1.2, 1.2, 1.2],
        ])
        self.mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
        ])
        self.name = "TestMap"
        self.unc_map = UncertaintyMap(array=self.unc_values, mask=self.mask, name=self.name)

    def test_class_mean_aggregation(self):
        self.assertAlmostEqual(am.class_mean(self.unc_map, 1), 0.8)
        self.assertAlmostEqual(am.class_mean(self.unc_map, 2), 2.4)

    # def test_adjusted_class_mean_aggregation(self):
    #     self.assertAlmostEqual(am.class_mean_adjusted(self.unc_map, 1), 2.4)
    #     self.assertAlmostEqual(am.class_mean_adjusted(self.unc_map, 2), 19.2 / np.sqrt(8))

    def test_class_boundary_mean_aggregation(self):
        self.assertAlmostEqual(am.class_boundary_mean(self.unc_map, {'id': 1}), 0.4) # Boundary pixels are all class 1 pixels except center pixel.
        self.assertAlmostEqual(am.class_boundary_mean(self.unc_map, {'id': 2}), 2.4) # Boundary pixels are all class 2 pixels.

    def test_all_class_return(self):
        result = am.class_mean_dict(self.unc_map) # Returns a dictionary of class means per class index.
        expected_keys = self.unc_map.class_indices
        expected_values = np.array([0.0, 0.8, 2.4], dtype=np.float64)
        npt.assert_allclose(list(result.keys()), expected_keys)
        npt.assert_allclose(list(result.values()), expected_values, equal_nan=True)
        
        result = am.class_boundary_mean_dict(self.unc_map)
        expected = np.array([0.0, 0.4, 2.4], dtype=np.float64)
        npt.assert_allclose(list(result.values()), expected, equal_nan=True)

        result = am.class_interior_mean_dict(self.unc_map)
        expected = np.array([np.nan, 4.0, np.nan], dtype=np.float64)
        npt.assert_allclose(list(result.values()), expected, equal_nan=True)


class TestFGRatio(unittest.TestCase):
    def setUp(self):
        self.unc_values = np.array([
            [0.4, 0.4, 0.4, 0, 0],
            [0.4, 4, 0.4, 0, 0],
            [0.4, 0.4, 0.4, 1.2, 1.2],
            [0, 0, 1.2, 10.8, 1.2],
            [0, 0, 1.2, 1.2, 1.2],
        ])
        self.mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
        ])
        self.name = "TestMap"
        self.unc_map = UncertaintyMap(array=self.unc_values, mask=self.mask, name=self.name)

    def test_fg_ratio(self):
        result = get_fg_ratio(self.unc_map.mask)
        expected = 17/25
        npt.assert_almost_equal(result, expected)

    def test_aqa_with_fg_ratio(self):
        unc_values = np.array([
            [0.5, 0.6],
            [0.3, 0.8],
        ])
        mask = np.array([
            [1, 0],
            [1, 0],
        ])
        unc_map = UncertaintyMap(array=unc_values, mask=mask, name="")

        result = am.above_quantile_mean_fg_ratio(unc_map)
        expected = 0.7
        npt.assert_almost_equal(result, expected)

    def test_aqa_with_fg_ratio_2(self):
        result = am.above_quantile_mean_fg_ratio(self.unc_map)
        expected = (4+8*0.4 + 10.8+7*1.2)/17
        npt.assert_almost_equal(result, expected)
 

    


class TestAggregatedClassMean(unittest.TestCase):
    def setUp(self):
        self.unc_values = np.array([
            [0.4, 0.4, 0.4, 0, 0],
            [0.4, 4, 0.4, 0, 0],
            [0.4, 0.4, 0.4, 1.2, 1.2],
            [0, 0, 1.2, 10.8, 1.2],
            [0, 0, 1.2, 1.2, 1.2],
        ])
        self.mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 2, 2],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 2],
        ])
        self.name = "TestMap"
        self.unc_map = UncertaintyMap(array=self.unc_values, mask=self.mask, name=self.name)

    def test_class_mean_w_equal_weights(self):
        result = am.class_mean_w_equal_weights(self.unc_map)
        expected = 0.8 * 0.5 + 2.4 *0.5
        self.assertEqual(result, expected)

    def test_class_mean_weighted_by_occurrence(self):
        result = am.class_mean_weighted_by_occurrence(self.unc_map)
        expected = 0.8 * 9/17 + 2.4 * 8/17
        self.assertEqual(result, expected)

    def test_class_mean_w_custom_weights(self):
        weights = {1: 0.3, 2: 0.7}
        result = am.class_mean_w_custom_weights(self.unc_map, weights)
        expected = 0.8 * 0.3 + 2.4 * 0.7
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
