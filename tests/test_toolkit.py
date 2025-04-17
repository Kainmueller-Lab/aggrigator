import unittest
import numpy as np

from torchvision import datasets, transforms

from aggrigator.methods import AggregationMethods as am
from aggrigator.summary import AggregationSummary
from aggrigator.uncertainty_maps import UncertaintyMap

class TestSummary(unittest.TestCase):
    def setUp(self):
        self.mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    def test_uncertainty_map_classes(self):
        array = self.mnist_dataset[0][0].numpy()
        mask = (self.mnist_dataset[0][0].numpy() > 0).astype(int)
        name = f"UncMap{0}"

        unc_map = UncertaintyMap(array=array, mask=mask, name=name)
        self.assertEqual(unc_map.class_indices[0], 0)
        self.assertEqual(unc_map.class_indices[1], 1)

    def test_uncertainty_map_volumes(self):
        array = self.mnist_dataset[0][0].numpy()
        mask = (self.mnist_dataset[0][0].numpy() > 0).astype(int)
        name = f"UncMap{0}"

        unc_map = UncertaintyMap(array=array, mask=mask, name=name)
        self.assertDictEqual(unc_map.class_volumes, {0: 618, 1: 166})

    def test_results(self):
        N = 5
        arrays = [self.mnist_dataset[index][0].numpy() for index in range(N)]
        masks = [(self.mnist_dataset[index][0].numpy() > 0).astype(int) for index in range(N)]
        names = [f"UncMap{index}" for index in range(N)]
        unc_maps = [UncertaintyMap(array=array, mask=mask, name=name) for array, mask, name in zip(arrays, masks, names)]

        strategy_list = [(am.mean, None),
                         (am.sum, None)]
        summary = AggregationSummary(strategy_list)
        results_df = summary.apply_methods(unc_maps)

        true_mean_list = ['mean', np.float32(0.13768007), np.float32(0.15553722), np.float32(0.097253904), np.float32(0.08570928), np.float32(0.11611645)]
        for result, true in zip(results_df.loc[0].to_list(), true_mean_list):
            self.assertAlmostEqual(result, true, delta=1e-7)

        true_sum_list = ['sum', np.float64(107.94117754790932), np.float64(121.94117757305503), np.float64(76.247059895657), np.float64(67.1960789039731), np.float64(91.035294912755499)]
        for result, true in zip(results_df.loc[1].to_list(), true_sum_list):
            self.assertAlmostEqual(result, true, delta=1e-7)


if __name__ == "__main__":
    unittest.main()

