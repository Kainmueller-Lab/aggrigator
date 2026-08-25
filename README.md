# Aggrigator 🐊

**Aggrigator** is a lightweight Python library for uncertainty aggregation in deep learning workflows.  
Whether you're working with segmentation maps or just want to summarize per-pixel uncertainties — Aggrigator gives you a powerful and flexible toolbox to make sense of it all.

With a clean API and built-in strategies, you can easily:
- Reduce pixelwise uncertainty maps to scalar scores for evaluation or ranking.
- Apply patch-based, class-specific, or thresholded aggregation.
- Incorporate spatial correlation metrics like Moran's I or Geary’s C.
- Compare strategies side-by-side with summaries and plots.

Designed to be modular, explainable, and research-friendly.  
Use it out of the box, or extend it with your own aggregation logic!

## Installation

To install Aggrigator, first make sure you have ``python>=3.10`` installed in your environment. To install the latest release from PyPI, simply run:
```bash
pip install aggrigator
```
To install the development version, clone the repository and navigate inside the directory, then run the following command:

```bash
pip install .
```

Now you can import the library in your python code with:

```python
import aggrigator
```

## Original publication

The repository was released as part of the publication: Guarino* VE, Winklmayr* C, Franzen* J, Rumberger JL, Pfeuffer M, Greven S, Maier-Hein K, Kainmueller D, Karg C, Lüth CT. [**Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance.**](https://openaccess.thecvf.com/content/CVPR2026/papers/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.pdf) InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 (pp. 13145-13156).

This paper was published at CVPR 2026 (highlight) and investigates Uncertainty Quantification (UQ), which can improve the reliability of automated segmentation but produces pixel-wise uncertainty maps. In turn, these must be aggregated into image-level scores to be used in downstream tasks such as out-of-distribution (OoD) and failure detection. However, aggregation strategies remain underexplored, with Global Average (AVG) commonly used despite ignoring spatial structure in uncertainty maps. To address this, we systematically analyze existing aggregation strategies, introduce spatially informed alternatives, and benchmark them across 10 diverse datasets. We show that spatial aggregators often improve performance, though effectiveness depends on dataset characteristics. Based on these results, we propose a meta-aggregator that combines multiple strategies and performs robustly across datasets.

To recreate the experiment of the paper please refer to https://github.com/Kainmueller-Lab/aggrigator_experiments

For the arxiv version including the **appendix** please refer to https://arxiv.org/pdf/2603.29941v1

## How to use

Check out the interactive [example_notebook.ipynb](example_notebook.ipynb) to see **Aggrigator** in action.  
You’ll learn how to:

- Generate and visualize uncertainty maps.  
- Apply and compare aggregation strategies.  
- Use class-aware masks for targeted aggregation.
