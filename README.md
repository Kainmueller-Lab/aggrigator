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

To install the aggrigator, clone the repository, navigate inside the directory and run the following command:

```bash
pip install -e .
```

now you can import the library in your python code with:

```python
import aggrigator
```

## Testing

To run the tests locally, navigate inside the aggrigator directory and first install the dev dependencies if needed:

```bash
pip install .[dev]
```

and then run the tests with:

```bash
pytest tests
```

## Try it out yourself

Check out the interactive [example_notebook.ipynb](example_notebook.ipynb) to see **Aggrigator** in action.  
You’ll learn how to:

- ✅ Generate and visualize uncertainty maps.  
- ⚙️ Apply and compare aggregation strategies.  
- 🧠 Use class-aware masks for targeted aggregation.