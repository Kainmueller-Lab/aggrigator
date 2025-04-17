# AggroUQ

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