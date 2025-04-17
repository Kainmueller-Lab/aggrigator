import os
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec

from aggrigator.util import get_num_cpus

class AggregationSummary:
    """Toolkit to manage and apply aggregation methods."""
    def __init__(self, strategies, name="summary", num_cpus=1):
        self.strategies = strategies if isinstance(strategies, list) else strategies._value_
        self.name = name
        self.num_cpus = min(num_cpus, get_num_cpus())
    
    def apply_methods(self, unc_maps, save_to_excel=False, do_plot=False, max_value=1.0):
        self.max_value = max_value # Maximum value for uncertainty map plot. Pixels with this value will be shown white.

        def param_string(param):
            return f"_{param}" if not param is None else ""
        
        results_dict = {
            "Name" : [f"{method.__name__}{param_string(param)}" for method, param in self.strategies]
        }

        if self.num_cpus > 1:
            print(f"Using {self.num_cpus} CPUs.")
            eval_func = lambda unc_map: [method(unc_map, param) for method, param in self.strategies]
            eval_results = Parallel(n_jobs=self.num_cpus)(delayed(eval_func)(unc_map) for unc_map in unc_maps)
        else:
            eval_results = [[method(unc_map, param) for method, param in self.strategies] for unc_map in unc_maps]

        # Store results
        results_dict.update({unc_map.name : result for unc_map, result in zip(unc_maps, eval_results)})
        results = pd.DataFrame(results_dict)
        if save_to_excel:
            results.to_excel(os.path.join("output", f"{self.name}_results.xlsx"))
        if do_plot:
            self.plot_results(unc_maps, results)
        return results
    
    def plot_results(self, unc_maps, results_df):  
        num_cols = len(results_df.columns)
        masks_provided = all([unc_map.mask_provided for unc_map in unc_maps])
        num_rows = 3 if masks_provided else 2

        def format_img_axes(fig):
            for i, ax in enumerate(fig.axes):
                ax.axis('off') 
                #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
                #ax.tick_params(labelbottom=False, labelleft=False)

        fig = plt.figure(figsize=(12, 4))

        gs = GridSpec(num_rows, num_cols, figure=fig)

        # Table subplot
        ax0 = fig.add_subplot(gs[0, :])
        # Format floats dynamically for table display
        cell_text = results_df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).values
        # Create the table
        unc_map_names = [unc_map.name for unc_map in unc_maps]
        col_labels = [""] + unc_map_names
        table = ax0.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
        for (c, r) in table.get_celld().keys():
            if c*r == 0 and not c+r == 0:
                table[(c,r)].set_facecolor("lightgrey")
        # Style the table
        #table.auto_set_font_size(False)
        #table.set_fontsize(10)
        #table.auto_set_column_width(col=list(range(len(results_df.columns))))  # Adjust column width
        # Remove all cell edges
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("white")  # Set all edges to white (invisible)

        # Heatmap subplots
        img_axes = [None for _ in range(num_cols)]
        for idx in range(num_cols):
            img_axes[idx] = fig.add_subplot(gs[1, idx])
            if idx > 0:
                img_axes[idx].imshow(unc_maps[idx-1].array.squeeze(), cmap="viridis", vmin=0, vmax=self.max_value)
            else:
                img_axes[idx].text(0.5, 0.5, "Uncertainty Map", va="center", ha="center")

        if masks_provided:
            # Mask subplots
            mask_axes = [None for _ in range(num_cols)]
            for idx in range(num_cols):
                mask_axes[idx] = fig.add_subplot(gs[2, idx])
                if idx > 0:
                    mask_axes[idx].imshow(unc_maps[idx-1].mask.squeeze(), cmap="gray")
                else:
                    mask_axes[idx].text(0.5, 0.5, "Predicted Mask", va="center", ha="center")

        #fig.suptitle("Uncertainty Aggregations")
        format_img_axes(fig)

        plt.show()