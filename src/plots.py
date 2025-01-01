import numpy as np
import pandas as pd 

import seaborn as sns

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

import PyComplexHeatmap as pch # Complex Heatmaps in Python

from typing import Union, Optional, Tuple, List

from src import tests

#################### Global variables ####################
# Colormap for the questvar package
stat_colors = [
    "#29335c", "#7e8cc6", "#565d61",
    "#ff8020", "#eeeeee", "#70161e",
    "#d06770"
]

# Define the status palette
status_palette = {
    "Excluded": "#565d61",
    "Different": "#70161e",
    "Unexplained": "#99999950",
    "Equivalent": "#29335c",    
}

#################### Utility functions ####################
def hex_to_rgb(
        hex_code: str,
    ):
    """
        Converts a hex color code to RGB
        
        Args:
            hex_code (str) Hex color code
        
        Returns:
            tuple: RGB color code
    """
    hex_code = hex_code.lstrip('#').rstrip(";")
    lv = len(hex_code)
    return tuple(
        int(hex_code[i:i + lv // 3], 16) 
        for i in range(0, lv, lv // 3)
    )

def pick_color_based_on_background(
        bgColor: str,
        lightColor: str = "#FFFFFF",
        darkColor: str = "#000000",
        hex: bool = False,
        rgb: bool = False,
        uicolor: bool = False,
    ):
    """
        Picks a light or dark color based on the background color
        Built from and answer on StackOverflow:
            https://stackoverflow.com/a/76275241

        Args:
            bgColor (str): Background color
            lightColor (str): Light color
            darkColor (str): Dark color
            hex (bool): If the background color is in hex format
            rgb (bool): If the background color is in RGB format
            uicolor (bool): If the background color is in UIColor format

        Returns:
            str: Light or dark color based on the background color
    """
    pass
    if hex:
        color = bgColor.lstrip("#").rstrip(";")
        r, g, b = hex_to_rgb(color)
        uicolors = [r/255, g/255, b/255]
    elif rgb:
        r, g, b = bgColor
        uicolors = [r/255, g/255, b/255]
    elif uicolor:
        uicolors = bgColor
    else:
        raise ValueError(
            """Please turn on one of the color modes relevant to bgColor passed 
            Options: hex, rgb, uicolor."""
        )

    adjusted = []
    for col in uicolors:
        col2 = col
        if col <= 0.03928:
            col2 = col/12.92
        
        col2 = ((col2+0.055)/1.055)**2.4
        adjusted.append(col2)

    L = 0.2126 * adjusted[0] + 0.7152 * adjusted[1] + 0.0722 * adjusted[2]

    return darkColor if L > 0.179 else lightColor

def save_figures(
        fig_obj,
        filename: str,
        filepath: str = '',
        fileformat: list[str] = ['png', 'svg', 'pdf'],
        dpi: int = 300,
        transparent: bool = True
    ):
    """
        Saves the plot in multiple file formats in the specified path

        Args:
            fig_obj (matplotlib.figure.Figure): Plot object
            filename (str): Name of the file to be saved
            filepath (str): Path to save the file
            fileformat (list[str]): List of file formats to save the plot in
            dpi (int): Dots per inch
            transparent (bool): If the background should be transparent

        Returns:
            None
    """
    for i in fileformat:
        fig_obj.savefig(
            filepath + "/" + i + "/" + filename + '.' + i,
            format=i,
            dpi=dpi, 
            transparent=transparent,
            bbox_inches='tight',
            pad_inches=0.01
        )

################ Sequence Visualization functions ################
# Source: alphamap/sequence_plot.py
# Most of the main logic is the same but modified to work well with my project

def format_uniprot_annotation(
        uniprot_ann: pd.DataFrame, 
        uniprot_feature_dict: dict, 
        feature_col: str = 'feature',
        note_col: str = 'note',
        annotation_col: str = 'annotation'
    ) -> pd.DataFrame:
    """
        Function to format uniprot annotation for plotting.

        Args:
            uniprot_ann (pd.DataFrame): Formatted uniprot annotations from alphamap.
            uniprot_feature_dict (dict): Uniprot feature dictionary defined by alphamap.
            feature_col (str): Column name for uniprot feature. Default is 'feature'.
            note_col (str): Column name for uniprot note. Default is 'note'.
        Returns:
            pd.DataFrame: Uniprot annotation with a combined structure entry for helix, strand and turn.

    """
    
    uniprot = uniprot_ann.copy(deep=True)
    uniprot.loc[uniprot[feature_col] == "HELIX", note_col] = "Helix"
    uniprot.loc[uniprot[feature_col] == "STRAND", note_col] = "Beta strand"
    uniprot.loc[uniprot[feature_col] == "TURN", note_col] = "Turn"
    uniprot.loc[uniprot[feature_col].isin(["HELIX","STRAND","TURN"]), feature_col] = "STRUCTURE"

    uniprot_feature_dict_rev = {v: k for k, v in uniprot_feature_dict.items()}

    uniprot[annotation_col] = uniprot[note_col]
    uniprot.loc[uniprot[annotation_col].isnull(), annotation_col] = uniprot[feature_col]
    uniprot = uniprot.replace({annotation_col: uniprot_feature_dict_rev})
    
    return uniprot

#################### Visualizations functions ####################

# Plots a color palette with added functionality
def color_palette(
        pal: Union[list, dict],
        size: int = 1, 
        name: str = "default colors",
        save: bool = False,
        filename: str = 'default_colors_pal',
        fileformats: list[str] = ['png', 'svg', 'pdf'],
        filepath: str = '',
        transparent: bool = False,
        dpi: int = 100
    ):    

    """
        Plots a color palette and saves it in multiple file formats if needed

        Args:
            pal (Union[list, dict]): List of colors or dictionary of colors with labels
            size (int): Size of the plot
            name (str): Name of the color palette
            save (bool): If the plot should be saved
            filename (str): Name of the file to be saved
            fileformats (list[str]): List of file formats to save the plot in
            filepath (str): Path to save the file
            vectorized (bool): If the plot should be saved in vector format
            dpi (int): Dots per inch

        Returns:
            None

    """
    # Check if the palette is a dictionary
    if isinstance(pal, dict):
        # Get the labels
        labels = list(pal.keys())
        # Get the colors
        pal = list(pal.values())
    else:
        # Check if the palette is a list
        if not isinstance(pal, list):
            raise ValueError(
                "The palette should be either a list or a dictionary"
            )
        # Get the labels
        labels = None

    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mcl.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    if labels is not None:
        # Set the labels as text inside the boxes diagonally
        for i, label in enumerate(labels):
            ax.text(
                i, 0, label, 
                rotation=45, 
                rotation_mode="anchor",
                ha="center", 
                va="center", 
                # Color is picked based on the background color
                color=pick_color_based_on_background(
                    pal[i],
                    hex=True,
                ),
                fontsize=11,
            )
    # Ensure nice border between colors
    ax.set_xticklabels(["" for _ in range(n)])
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(
        f"Palette for {name}", 
        fontsize=14,
        loc="left"
    )
    # Remove all axes
    ax.axis('off')

    # Save the plot if needed
    if save:
        save_figures(
            f,
            filename,
            filepath,
            fileformats,
            dpi,
            transparent
        )

    plt.show()

def confidence_ellipse(
        x: np.ndarray,
        y: np.ndarray,
        ax: plt.Axes,
        n_std: float = 3.0,
        facecolor: str = 'none',
        **kwargs
    ) -> Ellipse:
    """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Args:
            x, y : array-like, shape (n, )
                Input data.

            ax : matplotlib.axes.Axes
                The axes object to draw the ellipse into.

            n_std : float
                The number of standard deviations to determine the ellipse's radiuses.

            **kwargs
                Forwarded to `~matplotlib.patches.Ellipse`

        Returns:
            matplotlib.patches.Ellipse
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# QC Visualization functions
def grouped_samples_density(
        ## Data options
        data: pd.DataFrame,
        group_sample_dict: dict,
        take_log2: bool = False,
        ## Plot options
        figsize: Tuple[int, int] = (8, 4),
        color_dict: dict = None,
        title: str = "Samples Distribution grouped by Condition",
        legend_title: str = "Group",
        # sample_size: int = 750, # Number of samples to take
        gridsize: int = 50, # Number of points in the grid
        ## Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'Initial_Grouped_Density',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False
    ):
    """
    
    """
    if take_log2:
        plot_data = np.log2(data)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (grp, cols) in enumerate(group_sample_dict.items()):
        for _, col in enumerate(cols):
            sns.kdeplot(
                plot_data[col].dropna(),#.sample(sample_size),
                ax=ax,
                color=color_dict[grp],
                gridsize=gridsize,
                alpha=0.7,
                rasterized=True,
            )
    ax.set_title(title, fontsize=14, loc="left", pad=10)
    ax.set_xlabel("Log2 Intensity")
    ax.set_ylabel("Density")

    # Create Legend
    legend_elements = [
        Line2D([0], [0], color=clr, label=grp)
        for grp, clr in color_dict.items()
    ]

    ax.legend(
        handles=legend_elements,
        title=legend_title,
        loc="upper right",
        fontsize=10,
        title_fontsize=10,
        frameon=False,
    )

    ax.grid(
        axis="both",
        which="major",
        color="lightgrey",
        linestyle="--",
        linewidth=.5,
        alpha=0.75,
    )

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
    if dont_show:
        plt.close(fig)
    else:
        return fig


def perturbation_lineplot(
        data: pd.DataFrame,
        protein_subset: list,
        perturb_dict: dict,
        column_order: list,
        
        ## Plot options
        figsize: Tuple[int, int] = (12, 4),
        color_dict: dict = {"Non-Perturbed": "#139593", "Perturbed": "#fca311"},
        title: str = "Comparing Peptides Intensities Across Samples with -/+ Perturbation",
        legend_title: str = "Perturbation",
        
        ## Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'PerturbationStatus_Lineplot',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False
    ):
    """
    """
     
    subset = data.loc[protein_subset].index.to_frame().set_index("Protein")
    subset["Perturbation"] = "Non-Perturbed"

    new_data = pd.DataFrame()
    for k, v in perturb_dict.items():
        if k in protein_subset:
            protein_data = subset.loc[k]
            protein_data.iloc[v["peptides"], 1] = "Perturbed"
            new_data = pd.concat([new_data, protein_data])

    if new_data.empty:
       plot_data = data.loc[protein_subset].reset_index().melt(
            id_vars=["Protein", "Mod.Peptide"],
            var_name="Sample",
            value_name="Intensity"
        ).set_index("Protein").assign(
            **{"log2(Intensity)": lambda x: np.log2(x["Intensity"] + 1)}
        ).assign(Perturbation="Non-Perturbed")
    else:
        plot_data = pd.concat(
            [
                new_data.reset_index().set_index(["Protein", "Mod.Peptide"]),
                data.loc[protein_subset]
            ],
            axis=1
        ).reset_index().melt(
            id_vars=["Protein", "Mod.Peptide", "Perturbation"],
            var_name="Sample",
            value_name="Intensity"
        ).set_index("Protein").assign(
            **{"log2(Intensity)": lambda x: np.log2(x["Intensity"] + 1)}
        )
    # Reorder the columns
    plot_data['Sample'] = pd.Categorical(
        plot_data['Sample'], 
        categories=column_order, 
        ordered=True
    )

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot the data
    sns.lineplot(
        ax=ax,
        data=plot_data,
        x="Sample",
        y="log2(Intensity)",
        hue="Perturbation",
        palette=color_dict,
        alpha=0.7,
        style="Perturbation",
        dashes=False,
        markers=["o", "X"],
        ci=95,
        err_style="bars",
        err_kws={"capsize": 0},
        legend="full",
        markersize=10  # Increased marker size for better visibility
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title, fontsize=14, loc="left", pad=10)
    ax.set_xlabel("Log2 Intensity")
    ax.set_ylabel("Density")

    ax.legend(
        title=legend_title,
        loc="upper right",
        ncol=2, 
        fontsize=10,
        title_fontsize=10,
        frameon=False,
        bbox_to_anchor=(.95, 1.25)
    )

    ax.grid(
        axis="both",
        which="major",
        color="lightgrey",
        linestyle="--",
        linewidth=.5,
        alpha=0.75,
    )

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig

def combined_curves(
        data: pd.DataFrame,
        pthr: float = 0.001,
        threshold_column: str = "threshold",
        prScore: str = "F1",

        # Figure settings
        figsize: Tuple[int, int] = (12, 6),
        title: str = "ROC and PR Curves",

        # Styling options
        curve_line_color: str = 'black',
        curve_line_width: float = 1.5,
        curve_line_alpha: float = 1,
        marker_size: int = 15,

        # Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'combined_curves_for_performance',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False
    ):
    """
    
    """

    # Check if the column names are correct
    if not {"FPR", "TPR"}.issubset(data.columns):
        raise ValueError("The data does not contain the required columns 'FPR' and 'TPR'")
    
    # Check if the column names are correct
    if not {"Precision", "Recall"}.issubset(data.columns):
        raise ValueError("The data does not contain the required columns 'Precision' and 'Recall'")
    roc_data = data.copy()
    pr_data = data.copy()
    # Fix the ROC with starting point 0,0 and ending point 1,1
    roc_data = pd.concat(
        [ roc_data, pd.DataFrame( { "FPR": [0, 1], "TPR": [0, 1] } ) ],
        ignore_index=True
    )
    # Sort the data by the FPR
    roc_data = roc_data.sort_values("FPR")
    
    # Roc auc
    roc_auc = np.trapz(roc_data["TPR"], roc_data["FPR"])

    # Initialize the figure
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot the ROC curve
    sns.lineplot(
        data=roc_data,
        x="FPR",
        y="TPR",
        ax=ax[0],
        color=curve_line_color,
        linewidth=curve_line_width,
        alpha=curve_line_alpha,
        markers=True,
        dashes=False,
        marker="o",
        markersize=marker_size, 
        markerfacecolor="white",
        markeredgewidth=curve_line_width,
        markeredgecolor=curve_line_color,
        errorbar=None
    )

    # Add the diagonal line
    ax[0].plot(
        [0, 1], [0, 1], 
        color="black", alpha=0.5,
        linestyle="--", linewidth=1.5, zorder=0
    )
    # Set the labels
    ax[0].set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax[0].set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax[0].set_title('ROC Curve', fontsize=14, fontweight="bold", pad=10)
    ax[0].grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")
    ax[0].text(
        0.95, 0.05, 
        f"AUC: {roc_auc:.2f}", 
        ha="right", fontsize=15, fontweight="bold", 
        color=curve_line_color, transform=ax[0].transAxes
    )
    ax[0].set_xlim(-0.05, 1.05)
    ax[0].set_ylim(-0.05, 1.05)

    # Add "X" marker for the same or closest threshold to the pthr and draw X at FPR, TPR 
    # Find the closest threshold to the pthr
    subset = data.loc[(data[threshold_column] - pthr).abs().idxmin()]
    ax[0].scatter(
        subset["FPR"], subset["TPR"],
        color="red", s=100, marker="x", zorder=10
    )

    # Fix the precision and recall values to start from 1,0 and end at 0,1
    pr_data = pd.concat(
        [
            pr_data,
            pd.DataFrame( { "Precision": [1, 0], "Recall": [0, 1], 'F1': [1, 0] } )            
        ],
        ignore_index=True
    )
    pr_data = pr_data.sort_values("Recall")
    # Remove P,R = 0,0 and 1,1
    pr_data = pr_data[~((pr_data["Recall"] == 0) & (pr_data["Precision"] == 0))]
    # pr_auc = np.trapz(data["Precision"], data["Recall"])
    # Get average precision and F1 score
    pr_auc = pr_data[prScore].mean()

    # Plot the PR curve
    sns.lineplot(
        data=pr_data,
        x="Recall",
        y="Precision",
        ax=ax[1],
        color=curve_line_color,
        linewidth=curve_line_width,
        alpha=curve_line_alpha,
        markers=True,
        dashes=False,
        marker="o",
        markersize=marker_size, 
        markerfacecolor="white",
        markeredgewidth=curve_line_width,
        markeredgecolor=curve_line_color,
        errorbar=None
    )

    # Set the labels
    ax[1].set_xlabel("Recall", fontsize=12)
    ax[1].set_ylabel("Precision", fontsize=12)
    ax[1].set_title('PR Curve', fontsize=14, fontweight="bold", pad=10)
    ax[1].grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")
    ax[1].text(
        0.05, 0.05, 
        f"{prScore}: {pr_auc:.2f}", 
        ha="left", fontsize=15, fontweight="bold", 
        color=curve_line_color , transform=ax[1].transAxes
    )
    ax[1].set_xlim(-0.05, 1.05)
    ax[1].set_ylim(-0.05, 1.05)
    # Draw the X marker
    ax[1].scatter(
        subset["Recall"], subset["Precision"],
        color="red", s=100, marker="x", zorder=10
    )

    # sns.despine(fig, top=True, right=True, left=True, bottom=True)
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig    

def single_ROC_curve(
        data: pd.DataFrame,
        
        # Figure settings
        figsize: Tuple[int, int] = (6, 6),

        title: str = "ROC Curve",

        # Styling options
        curve_line_color: str = 'black',
        curve_line_width: float = 1.5,
        curve_line_alpha: float = 1,
        marker_size: int = 15,

        # Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'single_ROC_curve',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False
    ):
    """
    
    """

    # Check if the column names are correct
    if not {"FPR", "TPR"}.issubset(data.columns):
        raise ValueError("The data does not contain the required columns 'FPR' and 'TPR'")
    
    # Calculate the AUC
    auc = np.trapz(data["TPR"], data["FPR"])

    # Initialize the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the ROC curve
    sns.lineplot(
        data=data,
        x="FPR",
        y="TPR",
        ax=ax,
        color=curve_line_color,
        linewidth=curve_line_width,
        alpha=curve_line_alpha,
        markers=True,
        dashes=False,
        marker="o",
        markersize=marker_size, 
        markerfacecolor="white",
        markeredgewidth=curve_line_width,
        markeredgecolor=curve_line_color,
    )

    # Add the diagonal line
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")

    # Set the labels
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")
    ax.text(0.95, 0.05, f"AUC: {auc:.2f}", ha="right", fontsize=15, fontweight="bold", color=curve_line_color, transform=ax.transAxes)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    sns.despine(fig, top=True, right=True, left=True, bottom=True)
    plt.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig

# PeCorA's significantly different peptide compared to other peptides in a given protein (boxplot)
def visualize_protein_peptides_boxplot(
        plot_dict: dict,
        peptide_column: str = "Peptide",
        condition_column: str = "Condition",
        intensity_column: str = "ms1adj",
        figsize: Tuple[int, int] = (10, 5),
        palette: Optional[List[str]] = None,  # Optionally provide a custom color palette
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        legend_fontsize: int = 10,
        use_sequence_legend: bool = True,
        jitter: float = 0.2,
        alpha: float = 0.7,
        showfliers: bool = False,
        # Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'PeCorA_Protein_boxplot',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False
    ) -> plt.Figure:
    """
        Visualizes a boxplot with stripplot overlay for a specific protein and its peptides,
        highlighting one or more significant peptides with their associated p-values.

        Args:
            data: DataFrame containing peptide-level data with intensity values and conditions.
            protein_id: The ID of the protein to plot.
            peptide_ids: A string or a list of strings representing the IDs of significant peptides.
            p_values: A float or a list of floats representing the p-values of the significant peptides (same order as peptide_ids).
            peptide_column: Column in data indicating the peptide IDs. Default is "Peptide".
            condition_column: Column in data indicating the experimental condition. Default is "Condition".
            intensity_column: Column in data indicating the intensity values. Default is "ms1adj".
            figsize: Figure size (width, height) in inches. Default is (8, 6).
            palette: A list of colors for the significant peptides. If None, a default palette is used.
            title_fontsize: Font size for the plot title. Default is 16.
            label_fontsize: Font size for axis labels. Default is 14.
            legend_fontsize: Font size for the legend. Default is 12.
            jitter: Jitter width for the stripplot points. Default is 0.2.
            alpha: Transparency of the stripplot points. Default is 0.7.
            showfliers: Whether to show outliers in the boxplot. Default is False.

        Returns:
            Matplotlib Figure: The generated figure object.
    """
    # Extract the information from the dictionary
    gene_name = plot_dict["Gene"]
    protein_id = plot_dict["Protein"]
    peptide_ids = plot_dict["Peptides"]
    p_values = plot_dict["p_values"]
    peptideName = plot_dict["PeptideIds"]
    data = plot_dict["PlotData"]

    if not isinstance(peptide_ids, list):
        peptide_ids = [peptide_ids]

    if not isinstance(p_values, list):
        p_values = [p_values]

    if not isinstance(peptideName, list):
        peptideName = [peptideName]

    if len(peptide_ids) != len(p_values):
        raise ValueError("The number of peptides and p-values should match")

    if palette is None:
        palette = [
            "grey", "salmon", "skyblue", "lightgreen", "orange", 
            "purple", "brown", "pink", "olive", "cyan", "magenta"
        ]
    
    if len(peptide_ids) > len(palette):
        # raise ValueError(f"Not enough colors in the palette for {len(peptide_ids)} peptides")
        # Reuse colors if there are more peptides than colors
        palette = palette * (len(peptide_ids) // len(palette) + 1)
    
    # Prepare data for plotting
    data["status"] = "Other Peptides"
    data = data.set_index(peptide_column)
    # Create a unique name for each peptide (name, seq, p-value)
    for i, peptide_id in enumerate(peptide_ids):
        # pvalue_str = f" (p={p_values[i]:.3f})"
        # Scientific notation for p-values
        pvalue_str = f" (p={p_values[i]:.2e})"
        if use_sequence_legend:
            data.loc[peptide_id, "status"] = f"ID:{peptideName[i]} - {peptide_id} {pvalue_str}"
        else:
            data.loc[peptide_id, "status"] = f"ID:{peptideName[i]} {pvalue_str}"

    # Define the desired order, with "Other Peptides" first
    status_order = (
        ["Other Peptides"] + 
        [category for category in data["status"].unique() if category != "Other Peptides"]
    )

    # Check if "status" is already categorical
    if not pd.api.types.is_categorical_dtype(data["status"]):
        # Convert "status" to a categorical data type
        data["status"] = pd.Categorical(data["status"])

    try: # Create a new categorical object with the desired order
        data["status"] = data["status"].cat.reorder_categories(status_order)  
    except: 
        # Ignore the reorder_categories error
        pass

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # The boxplots underneath
    sns.boxplot(
        x=condition_column,
        y=intensity_column,
        data=data,
        hue="status",
        palette=palette,
        ax=ax,
        linewidth=0.5,
        showfliers=showfliers,
        width=.75,
    )
    # The stripplot on top
    sns.stripplot(
        ax=ax,
        x=condition_column,
        y=intensity_column,
        data=data,
        hue="status",
        dodge=True,
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        alpha=alpha,
        jitter=jitter,
        size=6,
        legend=False,
    )

    # Add annotations and formatting
    ax.set_title(
        f"{gene_name} ({protein_id}) - Result of Linear Model Analysis", 
        fontsize=title_fontsize, 
        fontweight="bold", 
        loc="left",
        pad=20
    )
    ax.set_xlabel("", fontsize=label_fontsize)
    ax.set_ylabel("Normalized against Normal", fontsize=label_fontsize)
    ax.grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        # title="Status",
        # title_fontsize=legend_fontsize,
        fontsize=10,
        ncol=1,
        frameon=False,
    )

    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig


def proteinAnnotation(
        plot_dict: dict,
        figsize: tuple = (15, 5),
        # Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'ProteinAnnotation',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False       
    ):
    
    # Handle default regulation_colors
    fig, ax = plt.subplots(figsize=figsize)
    annot_data = plot_dict["UniprotAnnotData"]
    res_data = plot_dict["ResultSubset"]
    regulation_colors = plot_dict["RegulationColors"]
    npeps = len(res_data)

    # Scale the figure size based on the number of peptides
    figsize = (figsize[0]+(npeps/10), figsize[1]+(npeps/25))

    # --- Default Styles and Settings ---
    protein_height = 1
    domain_height_scale = 1.1   # The height of each domain rectangle
    
    protein_color = '#e5e5e5'
    default_colors = {
        'DOMAIN': '#83c5be',
        'CHAIN': '#ffddd2',
        'SITE': '#540b0e',
        'MOD_RES': '#ffbe0b',
    }
    
    # --- Plotting UniProt Annotations ---
    def add_uniprot_features(subset, feature_type, y_offset=0, height_scale=1):
        """Plots UniProt annotations (domains, chains, sites)."""
        if subset.empty: return
        vertical_padding = ((domain_height_scale * protein_height) - protein_height) / 2
        for _, row in subset.iterrows():
            start, end = row["start"], row["end"]
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            width = end - start

            # Calculate centered x and y positions
            rect_x = (start + end) / 2 - width / 2
            rect_y = y_offset - vertical_padding * height_scale
            
            patch = Rectangle((rect_x, rect_y), width, protein_height * height_scale,
                                    facecolor=default_colors.get(feature_type, default_colors[feature_type]),
                                    alpha=0.5, edgecolor='black')
            ax.add_patch(patch)

            # Add labels (if available)
            if 'note' in row:
                ax.text((start + end) / 2, rect_y + (protein_height * height_scale) / 2,
                        row['note'], ha='center', va='center',
                        rotation=90 if feature_type == 'DOMAIN' else 0,
                        fontsize=10, fontweight='bold')
    
    # --- Plotting UniProt Annotations ---
    def add_mod_res(subset, y_offset=1.25):
        """Plots MOD_RES annotations."""
        marker_styles = {  
            'phosphorylation': 'o',
            'acetylation': '^',
            'methylation': 's',
        }
        if subset.empty: return
        for _, row in subset.iterrows():
            mod_pos = row["start"]
            if not np.isfinite(mod_pos):
                continue
            mod_type = row["note"].lower() 
            marker = marker_styles.get(mod_type, '*')  
            ax.plot(mod_pos, y_offset, marker=marker, markersize=10, color=default_colors['MOD_RES'])

            # Connect marker to protein rectangle
            ax.vlines(mod_pos, 1, y_offset, linestyles='dotted', linewidth=0.5, color='gray')

    # --- Plotting UniProt Annotations ---
    def add_peptide_markers(subset):
        """Plots peptide markers."""
        if subset.empty: return
        for _, peptide in subset.iterrows():
            startpos, endpos = peptide['startpos'], peptide['endpos']
            if not np.isfinite(startpos) or not np.isfinite(endpos):
                continue
            color = regulation_colors[peptide['TumorRegulation']]
            y_offset = 1.5 + 0.2 * peptide['trace']  
            ax.add_patch(Rectangle(
                (startpos, y_offset - 0.1), endpos - startpos, 0.2,
                facecolor=color, alpha=0.5, edgecolor='black'
            ))
            pepID = str(peptide['PeptideID'])
            if peptide["isSignificant"]:
                pepID += "*"
                fs = 12
                fw = 'bold'
            else:
                fs = 10
                fw = 'normal'
            ax.text(
                (startpos + endpos) / 2, y_offset,
                pepID, ha='center', va='center',
                fontsize=fs, fontweight=fw, color="white"
            )

    # --- Plotting ---

    # Protein rectangle
    ax.add_patch(Rectangle((0, 0), plot_dict["ProteinLength"], protein_height, color=protein_color))

    # UniProt annotations
    add_uniprot_features(
        annot_data[annot_data["feature"] == "DOMAIN"], 
        'DOMAIN', 0.2, domain_height_scale
    )
    
    add_uniprot_features(
        annot_data[annot_data["feature"] == "CHAIN"], 
        'CHAIN', -0.5, 0.5
    )
    add_uniprot_features(
        annot_data[annot_data["feature"] == "SITE"], 
        'SITE'
    )
    add_mod_res(
        annot_data[annot_data["feature"] == "MOD_RES"]
    ) 
    # Peptide markers
    add_peptide_markers(res_data)

    # --- Styling ---
    # Additional adjustments
    ax.set_xlim([0, plot_dict["ProteinLength"]])
    ax.set_ylim([ax.get_ylim()[0], 2 + 0.05 * res_data['trace'].max()])
    ax.set_xlabel('Protein Position')
    ax.set_ylabel('Peptide Markers')
    ax.grid("both", linestyle="--", linewidth=0.75, alpha=0.5, color="lightgrey")
    # ax2.set_ylabel('Boxplot Values')
    ax.set_title(f"{plot_dict['Gene']} ({plot_dict['Protein']}) - Protein Coverage: {plot_dict['ProteinCoverage']:.2f}%", fontsize=14, loc="left")

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='#83c5be', lw=10),
            plt.Line2D([0], [0], color='#ffddd2', lw=10),
            plt.Line2D([0], [0], color='#540b0e', marker='v', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color='#ffbe0b', marker='*', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color=regulation_colors['Up'], lw=10),
            plt.Line2D([0], [0], color=regulation_colors['Down'], lw=10),
        ],
        labels=['Domain', 'Chain', 'Site', 'Modification', 'Upregulated in Tumor', 'Downregulated in Tumor'],
        title='Annotations',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        ncols = 10,
        frameon=False,
        bbox_to_anchor=(1, 1.15),
    )

    ax.get_yaxis().set_visible(False)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig

# #### PDF Report Generation ####        

# # Helper function to split a DataFrame into smaller chunks
# def split_dataframe(df, chunk_size):
#     """Splits a DataFrame into smaller chunks."""
#     for start in range(0, len(df), chunk_size):
#         yield df[start:start + chunk_size]

# def generate_multipage_pdf_report(
#         plot_dict, 
#         fdr_threshold=10**-3,
#         output_filename="protein_report.pdf"
#     ):
#     """Generates a multi-page PDF report with data, tables, and figures."""

#     with PdfPages(output_filename) as pdf:
#         # Page 1: Protein Info
#         first_page = plt.figure(figsize=(11.69, 8.27))  # A4 size
#         first_page.clf()  # Clear the figure

#         # Add the protein info:
#         ax = first_page.add_subplot(111)  # Create a subplot for the protein info
#         tmp = (
#             f"{plot_dict['Gene']} ({plot_dict['Protein']}) - {plot_dict['ProteinLength']} aa\n\n" +
#             f"{plot_dict['ProteinName']}\n\n" +
#             f"{plot_dict['ProteinCoverage']:.2f}% sequence coverage\n\n"
#         )
#         ax.text(0.5, 0.5, tmp, ha="center", va="center", fontsize=18)
#         ax.axis("off")  # Remove plot axes from the protein info

#         pdf.savefig(first_page)
#         plt.close(first_page)  # Close to free up memory

#         # Pages for Result Subset Table
#         table_data = plot_dict["ResultSubset"]
#         # Check if ProteoformGroup is present 
#         if "ProteoformGroup" in table_data.columns:
#             table_data = table_data[[
#                 "PeptideID", "Peptide", "startpos", "endpos", "TumorRegulation", "isSignificant", "ProteoformGroup"
#             ]]
#         else:
#             table_data = table_data[[
#                 "PeptideID", "Peptide", "startpos", "endpos", "TumorRegulation", "isSignificant"
#             ]]

#         chunk_size = 30  # Adjust based on how many rows fit on one page

#         for i, chunk in enumerate(split_dataframe(table_data, chunk_size)):
#             page = plt.figure(figsize=(11.69, 8.27))
#             page.clf()
            
#             ax = page.add_subplot(111)
#             if i == 0:
#                 ax.text(0.5, 1.0, "Result Subset Table", ha="center", va="center", fontsize=14, transform=ax.transAxes)
#             table = pd.plotting.table(ax, chunk, loc="center")
#             table.auto_set_font_size(False)
#             table.set_fontsize(6)
#             ax.axis("off")

#             pdf.savefig(page)
#             plt.close(page)

#         # Figure 1: Signf Result Boxplot
#         fig = visualize_protein_peptides_boxplot(
#             plot_dict
#         )
#         pdf.savefig(fig)  # Add the figure directly to the PDF
#         plt.close(fig)
        
#         # Figure 2: Detailed Peptide Quant Value
#         fig = detailed_peptide_line(
#             plot_dict = plot_dict,
#             fdr_threshold = fdr_threshold,
#         )
#         pdf.savefig(fig)  # Add the figure directly to the PDF
#         plt.close(fig)

#         # Figure 3: Peptide Correlation Heatmap
#         fig = heatmap_with_clusters(
#             plot_dict=plot_dict,
#         )
#         pdf.savefig(fig)  # Add the figure directly to the PDF
#         plt.close(fig)

#         # Figure 4: Protein Annotation
#         fig = proteinAnnotation(
#             plot_dict,
#         )
#         pdf.savefig(fig)  # Add the figure directly to the PDF
#         plt.close(fig)

#         # Last Pages: Uniprot Annotation Table
#         table_data = plot_dict["UniprotAnnotData"][["feature", "start", "end", "note"]]
#         chunk_size = 30  # Adjust based on how many rows fit on one page

#         for i, chunk in enumerate(split_dataframe(table_data, chunk_size)):
#             page = plt.figure(figsize=(11.69, 8.27))
#             page.clf()
            
#             ax = page.add_subplot(111)
#             ax.text(0.5, 1.05, f"Uniprot Annotation Table (Page {i+1})", ha="center", va="center", fontsize=14, transform=ax.transAxes)
#             table = pd.plotting.table(ax, chunk, loc="center")
#             table.auto_set_font_size(False)
#             table.set_fontsize(8)
#             ax.axis("off")

#             pdf.savefig(page)
#             plt.close(page)

############################ Power Analysis Profile Plot ############################

def single_variable_power_profile(
        plot_data: pd.DataFrame,            # Input DataFrame with power analysis results
        x_axis_variable: str,               # The variable to plot on the x-axis ("eqThr", "cvMean", or "nRep")
        y_axis_variable: str,               # The variable to plot on the y-axis ("calc_power")
        target_power: float,                # Desired statistical power for the test
        # Figure parameters
        figsize: tuple = (6, 4),            # Dimensions (width, height) of the figure in inches
        line_color: str = "#003566",        # Color of the line plot
        target_line_color: str = "#fca311", # Color of the line indicating the target power
        figtitle: str = None,               # Title of the plot
        xlabel: str = None,                 # Label for the x-axis
        # Save parameters
        save: bool = False,                 # Whether to save the plot (True) or not (False)
        filename: str = "powerProfile_line",# Filename for saving the plot
        filepath: str = "",                 # Directory path for saving the plot
        fileformat: list[str] = ["png"],    # List of file formats to save the plot in (e.g., ["png", "pdf"])
        dont_show: bool = False,            # If True, the plot won't be displayed interactively
    ):
    """
        Plots the power analysis profile for a single variable against the achieved power.

        This function generates a line plot illustrating how the statistical power
        varies with changes in a single experimental parameter (equivalence threshold,
        mean coefficient of variation, or number of replicates). The plot includes a
        horizontal line indicating the target power, error bars representing standard
        deviation, and an annotation detailing the simulation parameters.

    """

    # Validate the input for x-axis variable
    if x_axis_variable not in ["eqThr", "cvMean", "nRep"]:
        raise ValueError("Invalid x_axis_variable. Choose from 'eqThr', 'cvMean', or 'nRep'.")

    cvMean = plot_data["cvMean"].unique()[0]
    nRep = plot_data["nRep"].unique()[0]
    pThr = plot_data["pThr"].unique()[0]
    corr = plot_data["corr"].unique()[0]
    eqThr = plot_data["eqThr"].unique()[0]
    nRepeats = plot_data["iteration"].max() + 1

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the mean power values
    sns.lineplot(
        x=x_axis_variable,   
        y=y_axis_variable, 
        data=plot_data,      
        ax=ax,              
        color=line_color,    
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor=line_color,
        n_boot=1000,       
        errorbar=("sd"),  
    )

    # Adjust the y-axis limits
    ax.set_ylim(-0.05, 1.05) 

    # Set the title dynamically based on the x-axis variable
    if figtitle is None:
        figtitle = f"Power Analysis: Effect of {x_axis_variable.replace('_', ' ').capitalize()}"
    ax.set_title(
        figtitle, 
        fontsize=12, 
        fontweight="bold", 
        loc="left",
        pad=20
    )

    # Set the x-axis label dynamically based on the x-axis variable
    if xlabel is None:
        xlabel = x_axis_variable.replace("_", " ").capitalize()
    ax.set_xlabel(
        xlabel, 
        fontsize=10, 
        fontweight="bold", 
        labelpad=10
    )

    # Set the y-axis label
    ax.set_ylabel(
        "Power (Difference from Adjusted SEI)", 
        fontsize=10, 
        fontweight="bold",
        labelpad=10
    )

    # Add text annotation with parameters, excluding the x-axis variable
    annotation_text = f"Target Power: {target_power} | Parameters: "
    if x_axis_variable != "eqThr":
        annotation_text += f"eqThr = {eqThr}, "
    if x_axis_variable != "meanCV":
        annotation_text += f"CV% = {cvMean:.2f}, "
    if x_axis_variable != "nRep":
        annotation_text += f"n = {nRep}, "
    annotation_text += f"pThr = {pThr}, Cor = {corr}, repeat = {nRepeats}"

    ax.text(
        x=0.0,
        y=1.025,
        s=annotation_text,
        fontsize=8,
        fontstyle="italic",
        ha="left",
        va="bottom",
        transform=ax.transAxes
    )

    # Add a horizontal line to indicate the target power
    ax.axhline(
        y=target_power,  
        color=target_line_color, 
        linestyle="--",
        linewidth=1.5,
        label="Target Power"
    )

    # Add gridlines for better readability
    ax.grid(
        axis="both", 
        color="lightgray", 
        alpha=0.5, 
        linestyle="--", 
        linewidth=0.5
    )

    # Remove top and right spines for a cleaner look
    sns.despine(left=True, bottom=True)
    plt.tight_layout()  # Adjust layout for better visual appeal

    # Save the plot if requested
    if save:
        save_figures(
            plt.gcf(), 
            filename=filename, 
            filepath=filepath, 
            fileformat=fileformat
        )
        if dont_show:  # Close the plot if not showing interactively
            plt.close()

def single_variable_SEI_profile(
        plot_data: pd.DataFrame,            # Input DataFrame with power analysis results
        cv_data: pd.DataFrame,              # CV distribution data (long format for plotting)
        x_axis_variable: str,               # The variable to plot on the x-axis ("eqThr", "cvMean", or "nRep")
        y_axis_variable: str,               # The variable to plot on the y-axis ("calc_SEI")
        target_variable: str,               # The variable to plot as a horizontal line for Adjusted SEI 

        # Figure parameters
        figsize: tuple = (6, 4),            # Dimensions (width, height) of the figure in inches
        figwidthratio: list = [3, 1],       # Width ratio for the subplots
        line_color: str = "#003566",        # Color of the line plot
        target_line_color: str = "#fca311", # Color of the line indicating the target power
        ideal_line_color: str = "#8d0801",  # Color of the line indicating the ideal SEI
        figtitle: str = None,               # Title of the plot
        xlabel: str = None,                 # Label for the x-axis

        # Save parameters
        save: bool = False,                 # Whether to save the plot (True) or not (False)
        filename: str = "SEI_profile_line", # Filename for saving the plot
        filepath: str = "",                 # Directory path for saving the plot
        fileformat: list[str] = ["png"],    # List of file formats to save the plot in (e.g., ["png", "pdf"])
        dont_show: bool = False,            # If True, the plot won't be displayed interactively
    ):
    """Plots the SEI analysis profile for a single variable against the achieved SEI.
    """

    # Validate the input for x-axis variable
    if x_axis_variable not in ["eqThr", "cvMean", "nRep"]:
        raise ValueError("Invalid x_axis_variable. Choose from 'eqThr', 'cvMean', or 'nRep'.")

    cvMean = plot_data["cvMean"].unique()[0]
    nRep = plot_data["nRep"].unique()[0]
    pThr = plot_data["pThr"].unique()[0]
    corr = plot_data["corr"].unique()[0]
    eqThr = plot_data["eqThr"].unique()[0]
    nRepeats = plot_data["iteration"].max() + 1

    # Create the plot
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": figwidthratio,
            "wspace": 0.0
        }
    )

    
    sns.lineplot(
        ax=ax[0],
        x=x_axis_variable,   
        y=y_axis_variable, 
        data=plot_data,                 
        color=line_color,    
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor=line_color,
        n_boot=1000,       
        errorbar=("sd"),  
    )

    # Adjust the y-axis limits
    ax[0].set_ylim(-0.05, 1.05) 

    # Set the title dynamically based on the x-axis variable
    if figtitle is None:
        figtitle = f"SEI Analysis: Effect of {x_axis_variable.replace('_', ' ').capitalize()}"
    ax[0].set_title(
        figtitle, 
        fontsize=12, 
        fontweight="bold", 
        loc="left",
        pad=20
    )

    # Set the x-axis label dynamically based on the x-axis variable
    if xlabel is None:
        xlabel = x_axis_variable.replace("_", " ").capitalize()
    ax[0].set_xlabel(
        xlabel, 
        fontsize=10, 
        fontweight="bold", 
        labelpad=10
    )

    # Set the y-axis label
    ax[0].set_ylabel(
        "Calculated SEI", 
        fontsize=10, 
        fontweight="bold",
        labelpad=10
    )

    # Set the Ideal SEI label
    ax[0].axhline(
        y=1,
        color=ideal_line_color,
        linestyle="--",
        label=f"Ideal SEI (1.00)",
    )

    if x_axis_variable != "cvMean": 
        # Add a horizontal line to indicate the target (adjusted) SEI
        target_var = 1 - cvMean / 100
        ax[0].axhline(
            y=target_var,  
            color=target_line_color, 
            linestyle="--",
            linewidth=1.5,
            label=f"Adjusted SEI ({target_var:.2f})"
        )
        annotation_text = f"CV Adjusted SEI: {target_var} | Parameters: "
    else:
        sns.lineplot(
            x=x_axis_variable,   
            y=target_variable, 
            data=plot_data,      
            ax=ax[0],              
            color=target_line_color,    
            linewidth=1.5,
            linestyle="--",
            label=f"Adjusted SEI"
        )
        annotation_text = f"Parameters: "

    # Add text annotation with parameters, excluding the x-axis variable
    if x_axis_variable != "eqThr":
        annotation_text += f"eqThr = {eqThr}, "
    if x_axis_variable != "cvMean":
        annotation_text += f"CV% = {cvMean:.2f}, "
    if x_axis_variable != "nRep":
        annotation_text += f"n = {nRep}, "
    annotation_text += f"pThr = {pThr}, Cor = {corr}, repeat = {nRepeats}"

    ax[0].text(
        x=0.0,
        y=1.025,
        s=annotation_text,
        fontsize=8,
        fontstyle="italic",
        ha="left",
        va="bottom",
        transform=ax[0].transAxes
    )
    # Set the legend
    ax[0].legend(
        frameon=False,
        # loc="upper right",
        fontsize=8,
        # bbox_to_anchor=(1, 1.125)
    )

    # Plot the CV distribution
    sns.violinplot(
        ax=ax[1],
        data=cv_data,
        x="cvMean",
        y="cvDist",
        color="lightgray",
        linewidth=0.5,
        cut=0,
        scale="width",
        # inner="quartile",
    )
    # ax[1].set_tit
    ax[1].set_ylabel("CV (ratio)")
    ax[1].set_xlabel(f"CV Distribution(s)")
    # Place y-axis on the right
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    # Add gridlines for better readability
    for i in range(2):
        ax[i].grid(
            axis="both", 
            color="lightgray", 
            alpha=0.5, 
            linestyle="--", 
            linewidth=0.5
        )

    # Remove top and right spines for a cleaner look
    sns.despine(left=True, bottom=True)
    plt.tight_layout()  
    
    # Save the plot if requested
    if save:
        save_figures(
            plt.gcf(), 
            filename=filename, 
            filepath=filepath, 
            fileformat=fileformat
        )
        if dont_show:  # Close the plot if not showing interactively
            plt.close()

def heatmap_powerPerSetup(
    plot_data: pd.DataFrame,  # Input DataFrame with Calculated_EQ and nReps
    adjs_SEI: float,  # Adjusted SEI value

    # Figure settings
    figsize: tuple = (8, 4),

    # Labels and titles
    title: str = "Power Analysis for Equivalence Boundaries and Replicates",
    title_fontsize: int = 12,
    title_loc: str = "left",
    title_pad: float = 10,
    xlabel: str = "Symmetrical Equivalence Boundary as LFC",
    ylabel: str = "# of Replicates",
    cbar_label: str = "Power (Difference from Adjusted SEI)",

    # Heatmap styling
    cmap: str = "Greys",
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = 0,
    vmax: float = 1,
    cbar_orientation: str = "vertical",
    cbar_pad: float = 0.02,

    # Save parameters
    save: bool = False,
    filename: str = "power_analysis_heatmap",
    filepath: str = "",
    fileformat: list[str] = ["png"],
    dont_show: bool = False,
):
    """
    Creates a heatmap to visualize the power analysis for different equivalence boundaries
    and replicate numbers.

    Args:
        plot_data: DataFrame containing 'Calculated_EQ' and 'nReps' columns.
        adjs_SEI: Adjusted SEI value.
        ...: Other parameters for customization.
    """

    # Calculate power
    plot_data["Power"] = plot_data.apply(
        lambda x: tests.calculate_power(
            x["Calculated_EQ"], 
            adjs_SEI
        ), axis=1
    )

    # Pivot the DataFrame
    tmp_df = plot_data.pivot_table(
        index="nReps", 
        columns="B_eq", 
        values="Power", 
        aggfunc="mean"
    )
    tmp_df.columns = tmp_df.columns.map(lambda x: f"{x:.2f}")
    tmp_df.index = tmp_df.index.map(lambda x: f"{x:.0f}")

    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data=tmp_df,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        cbar_kws={
            "label": cbar_label,
            "orientation": cbar_orientation,
            "pad": cbar_pad,
        },
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_title(
        title, 
        fontsize=title_fontsize, 
        fontweight="bold", 
        loc=title_loc, 
        pad=title_pad
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    # Saving
    if save:
        save_figures(
            plt.gcf(), 
            filename=filename, 
            filepath=filepath, 
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

# Call the function
def single_pair_summary(
        data: pd.DataFrame,
        total_proteins: int,
        pair_names: tuple,
        # Data related parameters
        df_pval: str = "df_p",
        df_qval: str = "df_adjp",
        eq_pval: str = "eq_p",
        eq_qval: str = "eq_adjp",
        log2FC: str = "log2FC",
        logQvalue: str = "log10(adj_pval)",
        status: str = "Status",
        # Stats related parameters
        pThr: float = 0.05, 
        dfThr: float = 1,
        eqThr: float = 1,            
        corr_name: str = "FDR",
        # Plot parameters
        figsize: tuple = (10, 5),
        offset: float = .75,
        save: bool = False,
        filename: str = "single_pair_summary",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"], 
        dont_show: bool = False,
    ):
    """

    """

    # Check if corr_name is None set as string
    if corr_name is None: corr_name = "None"
    
    # Create a counts dataframe 
    cnts = data[status].value_counts()
    cnts["Excluded"] = total_proteins - cnts.sum()
    cnts = cnts.reset_index()
    cnts.columns = ["Status", "Count"]

    # Initialize the figure
    fig = plt.figure( figsize=figsize )
    grid = gridspec.GridSpec(
        nrows=2, ncols=3, 
        width_ratios=[ 0.3, 0.6, 0.1 ], 
        wspace=0.2, hspace=0.3
    )

    # Calculate the lim maxes
    ymax = data[logQvalue].abs().max() + offset
    xmax = data[log2FC].abs().max() + offset

    # Initialize the axes
    ax1 = plt.subplot(grid[0, 0]) # T-test pvalue dist
    ax2 = plt.subplot(grid[1, 0]) # TOST pvalue dist
    ax3 = plt.subplot(grid[:, 1:2]) # Antlers Plot
    ax4 = plt.subplot(grid[:, 2]) # Protein Status Counts

    # Plot T-test Histogram of P- & Adj.P-values
    sns.histplot(
        data=data,
        x=df_pval,
        ax=ax1,
        color=stat_colors[-1],
        label="P-Value",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    sns.histplot(
        data=data,
        x=df_qval,
        ax=ax1,
        color=stat_colors[-2],
        label="Adj.P-Value ( " + corr_name + " )",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    ax1.set_title( "T-Test (Difference)", y=1, fontsize=12 )
    ax1.legend( loc="upper right", frameon=False )
    # Add styling to the plot
    ax1.set_xlim([0, 1])
    ax1.grid( axis="both", color="lightgray", alpha=0.5, linestyle="--", linewidth=0.5 )
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency")
    ax1.set_xticklabels([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    # ax1.spines["left"].set_visible(False)

    sns.histplot(
        data=data,
        x=eq_pval,
        ax=ax2,
        color=stat_colors[1],
        label="P-Value",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    sns.histplot(
        data=data,
        x=eq_qval,
        ax=ax2,
        color=stat_colors[0],
        label="Adj.P-Value ( " + corr_name + " )",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    ax2.set_title( "TOST (Equivalence)", y=1, fontsize=12 )
    ax2.legend( loc="upper right", frameon=False, )
    ax2.set_xlim([0, 1])
    ax2.grid( axis="both", color="lightgray", alpha=0.5, linestyle="--", linewidth=0.5 )
    ax2.set_xlabel("P-Value")
    ax2.set_ylabel("Frequency")
    # ax2.set_xticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # ax2.spines["bottom"].set_visible(False)
    # ax2.spines["left"].set_visible(False)

    # Plot Mutant Volcano Plot
    sns.scatterplot(
        data=data,
        x=log2FC,
        y=logQvalue,
        hue=status,
        ax=ax3,
        palette=status_palette,
        # alpha=0.5,
        s=100,
        linewidth=0.5,
        edgecolor="white",
        rasterized=True,
    )
    ax3.set_xlim([-xmax, xmax])
    ax3.set_ylim([-ymax, ymax])
    ax3.set_title( "Antlers Plot", y=1, fontsize=12 )
    ax3.set_xlabel("log2FC")
    ax3.set_ylabel( "log10(Adj.P-value)", labelpad=-5 )
    ax3.legend( loc="lower left", frameon=False, ).remove()
    ax3.grid( axis="both", color="lightgray", alpha=0.5, linestyle="--", linewidth=0.5 )

    # Add lines to the plot to indicate the thresholds
    # For the Adj. P-value threshold (pThr)
    ax3.axhline( y=-np.log10(pThr), color="#99999975", linestyle="--", linewidth=1.5, )
    ax3.axhline(y=np.log10(pThr), color="#99999975", linestyle="--", linewidth=1.5)
    ax3.axvline(x=dfThr, color=stat_colors[-1], linestyle="--", linewidth=1.5, alpha=0.75)
    ax3.axvline(x=-dfThr, color=stat_colors[-1], linestyle="--", linewidth=1.5, alpha=0.75)
    ax3.axvline(x=eqThr, color=stat_colors[1], linestyle="--", linewidth=1.5, alpha=0.75)
    ax3.axvline(x=-eqThr, color=stat_colors[1], linestyle="--", linewidth=1.5, alpha=0.75)

    # Plot Protein Status Count Plot
    sns.barplot(
        data=cnts,
        x="Count",
        y="Status",
        ax=ax4,
        palette=status_palette,
        rasterized=True,
        order=list(status_palette.keys())
    )   
    # Add the counts to the plot
    for p in ax4.patches:
        width = p.get_width()
        if width > 0: width = int(width)
        else: width = 0
        ax4.text(
            width + 0.15,
            p.get_y() + p.get_height() / 2,
            width,
            ha="left",
            va="center",
            fontsize=12,
            color="k",
        )

    ax4.set_title( "Protein Count per Status", y=1, fontsize=12 )
    ax4.set_xlabel("Protein Count")
    ax4.set_xticks([])
    ax4.set_xticklabels([])
    ax4.set_ylabel("")
    # Rotate the y-axis ticklabels to be parallel to the y-axis
    ax4.set_yticklabels(
        ax4.get_yticklabels(), 
        rotation=90, 
        horizontalalignment="center", 
        verticalalignment="center", 
        fontsize=12
    )
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    
    fig.suptitle(
        (
            "Single Pair's Test Summary ("+ 
            pair_names[0] + 
            " vs " + 
            pair_names[1] + 
            ")"
        ),
        fontsize=14, 
        y=.975,
    )

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename+"_"+pair_names[0]+"_vs_"+pair_names[1],
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close(fig)   

################################ Benchmarking Plots ################################

def draw_fpr_across_thresholds(
    pfg_metrics_data: pd.DataFrame,
    pthr: float,
    threshold_col: str = "threshold",
    figsize: tuple = (6, 4),
    color: str = 'black',
    linewidth: int = 2,
    marker: str = 'o',
    markersize: int = 15,
    markerfacecolor: str = 'white',
    markeredgewidth: float = 1.5,
    markeredgecolor: str = 'black'
) -> None:
    """
    Draws the False Positive Rate (FPR) across thresholds.

    Parameters:
    - pfg_metrics_data (pd.DataFrame): DataFrame containing the metrics data.
    - pthr (float): The threshold value for drawing the vertical line.
    - figsize (tuple): Size of the figure.
    - color (str): Color of the line plot.
    - linewidth (int): Width of the line plot.
    - marker (str): Marker style for the line plot.
    - markersize (int): Size of the markers.
    - markerfacecolor (str): Face color of the markers.
    - markeredgewidth (float): Edge width of the markers.
    - markeredgecolor (str): Edge color of the markers.
    """
    fig, ax = plt.subplots(figsize=figsize)
    pfg_metrics_data['-log10(Threshold)'] = -np.log10(pfg_metrics_data[threshold_col])
    # Remove infinite values
    pfg_metrics_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    pfg_metrics_data.dropna(subset=['-log10(Threshold)'], inplace=True)
    sns.lineplot(
        data=pfg_metrics_data,
        x='-log10(Threshold)',
        y='FPR',
        hue='Method',
        ax=ax,
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        dashes=False,
        markerfacecolor=markerfacecolor,
        markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor,
    )
    # Set ylimit to 1
    ax.set_ylim(-0.05, 1.05)
    # Set the axis labels
    ax.set_xlabel("-log10(Threshold)")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate across Thresholds")
    ax.grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")
    ax.legend(title="Method", frameon=False, loc="upper right", ncol=1)
    # Draw a vertical line at pthr and write the FPR value at that point
    ax.axvline(-np.log10(pthr), color="red", linestyle="--", linewidth=1)
    ax.text(
        -np.log10(pthr) + 0.1, .95, 
        f"FPR: {pfg_metrics_data[pfg_metrics_data[threshold_col] == pthr]['FPR'].values[0]:.2f}",
        rotation=0, fontsize=12, color="red"
    )
    # Write a note about the plot (no True Positives in this case that's why only FPR is shown)
    ax.text(
        0.5, 0.5, 
        "No True Positives expected in this case\nOnly False Positives are shown",
        ha="center", va="center", fontsize=12, fontstyle="italic", 
        transform=ax.transAxes
    )
    plt.tight_layout()

# TODO: This function is very specific to the data and needs to be generalized
# TODO: Need to make it more modular and controllable by the user
def detailed_peptide_line_simVer(
        data: pd.DataFrame,
        cur_protein: str,
        condition_palette: dict,
        cluster_palette: dict,
        preferred_pval: float = 0.05,
        
        # Plotting parameters
        figsize: tuple = (12, 12),

    ):

    # plot_data = data.sort_values(['startpos', 'endpos']).copy()
    plot_data = data.copy()
    # If ProteinID is not present, calculate based on startpos and endpost and make from 0-based to 1-based
    if "PeptideID" not in plot_data.columns:
        plot_data["PeptideID"] = plot_data['Peptide']
    plot_data["-log10(adj.pval)"] = -np.log10(plot_data['adj.pval'])
        
    ## Define the kwargs for plots
    lineplot_params = {
        "alpha": 0.75,
        "marker": "o",
        "markersize": 10,
        "markeredgewidth": 0,
        "linewidth": 1.5,
        "linestyle": "-",
        "dashes": False,
        "err_style": "bars",
        "errorbar": ('se', 1.5),
        "err_kws": {
            "capsize": 5, 
            "elinewidth": 2.5, 
            "capthick": 2.5, 
            "zorder": 1, 
            "linewidth": 1.5
        },
    }
    scatterplot_params = {
        "s": 125,
        "edgecolor": "None",
        "linewidth": 1,
        "alpha": 0.75,
        "markers": {True: "*", False: "s"}
    }
    legend_params = {
        "fontsize": 10,
        "title_fontsize": 12,
        "edgecolor": "black",
        "facecolor": "white",
        "frameon": False,
        "ncol": 1,
        "markerscale": 1.5,
        "labelspacing": 0.5,
    }

    ## Initialize the figure
    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=1, nrows=3, 
        sharex=True, sharey=False,
        gridspec_kw={ "height_ratios": [1, .5, 1], "hspace": 0.01 }
    )

    # First Plot shows the adjusted intensity values
    sns.lineplot(
        ax=ax[0],
        data=plot_data,
        x="PeptideID",
        y="adjIntensity",
        hue="Condition",
        palette=condition_palette,
        **lineplot_params
    )
    ax[0].set_ylabel("Control Adjusted Intensity", fontsize=12)
    ax[0].legend(**legend_params, title="Condition", loc="upper right", bbox_to_anchor=(1.1, 1))

    subset = plot_data[["PeptideID", "adj.pval", "pertPeptide"]].drop_duplicates()
    # Add * for the peptides with adjusted p-value < 0.05
    for i, row in subset[subset["adj.pval"] < preferred_pval].iterrows():
        ax[0].text(
            row["PeptideID"],
            0 ,
            "*",
            fontsize=25,
            color="black",
            ha="center",
            va="center",
        )

    # Create data for pepClusters
    pepClusters = plot_data[['PeptideID', '-log10(adj.pval)', 'cluster_id', 'isSignificant']].drop_duplicates()
    # Second Plot shows the adjusted p-values
    sns.scatterplot(
        ax=ax[1],
        data=pepClusters,
        x="PeptideID",
        y="-log10(adj.pval)",
        style="isSignificant",
        color='k',
        **scatterplot_params
    )
    ax[1].set_ylabel("-Log10(FDR)", fontsize=12)
    ax[1].legend(**legend_params, title="Singf. Peptide", loc="upper right", bbox_to_anchor=(1.1, 1))

    # Draw a line at preferred p-value threshold
    ax[1].axhline(
        -np.log10(preferred_pval),
        color="salmon",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        zorder=0
    )

    minLine = pepClusters["-log10(adj.pval)"].min()
    maxLine = pepClusters["-log10(adj.pval)"].max()
    
    # Add the clusters as rectangles with color
    for i, row in pepClusters.iterrows():
        # Add the rectangle using mixed coordinates
        ax[1].add_patch(
            plt.Rectangle(
                (row["PeptideID"] - 0.5, minLine),  # x in data coordinates, 
                1,
                maxLine - minLine,
                transform=ax[1].get_xaxis_transform(),  # Use x-axis coordinates
                color=cluster_palette[row["cluster_id"]],
                alpha=.5,
                zorder=1,
                linewidth=1.5,
                edgecolor="black",
            )
        )
    # Add a secondary legend for the clusters (without removing the condition legend)
    handles = []
    labels = []
    for cluster, color in cluster_palette.items():
        handles.append(plt.Line2D([0], [0], color=color, lw=10))
        labels.append(f"{cluster}")
    # add another axis for the legend
    ax2 = ax[1].twinx()
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.legend(
        handles,
        labels,
        **legend_params,
        title="Clusters",
        loc="upper right",
        bbox_to_anchor=(1.1, 0.5)
    )

    # Third Plot shows the log10 intensity values
    sns.lineplot(
        ax=ax[2],
        data=plot_data,
        x="PeptideID",
        y="log10Intensity",
        hue="Condition",
        palette=condition_palette,
        **lineplot_params
    )

    ax[2].set_ylabel("Log10 Raw Intensity", fontsize=12)
    ax[2].set_xlabel(f"PeptideIDs in {cur_protein} (Ordered by Start Position)", fontsize=12)
    ax[2].legend(**legend_params, title="Condition", loc="upper right", bbox_to_anchor=(1.1, 1))
    
    for i in range(3):
        ax[i].grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")
        for j in subset.loc[subset["pertPeptide"], ["PeptideID"]].drop_duplicates().values:
            ax[i].axvline(
                j,
                color='black', 
                linestyle='--', 
                linewidth=1.5, 
                alpha=0.5, 
                zorder=0
            )

    ax[0].set_title(
        f"Detailed Quantitative Look at ({cur_protein}) ", 
        fontsize=14, loc="left", fontweight="bold", pad=20
    )
    # Add a descriptive text below the title
    ax[0].text(
        0, 1.025, 
        f"Adjusted and Raw Intensities as well as the Adjusted P-value from the linear model interaction shown across peptides.",
        ha="left", va="center", fontsize=12, fontstyle="italic", 
        transform=ax[0].transAxes
    )

    # Align y-axis labels
    fig.align_ylabels(ax)
    plt.tight_layout()

################################### New Plotting Functions ###################################

def detailed_peptide_with_clusters(
        data: pd.DataFrame,    
        cur_protein: str,
        ## Varibles to be used
        pThr: float = 0.0001,

        ## Columns to be used
        pvalue_col: str  = "adj.pvalue",
        cluster_col: str  = "cluster_id",
        condition_col: str  = "Condition",
        protein_col: str = "Protein",
        peptide_col: str = "PeptideID",
        rawIntensity_col: str  = "log10(Intensity)",
        adjIntensity_col: str  = "adjIntensity",
        
        ## Palettes
        condition_palette: dict = None,
        cluster_palette: dict = None,
        # Plotting parameters
        figsize: tuple = (12, 9),
        # Saving options
        save: bool = False,
        dont_show: bool = False,
        filename: str = 'detailed_peptide_with_clusters',
        filepath: str = '',
        fileformats: list[str] = ['png', ],
        dpi: int = 100,
        transparent: bool = False 
    ):

    ## Plot Setup
    ## Define the kwargs for plots
    lineplot_params = {
        "alpha": 0.75,
        "marker": "o",
        "markersize": 10,
        "markeredgewidth": 0,
        "linewidth": 1.5,
        "linestyle": "-",
        "dashes": False,
        "err_style": "bars",
        "errorbar": ('se', 1.5),
        "err_kws": {
            "capsize": 5, 
            "elinewidth": 2.5, 
            "capthick": 2.5, 
            "zorder": 1, 
            "linewidth": 1.5
        },
    }
    scatterplot_params = {
        "s": 125,
        "edgecolor": "None",
        "linewidth": 1,
        "alpha": 0.75,
        "markers": {True: "*", False: "s"}
    }
    legend_params = {
        "fontsize": 10,
        "title_fontsize": 12,
        "edgecolor": "black",
        "facecolor": "white",
        "frameon": False,
        "ncol": 1,
        "markerscale": 1.5,
        "labelspacing": 0.5,
    }

    # Copy the data
    plot_data = data[data[protein_col] == cur_protein].copy()
    plot_data['isSignificant'] = plot_data[pvalue_col] <= pThr
    plot_data["-log10(adj.pval)"] = -np.log10(plot_data[pvalue_col])
    
    if condition_palette is None:
        colors = sns.color_palette(
            "Set1", n_colors=len(plot_data[condition_col].unique())
        ).as_hex()
        condition_palette = dict(zip(plot_data[condition_col].unique(), colors))
    if cluster_palette is None:
        colors = sns.color_palette(
            "Set2", n_colors=len(plot_data[cluster_col].unique())
        ).as_hex()
        cluster_palette = dict(zip(plot_data[cluster_col].unique(), colors))

    ## Initialize the figure
    fig, ax = plt.subplots(
        figsize=figsize, ncols=1, nrows=3, 
        sharex=True, sharey=False,
        gridspec_kw={ "height_ratios": [1, .5, 1], "hspace": 0.01 }
    )
    # First Plot shows the adjusted intensity values
    sns.lineplot(
        ax=ax[0],
        data=plot_data,
        x=peptide_col,
        y=adjIntensity_col,
        hue=condition_col,
        palette=condition_palette,
        **lineplot_params
    )
    ax[0].set_ylabel("Adjusted Intensity", fontsize=12)
    ax[0].legend(**legend_params, title="Condition", loc="upper right", bbox_to_anchor=(1.1, 1))

    # subset = plot_data[[peptide_col, pvalue_col]].drop_duplicates()
    # for i, row in subset[subset[pvalue_col] < pThr].iterrows():
    #     ax[0].text(
    #         row[peptide_col],
    #         0 ,
    #         "*",
    #         fontsize=25,
    #         color="black",
    #         ha="center",
    #         va="center",
    #     )
    # Create a cluster
    pepClusters = plot_data[[
        peptide_col, '-log10(adj.pval)', cluster_col, 'isSignificant'
    ]].drop_duplicates()
    # Second Plot shows the adjusted p-values
    sns.scatterplot(
        ax=ax[1],
        data=pepClusters,
        x=peptide_col,
        y='-log10(adj.pval)',
        style='isSignificant',
        color='k',
        **scatterplot_params
    )
    ax[1].set_ylabel("-Log10(FDR)", fontsize=12)
    ax[1].legend(**legend_params, title="Singf.", loc="upper right", bbox_to_anchor=(1.1, 1))

    # Draw a line at preferred p-value threshold
    ax[1].axhline(
        -np.log10(pThr),
        color="salmon",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        zorder=0
    )

    offset = 0.1
    # Draw rectangles as background for clusters in the second plot
    minLine = pepClusters["-log10(adj.pval)"].min() - offset
    maxLine = pepClusters["-log10(adj.pval)"].max() + offset
    for i, row in pepClusters.iterrows():
        ax[1].add_patch(
            plt.Rectangle(
                (row[peptide_col] - 0.5, minLine), 1, maxLine - minLine,
                color=cluster_palette[row[cluster_col]],
                alpha=0.5, linewidth=1.5, edgecolor="black",
                zorder=1, 
            )
        )

    # Add a secondary legend for the clusters (without removing the condition legend)
    handles, labels = [], []
    for cluster, color in cluster_palette.items():
        handles.append(plt.Line2D([0], [0], color=color, lw=10))
        labels.append(f"{cluster}")
    # add another axis for the legend
    ax2 = ax[1].twinx()
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.legend(
        handles, labels,
        **legend_params,
        title="Clusters",
        loc="upper right",
        bbox_to_anchor=(1.1, 0.5), 
    )

    # Third Plot shows the log10 intensity values
    sns.lineplot(
        ax=ax[2],
        data=plot_data,
        x=peptide_col,
        y=rawIntensity_col,
        hue=condition_col,
        palette=condition_palette,
        legend=False,
        **lineplot_params
    )

    ax[2].set_ylabel("Log10 Raw Intensity", fontsize=12)
    ax[2].set_xlabel(f"PeptideIDs in {cur_protein} (Ordered by Start Position)", fontsize=12)

    for i in range(3):
        ax[i].grid("both", linestyle="--", linewidth=0.5, alpha=0.5, color="lightgrey")

    ax[0].set_title(
        f"Detailed Quantitative Look at ({cur_protein}) ", 
        fontsize=14, loc="left", fontweight="bold", pad=20
    )
    # Add a descriptive text below the title
    ax[0].text(
        0, 1.025, 
        f"Adjusted and Raw Intensities as well as the Adjusted P-value from the linear model interaction shown across peptides.",
        ha="left", va="center", fontsize=12, fontstyle="italic", 
        transform=ax[0].transAxes
    )

    # Align y-axis labels
    fig.align_ylabels(ax)
    plt.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig

def heatmap_with_clusters(
        data: pd.DataFrame,
        cur_protein: str,
        pThr: float = 0.0001,
        corrMethod: str = 'kendall',
        distanceMetric: str = 'euclidean',
        linkageMethod: str = 'complete',
        
        ## Columns
        hue_col: str = 'TumorRegulation',
        pvalue_col: str = 'adj.pvalue',
        cluster_col: str = 'cluster_id',
        protein_col: str = 'Protein',
        peptideid_col: str = 'PeptideID',
        intensity_col: str = 'Intensity',
        sample_col: str = 'Sample',

        ## Palette    
        hue_palette: dict = None, 
        cluster_palette: dict = None, 

        ## Fi
        figsize: tuple = (4, 4),
        vmin: float = -.85,
        vmax: float = .85,
        main_cmap: str = "coolwarm",

        save: bool = False, 
        filename: Optional[str] = None, 
        filepath: Optional[str] = None, 
        fileformats: List[str] = ['png'], 
        dpi: int = 300, 
        transparent: bool = False, 
        dont_show: bool = False

    ):
    """
        
    """
    # Set the plot data
    plot_data = data[data[protein_col] == cur_protein].copy()
    plot_data['isSignificant'] = plot_data[pvalue_col] <= pThr
    plot_data['-log10(adj.pvalue)'] = -np.log10(plot_data[pvalue_col])
    # Setup the annotation data to be used
    corr_annot = plot_data[[
        peptideid_col, '-log10(adj.pvalue)', 'isSignificant',
        hue_col, cluster_col
    ]].drop_duplicates().set_index(peptideid_col)
    corr_annot["Annot"] = corr_annot.index
    corr_annot.loc[corr_annot["isSignificant"] == 0, "Annot"] = np.nan
    # Setup the correlation matrix 
    corr_matrix = (
        1 - plot_data.pivot(
            index=peptideid_col,
            columns=sample_col,
            values=intensity_col
        )
    ).T.corr(method=corrMethod).stack()
    corr_matrix.index.names = ["level_0", "level_1"]
    corr_matrix = corr_matrix.reset_index(name="Correlation")

    if cluster_palette is None:
        colors = sns.color_palette(
            "Set2", n_colors=len(plot_data[cluster_col].unique())
        ).as_hex()
        cluster_palette = dict(zip(plot_data[cluster_col].unique(), colors))

    # Setup row annotations
    row_ha = pch.HeatmapAnnotation(
        Module=pch.anno_simple(
            corr_annot[cluster_col], colors=cluster_palette, legend=False, height=5,
            add_text=True, text_kws={'color':'black','fontsize':12}
        ),
        # ProteoformGroup=pch.anno_simple(
        #     corr_annot["PFGroup"], colors=pfgPal, legend=False, height=5,
        #     add_text=True, text_kws={'color':'black','fontsize':12}
        # ),
        Significant=pch.anno_simple(
            corr_annot["isSignificant"], colors={True: '#540b0e', False: '#fff3b0'}, legend=True,
        ),
        Selected=pch.anno_label(
            corr_annot["Annot"], height=5, fontsize=12, colors='#540b0e'
        ),
        axis=0,verbose=0,label_kws={'visible':False}
    )
    # Setup column annotations
    col_ha = pch.HeatmapAnnotation(
        Cluster=pch.anno_simple(
            corr_annot[cluster_col], colors=cluster_palette, height=5, legend=False,
            add_text=True, text_kws={'color':'black','fontsize':12}
        ),
        # AdjPvalue=pch.anno_barplot(
        #     corr_annot["-log10(adj.pvalue)"], cmap='Greys',
        #     #
        # ),
        TumorRegulation=pch.anno_simple(
            corr_annot[hue_col], colors=hue_palette,
        ),
        verbose=0, label_side='right', label_kws={'horizontalalignment':'left'}
    )

    # Find the peptides for the protein
    nPeps = plot_data[peptideid_col].nunique()
    figsize = (figsize[0]+(nPeps/10), figsize[1]+(nPeps/10))

    # Initialize the figure
    fig, ax = plt.subplots(figsize=figsize)
    if np.any(corr_annot[cluster_col].value_counts() == 1):
        # Not using splitting of the clusters
        cm = pch.DotClustermapPlotter(
            data=corr_matrix, x='level_0',y='level_1',
            value='Correlation', c='Correlation',
            top_annotation=col_ha, right_annotation=row_ha,
            row_dendrogram=True, col_dendrogram=True,
            row_cluster_metric=distanceMetric, row_cluster_method=linkageMethod,
            col_cluster_metric=distanceMetric, col_cluster_method=linkageMethod,
            cmap=main_cmap, vmin=vmin, vmax=vmax,
            legend_hpad=1, legend_vpad=1, legend_gap=5,
            verbose=0, alpha=2, spines=False, tree_kws=dict(row_cmap='Set1',linewidth=0.5),
        )
    else:
        cm = pch.DotClustermapPlotter(
            data=corr_matrix, x='level_0',y='level_1',
            value='Correlation', c='Correlation',
            top_annotation=col_ha, right_annotation=row_ha,
            col_split=corr_annot[cluster_col], row_split=corr_annot[cluster_col], 
            col_split_gap=1, row_split_gap=1,
            row_dendrogram=True, 
            row_cluster_metric=distanceMetric, row_cluster_method=linkageMethod,
            col_cluster_metric=distanceMetric, col_cluster_method=linkageMethod,
            cmap=main_cmap, vmin=vmin, vmax=vmax,
            legend_hpad=1, legend_vpad=1, legend_gap=5,
            verbose=0, alpha=2, spines=False, tree_kws=dict(row_cmap='Set1',linewidth=0.5),
        )
        # plot custom spines
        for i in range(cm.heatmap_axes.shape[0]):
            for j in range(cm.heatmap_axes.shape[1]):
                if i != j:
                    continue
                ax = cm.heatmap_axes[i][j]
                for side in ["top", "right", "left", "bottom"]:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color('black')
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_zorder(10)

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig
        
def donut_chart(
        data: list,
        labels: list,
        colors: list,
        figsize: tuple = (6, 6),
        startangle: int = 90,
        text_color: str = "k",
        text_fontsize: int = 15,
        autotext_fontsize: int = 12,
        autotext_color: str = 'black',
        autotext_weight: str = 'bold',
        circle_radius: float = 0.70,
        circle_color: str = 'white',
        # label_as_legend: bool = True,
        ## Saving options
        save: bool = False, 
        filename: Optional[str] = "donut_chart", 
        filepath: Optional[str] = None, 
        fileformats: List[str] = ['png'], 
        dpi: int = 300, 
        transparent: bool = False, 
        dont_show: bool = False

    ):
    """
        Create a donut chart with the given data and parameters.

        Args:
    """

    fig, ax = plt.subplots(figsize=figsize)
    
    # Function to format the autopct
    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    
    # Plot
    wedges, texts, autotexts = ax.pie(
        data, labels=labels, colors=colors, autopct=lambda pct: func(pct, data), startangle=startangle, 
        textprops=dict(color=text_color)
    )

    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0, 0), circle_radius, fc=circle_color)
    fig.gca().add_artist(centre_circle)
    
    # Adjust text sizes
    for text in texts:
        text.set_fontsize(text_fontsize)
    for autotext in autotexts:
        autotext.set_fontsize(autotext_fontsize)
        autotext.set_color(autotext_color)
        autotext.set_fontweight(autotext_weight)
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')  
    plt.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformats,
            dpi=dpi,
            transparent=transparent
        )
        if dont_show:
            plt.close(fig)
    else:
        if dont_show:
            plt.close(fig)
        else:
            return fig