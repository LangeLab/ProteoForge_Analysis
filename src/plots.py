import numpy as np
import pandas as pd 

import seaborn as sns

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Patch

from matplotlib.colors import TwoSlopeNorm

from typing import Union, Optional, Tuple, List, Dict, Any

# ======================================================================================
# Global Variables and Settings
# ======================================================================================
# TODO list: 
# - Move the styling defaults to here
# - Add the default color palettes for plots
# - Include more plotting functions...



# ======================================================================================
# Utility Plotting Functions
# ======================================================================================

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

def save_plot(
        fig: Figure,
        filename: str,
        filepath: str = '',
        formats: Optional[List[str]] = None,
        verbose: bool = False,
        **savefig_kwargs: Any,
    ):
    """
    Saves a Matplotlib figure, assuming the target directories already exist.

    This function uses the path structure 'filepath/format/filename.format'.

    Args:
        fig (Figure): The Matplotlib figure object.
        filename (str): The base name for the saved file (without an extension).
        filepath (str): The root directory where the format subdirectories are located.
        formats (Optional[List[str]]): A list of file formats.
            Defaults to ['png', 'svg', 'pdf'].
        **savefig_kwargs (Any): Additional keyword arguments passed to fig.savefig().
            These will override the default settings below.
    """
    if formats is None:
        formats = ['png', 'svg', 'pdf']

    # Default save options that can be overridden by the user via **savefig_kwargs
    default_options: Dict[str, Any] = {
        'dpi': 300,
        'transparent': True,
        'bbox_inches': 'tight',
        'pad_inches': 0.01
    }
    # User's provided kwargs take precedence over the defaults
    final_save_options = {**default_options, **savefig_kwargs}

    for fmt in formats:
        save_path = f"{filepath}/{fmt}/{filename}.{fmt}"
        
        fig.savefig(save_path, format=fmt, **final_save_options)
        if verbose: print(f"Figure saved to: {save_path}")

def finalize_plot(
        fig: Figure,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs: Any
    ):
    """
    A wrapper to conveniently show, save, and/or close a Matplotlib figure.

    Args:
        fig (Figure): The Matplotlib figure object to process.
        show (bool): If True, display the figure interactively.
        save (bool): If True, save the figure using the save_plot() function.
        filename (Optional[str]): The base name for the file. Required if save=True.
        **kwargs (Any): Other keyword arguments ('filepath', 'formats', 'dpi', etc.)
                        are passed directly to the save_plot() function.
    """
    if save:
        if not filename:
            raise ValueError("A 'filename' must be provided when 'save' is True.")
        # Pass the figure, filename, and all other keyword arguments to save_plot
        save_plot(fig=fig, filename=filename, **kwargs)

    if show:
        plt.show()
    else:
        # If not showing the plot, close it to free up memory.
        plt.close(fig)


# Plots a color palette with added functionality
def color_palette(
        pal: Union[list, dict],
        size: int = 1, 
        name: str = "default colors",
        show: bool = True,
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
            show (bool): If the plot should be shown
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

    finalize_plot( 
        f, show, save, filename,
        # Arguments
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

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

# ======================================================================================
# Data Quality Check Visualizations
# ======================================================================================

def normalized_density_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    condition_map: Dict[str, List[str]],
    color_palette: Dict[str, str],
    log_transform: bool = True,
    title: str = "Distribution of Intensities Before and After Normalization",
    figsize: Tuple[int, int] = (16, 4),
    linewidth: float = 1.5,
    alpha: float = 0.6,
    xlabel: Optional[str] = None,
    ylabel: str = "Density",
    before_title: str = "Before Normalization",
    after_title: str = "After Normalization",
    legend_title: str = "Condition",
    legend_loc: str = 'upper right',
    legend_bbox: Tuple[float, float] = (0.98, 0.92),
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 12,
    grid: bool = True,
    grid_style: str = '--',
    grid_width: float = 0.75,
    grid_alpha: float = 0.7,
    despine: bool = True,
    show: bool = True,
    save: bool = False,
    filename: str = 'normalized_density_comparison',
    fileformats: List[str] = ['png', 'svg', 'pdf'],
    filepath: str = '',
    transparent: bool = True,
    dpi: int = 300,
):
    """
    Generates side-by-side KDE plots to compare data distributions before and after a transformation.

    TODO: Docstring
    """
    def _plot_single_kde(ax, data_df, plot_title):
        plot_data = np.log2(data_df) if log_transform else data_df
        xlab = xlabel if xlabel is not None else ("log2(Intensity)" if log_transform else "Intensity")
        for condition, columns in condition_map.items():
            color = color_palette.get(condition, 'grey')
            for col in columns:
                if col in plot_data.columns:
                    sns.kdeplot(
                        plot_data[col].dropna(),
                        ax=ax,
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha
                    )
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel(xlab, fontsize=12)
        if grid:
            ax.grid(axis='y', linestyle=grid_style, linewidth=grid_width, alpha=grid_alpha)
        if despine:
            sns.despine(ax=ax, top=True, right=True)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=18, y=0.98)

    _plot_single_kde(axes[0], before_df, before_title)
    axes[0].set_ylabel(ylabel, fontsize=12)
    _plot_single_kde(axes[1], after_df, after_title)

    legend_handles = [
        Patch(facecolor=color, label=condition, alpha=0.8)
        for condition, color in color_palette.items()
        if condition in condition_map
    ]
    fig.legend(
        handles=legend_handles,
        title=legend_title,
        title_fontsize=legend_title_fontsize,
        fontsize=legend_fontsize,
        frameon=False,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

def downshift_effect(
    df: pd.DataFrame,
    cond_dict: Dict[str, List[str]],
    is_log2: bool = False,
    shift_magnitude: float = 1.8,
    low_percentile: float = 0.10,
    sample_size: int = 1000,
    fill_alpha: float = 0.5,
    orig_color: str = 'skyblue',
    imputed_color: str = 'salmon',
    orig_mean_color: str = 'blue',
    imputed_mean_color: str = 'red',
    mean_linestyle: str = '--',
    xlabel: str = "Value",
    ylabel: str = "Density",
    title_prefix: str = "Condition: ",
    suptitle: Optional[str] = None,
    suptitle_fontsize: int = 16,
    suptitle_y: float = 1.03,
    legend_fontsize: int = 11,
    legend_loc: str = 'best',
    n_cols: Optional[int] = None,
    figsize_per_col: int = 8,
    figsize_per_row: int = 4,
    tight_layout_rect: Tuple[float, float, float, float] = (0, 0, 1, 0.96),
    show: bool = True,
    save: bool = False,
    filename: str = 'downshift_effect',
    fileformats: List[str] = ['png', 'svg', 'pdf'],
    filepath: str = '',
    transparent: bool = True,
    dpi: int = 300,
):
    """
    Visualizes the effect of the downshift imputation parameters for each condition.

    TODO: Docstring
    """

    def _find_downshift_params(
        data: pd.DataFrame, shift_magnitude: float, low_percentile: float
    ) -> Tuple[float, float]:
        """Internal helper to find the downshifted mean and low-value threshold."""
        quantified_values = data.values.flatten()
        quantified_values = quantified_values[~np.isnan(quantified_values)]
        if quantified_values.size == 0:
            return 0, 0
        low_value_threshold = np.percentile(quantified_values, low_percentile * 100)
        low_value_distribution = quantified_values[quantified_values < low_value_threshold]
        if low_value_distribution.size == 0:
            downshifted_mean = low_value_threshold - shift_magnitude
        else:
            downshifted_mean = low_value_distribution.mean() - shift_magnitude
        return downshifted_mean, low_value_threshold
    
    if not cond_dict:
        print("Visualization skipped: A condition dictionary is required.")
        return

    if is_log2:
        data = df.copy()
    else:
        data = np.log2(df).copy()

    n_conditions = len(cond_dict)
    if n_conditions == 0: return

    if n_cols is None: n_cols = 2 if n_conditions > 1 else 1
    n_rows = (n_conditions + n_cols - 1) // n_cols
    figsize = (figsize_per_col * n_cols, figsize_per_row * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    if suptitle is None:
        suptitle = f"Downshift Effect Analysis\n(shift_magnitude={shift_magnitude}, low_percentile={low_percentile})"
    fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)

    for i, (condition, samples) in enumerate(cond_dict.items()):
        ax = axes[i]
        valid_samples = [s for s in samples if s in data.columns]
        if not valid_samples:
            ax.text(0.5, 0.5, "No valid samples.", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title_prefix}{condition}", fontstyle='italic', fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        condition_data = data[valid_samples]
        downshifted_mean, low_value_threshold = _find_downshift_params(condition_data, shift_magnitude, low_percentile)
        quantified_values = condition_data.values.flatten()
        quantified_values = quantified_values[~np.isnan(quantified_values)]
        if quantified_values.size == 0:
            ax.text(0.5, 0.5, "No quantified data.", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title_prefix}{condition}", fontstyle='italic', fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        low_value_dist = quantified_values[quantified_values < low_value_threshold]
        if low_value_dist.size < 2:
            ax.text(0.5, 0.5, "Not enough low values to plot.", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title_prefix}{condition}", fontstyle='italic', fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        imputed_dist_sample = np.random.normal(
            loc=downshifted_mean,
            scale=np.std(low_value_dist, ddof=1),
            size=sample_size
        )
        imputed_dist_sample = np.clip(imputed_dist_sample, a_min=None, a_max=low_value_threshold)

        bins = min(30, max(10, int(np.sqrt(low_value_dist.size))))
        # Plot original low values histogram
        ax.hist(
            low_value_dist, bins=bins, color=orig_color, alpha=fill_alpha,
            label=f'Original Low Values (n={len(low_value_dist)})', density=True, edgecolor='black', linewidth=1
        )
        # Plot imputed values histogram
        ax.hist(
            imputed_dist_sample, bins=bins, color=imputed_color, alpha=fill_alpha,
            label=f'Imputed Values (Sample)', density=True, edgecolor='black', linewidth=1
        )
        # Mean lines
        ax.axvline(np.mean(low_value_dist), color=orig_mean_color, linestyle=mean_linestyle,
                   label=f'Mean (Original): {np.mean(low_value_dist):.2f}')
        ax.axvline(downshifted_mean, color=imputed_mean_color, linestyle=mean_linestyle,
                   label=f'Mean (Imputed): {downshifted_mean:.2f}')
        ax.set_title(f"{title_prefix}{condition}", fontstyle='italic', fontsize=13)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)
        ax.grid(True, linestyle='--', alpha=0.5, color='lightgrey', linewidth=0.75)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=tight_layout_rect)

    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

def imputation_distribution_per_condition(
        original_data: pd.DataFrame,
        imputed_data: pd.DataFrame,
        cond_dict: Dict[str, List[str]],
        is_log2: bool = False,
        title: str = "Distribution of Original vs. Imputed Values Per Condition",
        bins: str = 'auto',
        orig_color: str = 'skyblue',
        imputed_color: str = 'salmon',
        orig_alpha: float = 0.6,
        imputed_alpha: float = 0.6,
        xlabel: str = "Value",
        ylabel: str = "Count",
        legend_fontsize: int = 11,
        legend_loc: str = 'upper right',
        grid: bool = True,
        grid_style: str = '--',
        grid_alpha: float = 0.5,
        grid_color: str = 'lightgrey',
        grid_width: float = 0.75,
        suptitle_fontsize: int = 18,
        suptitle_y: float = 0.98,
        tight_layout_rect: tuple = (0, 0, 1, 0.96),
        show: bool = True,
        save: bool = False,
        filename: str = 'imputation_distribution_per_condition',
        fileformats: list = ['png', 'svg', 'pdf'],
        filepath: str = '',
        transparent: bool = True,
        dpi: int = 300,
    ):
    """
    Visualizes and compares the distributions of original and imputed values per condition.
    """
    n_conditions = len(cond_dict)
    if n_conditions == 0:
        print("No conditions to plot.")
        return

    # Apply log2 transform if needed
    if not is_log2:
        original_data = np.log2(original_data)
        imputed_data = np.log2(imputed_data)

    # Prepare subplot grid
    n_cols = min(3, n_conditions)
    n_rows = (n_conditions + n_cols - 1) // n_cols
    figsize = (n_cols * 5, n_rows * 4)

    # Compute global min/max for all data to ensure comparable bins and axes
    all_values = []
    for samples in cond_dict.values():
        orig_vals = original_data[samples].values.flatten() if samples else np.array([])
        imp_vals = imputed_data[samples].values.flatten() if samples else np.array([])
        all_values.append(orig_vals)
        all_values.append(imp_vals)
    all_values = np.concatenate([v[~np.isnan(v)] for v in all_values if v.size > 0])
    if all_values.size == 0:
        print("No data to plot.")
        return
    global_min = np.nanmin(all_values)
    global_max = np.nanmax(all_values)
    if bins == 'auto':
        bins_used = np.linspace(global_min, global_max, 31)
    else:
        bins_used = bins

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    axes = axes.flatten()

    plotted_any = False
    for i, (cond, samples) in enumerate(cond_dict.items()):
        ax = axes[i]
        if not samples:
            ax.text(0.5, 0.5, "No samples", ha='center', va='center', fontsize=11, color='grey', transform=ax.transAxes)
            ax.set_title(str(cond), fontsize=13, fontstyle='italic')
            continue

        orig_vals = original_data[samples].values.flatten()
        imp_vals = imputed_data[samples].values.flatten()
        # Imputed values: those that were missing in original but present in imputed
        orig_flat = original_data[samples]
        imp_flat = imputed_data[samples]
        mask = orig_flat.isna() & imp_flat.notna()
        imputed_only = imp_flat[mask].values.flatten()
        orig_only = orig_vals[~np.isnan(orig_vals)]

        if orig_only.size > 0:
            ax.hist(
                orig_only, color=orig_color, label='Original', alpha=orig_alpha,
                bins=bins_used, edgecolor='black', linewidth=0.8
            )
        if imputed_only.size > 0:
            ax.hist(
                imputed_only, color=imputed_color, label='Imputed', alpha=imputed_alpha,
                bins=bins_used, edgecolor='black', linewidth=0.8
            )
            plotted_any = True

        ax.set_title(str(cond), fontsize=13, fontstyle='italic')
        row_idx = i // n_cols
        col_idx = i % n_cols
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        else:
            ax.set_ylabel("")
        if row_idx == n_rows - 1:
            ax.set_xlabel(xlabel, fontsize=11)
        else:
            ax.set_xlabel("")
        if grid:
            ax.grid(axis='both', linestyle=grid_style, alpha=grid_alpha, color=grid_color, linewidth=grid_width)
        if (orig_only.size > 0 or imputed_only.size > 0):
            ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=11, color='grey', transform=ax.transAxes)
        sns.despine(ax=ax)

    # Remove unused axes
    for j in range(n_conditions, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=suptitle_fontsize, y=suptitle_y)
    plt.tight_layout(rect=tight_layout_rect)

    if not plotted_any:
        print("Visualization skipped: No imputed values found in any condition.")
        plt.close(fig)
        return

    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

def cv_comparison(
    plot_data: pd.DataFrame,
    cv_group_palettes: dict,
    condition_colors: dict,
    condition_order: Optional[List[str]] = None,
    cv_col: str = "CV",
    condition_col: str = "Condition",
    cvgroup_col: str = "CVGroup",
    figsize: Tuple[int, int] = (15, 4),
    main_title: str = "Coefficient of Variation (CV) Comparison",
    bar_width_ratios: List[int] = [1, 1],
    bar_wspace: float = 0.05,
    bar_title: str = "Count of proteins by CV Group",
    box_title: str = "Distribution of CV in proteins",
    bar_xlabel: str = "Count",
    box_xlabel: str = "CV (%)",
    bar_legend_ncol: int = 5,
    bar_legend_loc: str = "upper center",
    bar_legend_bbox: Tuple[float, float] = (0.5, 1.105),
    box_legend_title: str = "Group Median",
    box_legend_loc: str = "upper right",
    box_legend_bbox: Tuple[float, float] = (1.05, 1.05),
    box_legend_ncol: int = 1,
    grid_style: str = "--",
    grid_alpha: float = 0.5,
    grid_color: str = "lightgrey",
    grid_width: float = 0.75,
    bar_title_fontsize: int = 12,
    box_title_fontsize: int = 12,
    show: bool = True,
    save: bool = False,
    filename: str = "cv_comparison",
    fileformats: List[str] = ['png', 'svg', 'pdf'],
    filepath: str = '',
    transparent: bool = True,
    dpi: int = 300,
):
    """
    Creates a 2-panel CV comparison plot: 
    1. Stacked horizontal barplot of CV groups per condition.
    2. Boxplot (or violin) of CV distribution per condition, with group medians.

    TODO: Docstring
    """
    # Order conditions if provided
    if condition_order is not None:
        plot_data = plot_data.copy()
        plot_data[condition_col] = pd.Categorical(plot_data[condition_col], categories=condition_order, ordered=True)

    # --- Prepare statistics table ---
    # Calculate mean and median
    stats_table = (
        plot_data
        .groupby(condition_col)
        .agg(
            Mean_CV=(cv_col, "mean"),
            Median_CV=(cv_col, "median"),
        )
        .reset_index()
    )
    # Calculate counts for each CVGroup
    cvgroup_counts = (
        plot_data
        .groupby([condition_col, cvgroup_col], observed=False)
        .size()
        .unstack(fill_value=0)
        .add_prefix("Count_")
        .reset_index()
    )
    # Merge the tables
    stats_table = stats_table.merge(cvgroup_counts, on=condition_col, how="left")
    # Format float columns
    stats_table["Mean_CV"] = stats_table["Mean_CV"].round(2)
    stats_table["Median_CV"] = stats_table["Median_CV"].round(2)

    # --- Setup figure with 3 panels (bar, box, table) ---

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=bar_width_ratios + [0.7],
        wspace=bar_wspace
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1], sharey=ax_bar)
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")

    # --- Panel 1: Stacked Barplot of CV Groups ---
    cv_counts = plot_data.groupby([condition_col, cvgroup_col], observed=False).size().unstack(fill_value=0)
    cv_counts = cv_counts.loc[condition_order] if condition_order is not None else cv_counts
    cv_counts.plot(
        ax=ax_bar,
        kind="barh",
        stacked=True,
        color=[cv_group_palettes.get(g, "#cccccc") for g in cv_counts.columns],
        edgecolor="black",
        linewidth=0.5,
        legend=False
    )
    ax_bar.set_xlabel(bar_xlabel)
    ax_bar.set_ylabel("")
    ax_bar.set_title(bar_title, pad=25, fontsize=bar_title_fontsize, fontstyle='italic')
    ax_bar.grid("both", linestyle=grid_style, linewidth=grid_width, alpha=grid_alpha, color=grid_color)
    # Custom legend for CV groups
    handles = [
        Patch(facecolor=cv_group_palettes.get(g, "#cccccc"), label=str(g))
        for g in cv_counts.columns
    ]
    ax_bar.legend(
        handles=handles, title="", frameon=False, bbox_to_anchor=bar_legend_bbox, 
        ncol=bar_legend_ncol, loc=bar_legend_loc
    )

    # --- Panel 2: CV Distribution as Boxplot ---
    sns.boxplot(
        data=plot_data,
        x=cv_col,
        y=condition_col,
        ax=ax_box,
        color="lightgrey",
        linewidth=0.75,
        width=0.75,
        fliersize=0.5,
        order=condition_order
    )
    ax_box.set_xlabel(box_xlabel)
    ax_box.set_ylabel("")
    ax_box.set_title(box_title, pad=25, fontsize=box_title_fontsize, fontstyle='italic')
    ax_box.grid("both", linestyle=grid_style, linewidth=grid_width, alpha=grid_alpha, color=grid_color)

    # Add median lines for each condition
    for cond in (condition_order if condition_order is not None else plot_data[condition_col].unique()):
        median_val = plot_data.loc[plot_data[condition_col] == cond, cv_col].median()
        ax_box.axvline(
            median_val,
            color=condition_colors.get(cond, "#333333"),
            linestyle="--",
            linewidth=1,
            label=f"{cond}: {median_val:.2f}%"
        )

    # Only show one legend for medians, outside the plot
    handles, labels = ax_box.get_legend_handles_labels()
    ax_box.legend(
        handles=handles, labels=labels, title=box_legend_title, frameon=False,
        bbox_to_anchor=box_legend_bbox, loc=box_legend_loc, ncol=box_legend_ncol
    )

    # --- Panel 3: Table of statistics ---
    # Prepare table data, transpose if more columns than rows
    table_df = stats_table.copy()
    transpose_table = table_df.shape[0] < table_df.shape[1]
    if transpose_table:
        table_df = table_df.set_index(condition_col).T
        table_data = table_df.values.tolist()
        row_labels = table_df.index.tolist()
        col_labels = table_df.columns.tolist()
        mpl_tbl = ax_table.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
    else:
        table_data = table_df.values.tolist()
        col_labels = table_df.columns.tolist()
        mpl_tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
    mpl_tbl.auto_set_font_size(False)
    mpl_tbl.set_fontsize(10)
    mpl_tbl.scale(1.1, 1.3)
    ax_table.set_title("CV Stats per Condition", fontsize=bar_title_fontsize, pad=15, fontstyle='italic')

    # --- Auto-size columns based on label length ---
    # Find the max character length for each column
    if transpose_table:
        labels = col_labels
    else:
        labels = col_labels
    for i, label in enumerate(labels):
        # Estimate width: 0.12 per character, min 0.8, max 2.0
        width = min(max(0.12 * len(str(label)), 0.8), 2.0)
        mpl_tbl.auto_set_column_width([i])
        for key, cell in mpl_tbl.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    sns.despine(left=True, bottom=True)
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.15)

    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )


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
        show: bool = True,
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

    finalize_plot( 
        fig, show, save, filename,
        # Arguments
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )


# ======================================================================================
# QuEStVar Visualizations
# ======================================================================================

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
        # Finalize plot parameters
        show: bool = True,
        save: bool = False,
        filename: str = "powerProfile_line",
        fileformats: List[str] = ['png', 'svg', 'pdf'],
        filepath: str = '',
        transparent: bool = True,
        dpi: int = 300,
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

    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

def questvar_test_summary(
    quest_data, 
    p_thr=0.05, 
    eq_thr=0.5, 
    df_thr=0.75, 
    cv_thr=1.5,
    correction="fdr",
    # cond_1 / cond_2 for log2fc and test
    cond_1: str = 'Condition 1', 
    cond_2: str = 'Condition 2',
    # Figure parameters
    figsize=(20, 15),
    title_add: str = '',
    rasterize_scatters=True,
    status_colors=None,
    show_excluded=True,
    legend_fontsize=11,
    title_fontsize=18,
    
    # Finalize plot parameters
    show: bool = True,
    save: bool = False,
    filename: str = "questvar_test_summary",
    fileformats: List[str] = ['png', 'svg', 'pdf'],
    filepath: str = '',
    transparent: bool = True,
    dpi: int = 300,
):
    """
        Creates a comprehensive summary plot for QuestVar test results.

        TODO: Docstring
    """

    # Default colors if not provided
    if status_colors is None:
        status_colors = {
            'Unexplained': "#C2C0C0",    
            'Excluded': '#565d61',       
            'Upregulated': '#780000',    
            'Downregulated': '#e36414',  
            'Equivalent': '#003049',     
        }
    
    # Main title styling
    main_title_styles = {
        'fontsize': title_fontsize,
        'fontweight': 'bold',
        'color': 'black',
        'y': 0.96
    }
    
    # Subplot title styling
    subplot_title_styles = {
        'fontsize': title_fontsize - 4,
        # 'fontweight': 'bold',
        'fontstyle': 'italic',
        'color': 'black',
        'pad': 5
    }
    
    # Axis label styling
    axis_label_styles = {
        'fontsize': title_fontsize - 5,
        # 'fontweight': 'bold',
        'color': 'black',
        'labelpad': 5
    }
    
    # Tick label styling
    tick_label_styles = { 'labelsize': title_fontsize - 7, 'pad': 2 }
    
    # Legend styling
    legend_styles = {
        'fontsize': legend_fontsize - 1,
        'frameon': True,
        'fancybox': True,
        'shadow': True,
        'framealpha': 0.95,
        'handlelength': 2.0,
        'handletextpad': 0.4,
        'columnspacing': 0.4,
        'labelspacing': 0.3,
        'borderpad': 0.6
    }
    
    # Grid styling
    grid_styles = {
        'alpha': 0.3,
        'linestyle': '--',
        'linewidth': 0.5,
        'color': 'lightgray'
    }
    
    # Annotation box styling
    annotation_box_styles = {
        'boxstyle': "round,pad=0.4",
        'facecolor': 'lightblue',
        'alpha': 0.7,
        'edgecolor': 'navy',
        'linewidth': 0.5
    }
    
    # Subplot letter styling
    subplot_letter_styles = {
        'fontsize': title_fontsize,
        'fontweight': 'bold',
        'color': 'black',
        'bbox': dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=1),
        'transform': None,  # Will be set per subplot
        'ha': 'center',
        'va': 'center'
    }
    
    # Helper function to add subplot letters
    def add_subplot_letter(ax, letter, styles=subplot_letter_styles):
        """Add a letter annotation to subplot"""
        styles_copy = styles.copy()
        styles_copy['transform'] = ax.transAxes
        ax.text(0.02, 1.0, letter, **styles_copy)
    
    # Filter data if excluding certain categories
    plot_data = quest_data.copy()
    if not show_excluded:
        plot_data = plot_data[plot_data['Status'] != 'Excluded']

    # Create figure with optimized settings
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4, 5,  hspace=0.4, wspace=0.5, 
        height_ratios=[0.65, 0.65, 0.65, 0.65], 
        width_ratios=[0.75, 1.0, 1.0, 1.0, 1.0], 
        left=0.07, right=0.94, top=0.89, bottom=0.14
    )

    # =============================================================================
    # COLUMN 1: P-value Distributions
    # =============================================================================
    
    # T-test p-values
    ax1 = fig.add_subplot(gs[0, 0])
    add_subplot_letter(ax1, 'A')
    
    valid_df_p = plot_data['df_p'].dropna()
    valid_df_adjp = plot_data['df_adjp'].dropna()
    
    if len(valid_df_p) > 0:
        ax1.hist(valid_df_p, bins=30, alpha=0.6, color='#F8AD9D',
                label='Raw p-values', density=True, edgecolor='black', linewidth=0.5)
    if len(valid_df_adjp) > 0:
        ax1.hist(valid_df_adjp, bins=30, alpha=0.8, color='#bc4749',
                label=f'Adjusted ({correction})', density=True, edgecolor='black', linewidth=0.5)
    
    ax1.axvline(x=p_thr, color='black', linestyle='--', linewidth=1.5, 
               label=f'Threshold ({p_thr})', alpha=0.8)
    ax1.set_ylabel('Difference Testing\nP-value Density', **axis_label_styles)
    ax1.legend(**legend_styles)
    ax1.grid(True, **grid_styles)
    ax1.set_title('T-test P-values', **subplot_title_styles)
    ax1.tick_params(**tick_label_styles)
    
    # TOST p-values
    ax2 = fig.add_subplot(gs[1, 0])
    add_subplot_letter(ax2, 'B')
    
    valid_eq_p = plot_data['eq_p'].dropna()
    valid_eq_adjp = plot_data['eq_adjp'].dropna()
    
    if len(valid_eq_p) > 0:
        ax2.hist(valid_eq_p, bins=30, alpha=0.6, color='#A8DADC',
                label='Raw p-values', density=True, edgecolor='black', linewidth=0.5)
    if len(valid_eq_adjp) > 0:
        ax2.hist(valid_eq_adjp, bins=30, alpha=0.8, color='#457B9D',
                label=f'Adjusted ({correction})', density=True, edgecolor='black', linewidth=0.5)
    
    ax2.axvline(x=p_thr, color='black', linestyle='--', linewidth=1.5, 
               label=f'Threshold ({p_thr})', alpha=0.8)
    ax2.set_xlabel('P-value', **axis_label_styles)
    ax2.set_ylabel('Equivalence Testing\nP-value Density', **axis_label_styles)
    ax2.legend(**legend_styles)
    ax2.grid(True, **grid_styles)
    ax2.set_title('TOST P-values', **subplot_title_styles)
    ax2.tick_params(**tick_label_styles)
    
    # Combined p-value comparison
    ax3 = fig.add_subplot(gs[2, 0])
    add_subplot_letter(ax3, 'C')
    
    valid_mask = plot_data[['df_adjp', 'eq_adjp']].notna().all(axis=1)
    valid_data = plot_data[valid_mask]
    
    if len(valid_data) > 0:
        scatter_colors = [status_colors[status] for status in valid_data['Status']]
        scatter = ax3.scatter(
            valid_data['df_adjp'], valid_data['eq_adjp'], 
            c=scatter_colors, s=25, alpha=0.7,
            edgecolor='white', linewidth=0.3, 
            rasterized=rasterize_scatters
        )
    
    ax3.axhline(y=p_thr, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(x=p_thr, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Difference Test\nAdjusted P-value', **axis_label_styles)
    ax3.set_ylabel('Equivalence Test\nAdjusted P-value', **axis_label_styles)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, **grid_styles)
    ax3.set_title('Comparison', **subplot_title_styles)
    ax3.tick_params(**tick_label_styles)

    # =============================================================================
    # COLUMN 2-3: Antler's Plot (Modified Volcano)
    # =============================================================================
    
    ax4 = fig.add_subplot(gs[0:2, 1:3])
    add_subplot_letter(ax4, 'D')
    
    # Plot data by status with controlled order for proper layering
    plot_order = ['Excluded', 'Unexplained', 'Downregulated', 'Upregulated', 'Equivalent']
    
    for status in plot_order:
        if status in plot_data['Status'].unique():
            if not show_excluded and status == 'Excluded':
                continue
            subset = plot_data[plot_data['Status'] == status]
            if len(subset) > 0:
                ax4.scatter(
                    subset['log2FC'], subset['log10(adj_pval)'],
                    c=status_colors[status], label=status, s=40,
                    edgecolor='white', linewidth=0.3, alpha=0.8,
                    rasterized=rasterize_scatters, zorder=5
                )
    
    # Optimize plot limits for Antler's (Volcano) plot
    x_data = plot_data['log2FC'].dropna()
    y_data = plot_data['log10(adj_pval)'].dropna()

    if len(x_data) > 0 and len(y_data) > 0:
        # X-axis: symmetric around zero with offset
        x_abs_max = np.max(np.abs(x_data))
        x_offset = x_abs_max * 0.07 if x_abs_max > 0 else 1.0
        ax4.set_xlim(-(x_abs_max + x_offset), x_abs_max + x_offset)

        # Y-axis: use min/max with offset
        y_min, y_max = y_data.min(), y_data.max()
        y_offset = (y_max - y_min) * 0.07 if (y_max - y_min) > 0 else 1.0
        ax4.set_ylim(y_min - y_offset, y_max + y_offset)
    
    # Add threshold lines with proper z-ordering
    ax4.axhline(y=0, color='lightgray', linestyle='-', linewidth=1, alpha=0.6, zorder=1)
    ax4.axvline(x=0, color='lightgray', linestyle='-', linewidth=1, alpha=0.6, zorder=1)
    
    # Equivalence thresholds (blue dashed)
    ax4.axhline(y=np.log10(p_thr), color='#457B9D', linestyle='--', linewidth=2, 
               alpha=0.8, zorder=2)
    ax4.axvline(x=eq_thr, color='#457B9D', linestyle='--', linewidth=2, 
               alpha=0.8, zorder=2)
    ax4.axvline(x=-eq_thr, color='#457B9D', linestyle='--', linewidth=2, 
               alpha=0.8, zorder=2)
    
    # Difference thresholds (red dotted)
    ax4.axhline(y=-np.log10(p_thr), color='#bc4749', linestyle=':', linewidth=2, 
               alpha=0.8, zorder=2)
    ax4.axvline(x=df_thr, color='#bc4749', linestyle=':', linewidth=2, 
               alpha=0.8, zorder=2)
    ax4.axvline(x=-df_thr, color='#bc4749', linestyle=':', linewidth=2, 
               alpha=0.8, zorder=2)
    
    ax4.set_xlabel(f'log2 Fold Change ({cond_1} vs {cond_2})', **axis_label_styles)
    ax4.set_ylabel('log10 Adj. p-val (equiv.) | -log10 Adj. p-val (diff.)', **axis_label_styles)
    ax4.set_title("Antler's Plot: Equivalence + Difference Testing", **subplot_title_styles)
    ax4.grid(True, **grid_styles)
    ax4.tick_params(**tick_label_styles)

    # =============================================================================
    # COLUMN 4-5: MA Plot  
    # =============================================================================
    
    ax5 = fig.add_subplot(gs[0:2, 3:5])
    add_subplot_letter(ax5, 'E')
    
    # Plot with same order as Antler's plot
    for status in plot_order:
        if status in plot_data['Status'].unique():
            if not show_excluded and status == 'Excluded':
                continue
            subset = plot_data[plot_data['Status'] == status]
            if len(subset) > 0:
                ax5.scatter(
                    subset['average'], subset['log2FC'],
                    c=status_colors[status], label=status, s=40,
                    edgecolor='white', linewidth=0.3, alpha=0.8,
                    rasterized=rasterize_scatters, zorder=5
                )
    
    # Optimize MA plot limits with symmetric y-axis and proportional x-axis offset
    avg_data = plot_data['average'].dropna()
    fc_data = plot_data['log2FC'].dropna()
    
    if len(avg_data) > 0 and len(fc_data) > 0:
        # X-axis: add proportional offset for visibility
        x_min, x_max = avg_data.min(), avg_data.max()
        offset = (x_max - x_min) * 0.07 if (x_max - x_min) > 0 else 1.0
        ax5.set_xlim(x_min - offset, x_max + offset)
        
        # Y-axis: symmetric around zero with offset
        y_abs_max = np.max(np.abs(fc_data))
        y_offset = y_abs_max * 0.07 if y_abs_max > 0 else 1.0
        ax5.set_ylim(-(y_abs_max + y_offset), y_abs_max + y_offset)
    
    # Add threshold lines
    ax5.axhline(y=0, color='lightgray', linestyle='-', linewidth=1, alpha=0.6, zorder=1)
    ax5.axhline(y=df_thr, color='#bc4749', linestyle=':', linewidth=2, alpha=0.8, zorder=2)
    ax5.axhline(y=-df_thr, color='#bc4749', linestyle=':', linewidth=2, alpha=0.8, zorder=2)
    ax5.axhline(y=eq_thr, color='#457B9D', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax5.axhline(y=-eq_thr, color='#457B9D', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    
    ax5.set_xlabel(f'Average Expression ({cond_1} & {cond_2})', **axis_label_styles)
    ax5.set_ylabel(f'log2 Fold Change ({cond_1} vs {cond_2})', **axis_label_styles)
    ax5.set_title("MA Plot: Mean Expression vs Fold Change", **subplot_title_styles)
    ax5.grid(True, **grid_styles)
    ax5.tick_params(**tick_label_styles)

    # =============================================================================
    # BOTTOM ROW: Horizontal Status Counts, Exclusion Matrix, and Sample Size
    # =============================================================================
    
    # Horizontal Status counts 
    ax6 = fig.add_subplot(gs[2, 1])
    add_subplot_letter(ax6, 'F')
    
    status_order = ['Downregulated', 'Unexplained', 'Equivalent', 'Upregulated']
    if show_excluded:
        status_order.append('Excluded')
    
    status_counts = plot_data['Status'].value_counts().reindex(status_order, fill_value=0)
    colors_status = [status_colors[status] for status in status_order]
    
    # Create horizontal bars with better spacing
    y_positions = np.arange(len(status_order))
    bars = ax6.barh(
        y_positions, status_counts.values, 
        color=colors_status, alpha=0.9,
        edgecolor='black', linewidth=0.8,
        height=0.6  # Even thinner bars for more spacing
    )
    
    # Improve y-axis labels and spacing
    ax6.set_yticks(y_positions)
    ax6.set_yticklabels(status_order)
    ax6.tick_params(axis='y', labelsize=tick_label_styles.get('labelsize', 10), pad=tick_label_styles.get('pad', 3))
    # ax6.set_ylabel('Testing Result Categories', **axis_label_styles)
    ax6.set_xlabel('Count', **axis_label_styles)
    
    # Add value labels with better positioning
    max_count = max(status_counts.values) if len(status_counts.values) > 0 else 1
    for i, count in enumerate(status_counts.values):
        if count > 0:
            ax6.text(
                count + max_count * 0.02, i, f'{count:,}', 
                va='center', ha='left',
            )
    
    ax6.set_xlim(right=max_count * 1.25)
    ax6.grid(axis='x', **grid_styles)
    ax6.set_title('Category Counts', **subplot_title_styles)
    ax6.tick_params(**tick_label_styles)
    ax6.invert_yaxis()  # Put first category at top
    
    # Exclusion Matrix
    ax_matrix = fig.add_subplot(gs[2, 2])
    add_subplot_letter(ax_matrix, 'G')
    
    # Create exclusion matrix data
    if 'S1_Status' in plot_data.columns and 'S2_Status' in plot_data.columns:
        # Convert status to categorical codes
        def status_to_code(status_series):
            return status_series.map({
                -1: 'Filtered',    # CV > cv_thr
                0: 'Missing',      # NaN
                1: 'Retained'      # CV <= cv_thr
            }).fillna('Missing')
        
        s1_cat = status_to_code(plot_data['S1_Status'])
        s2_cat = status_to_code(plot_data['S2_Status'])
        
        # Create crosstab
        matrix_data = pd.crosstab(s1_cat, s2_cat, margins=False)
        
        # Ensure all categories are present
        categories = ['Retained', 'Missing', 'Filtered']
        matrix_data = matrix_data.reindex(index=categories, columns=categories, fill_value=0)
        
        im = ax_matrix.imshow(matrix_data.values, cmap='Grays', aspect='auto', alpha=0.8)
        
        # Add text annotations with dynamic color based on background
        max_val = matrix_data.values.max()
        for i in range(len(categories)):
            for j in range(len(categories)):
                value = matrix_data.iloc[i, j]
                # Use white text for darker backgrounds, black for lighter
                text_color = "white" if value > max_val * 0.4 else "black"
                text = ax_matrix.text(j, i, f'{value:,}',
                                    ha="center", va="center", 
                                    color=text_color,
                                    fontweight='bold', fontsize=tick_label_styles['labelsize'])
        
        # Set labels and title with improved spacing
        ax_matrix.set_xticks(range(len(categories)))
        ax_matrix.set_yticks(range(len(categories)))
        ax_matrix.set_xticklabels(categories)
        ax_matrix.set_yticklabels(categories)
        ax_matrix.tick_params(axis='x', labelsize=tick_label_styles.get('labelsize', 10), pad=tick_label_styles.get('pad', 3))
        ax_matrix.tick_params(axis='y', labelsize=tick_label_styles.get('labelsize', 10), pad=tick_label_styles.get('pad', 3))
        ax_matrix.set_xlabel(f'{cond_2} Status', **axis_label_styles)
        ax_matrix.set_ylabel(f'{cond_1} Status', **axis_label_styles)
        ax_matrix.set_title('Exclusion Matrix', **subplot_title_styles)
        
        # Improve tick parameters and remove spines for cleaner look
        ax_matrix.tick_params(**tick_label_styles)
        for spine in ax_matrix.spines.values():
            spine.set_visible(False)
        
    else:
        ax_matrix.text(0.5, 0.5, 'S1_Status and S2_Status\ncolumns not found', 
                      ha='center', va='center', transform=ax_matrix.transAxes,
                      fontsize=axis_label_styles['fontsize'], 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        ax_matrix.set_title('Exclusion Matrix', **subplot_title_styles)
        ax_matrix.axis('off')
    
    # Sample size comparison
    ax7 = fig.add_subplot(gs[2, 3])
    add_subplot_letter(ax7, 'H')
    
    n1_data = plot_data['N1'].dropna()
    n2_data = plot_data['N2'].dropna()

    if len(n1_data) > 0 and len(n2_data) > 0:
        max_bins = min(20, len(np.unique(n1_data)), len(np.unique(n2_data)))
        if max_bins >= 2:
            hb = ax7.hexbin(
                n1_data, n2_data,
                gridsize=max_bins, cmap='Grays', mincnt=1, 
                linewidths=0.3, alpha=0.85,
                rasterized=True  # Always rasterize hexbin for performance
            )
            # Add a minimal colorbar
            cb = fig.colorbar(hb, ax=ax7, shrink=0.7, pad=0.04)
            cb.set_label('Count', fontsize=tick_label_styles['labelsize'])
            cb.ax.tick_params(labelsize=tick_label_styles['labelsize'])
            nmax1, nmax2 = int(n1_data.max()), int(n2_data.max())
            ax7.set_xlim(0, nmax1 + 1)
            ax7.set_ylim(0, nmax2 + 1)
        else:
            # All proteins have the same sample size; show this info
            unique_n1 = n1_data.unique()
            unique_n2 = n2_data.unique()
            if len(unique_n1) == 1 and len(unique_n2) == 1:
                ax7.text(
                    0.5, 0.5,
                    f"All proteins have the same sample size:\n"
                    f"N1 = {unique_n1[0]}, N2 = {unique_n2[0]}\n"
                    f"(n = {len(n1_data)})",
                    ha='center', va='center', transform=ax7.transAxes,
                    fontsize=axis_label_styles['fontsize'],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray')
                )
            else:
                ax7.text(
                    0.5, 0.5,
                    'Not enough unique sample sizes\nto plot hexbin',
                    ha='center', va='center', transform=ax7.transAxes,
                    fontsize=axis_label_styles['fontsize'],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray')
                )
    
    ax7.set_xlabel(f'N1 ({cond_1} Samples)', **axis_label_styles)
    ax7.set_ylabel(f'N2 ({cond_2} Samples)', **axis_label_styles)
    ax7.set_title('Sample Size\nComparison', **subplot_title_styles)
    ax7.grid(True, **grid_styles)
    ax7.tick_params(**tick_label_styles)

    # =============================================================================
    # COMPREHENSIVE LEGEND AND ANNOTATIONS
    # =============================================================================
    
    ax8 = fig.add_subplot(gs[2, 4:5])
    
    # Create legend sections
    legend_elements = []
    
    # Section 1: Status categories
    legend_elements.append(plt.Line2D([0], [0], color='none', label='Status Categories:'))
    for status in status_order:
        if status in plot_data['Status'].unique():
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=status_colors[status], 
                          markersize=7, label=f'  {status}',
                          markeredgecolor='white', markeredgewidth=0.5)
            )
    
    # Section 2: Threshold lines
    legend_elements.append(plt.Line2D([0], [0], color='none', label=''))  # Spacer
    legend_elements.append(plt.Line2D([0], [0], color='none', label='Equivalence Thresholds:'))
    legend_elements.extend([
        plt.Line2D([0], [0], color='#457B9D', linestyle='--', linewidth=2.2, 
                  label=f'  p-value = {p_thr}'),
        plt.Line2D([0], [0], color='#457B9D', linestyle='--', linewidth=2.2, 
                  label=f'  |log2FC| = ±{eq_thr}')
    ])
    
    legend_elements.append(plt.Line2D([0], [0], color='none', label=''))  # Spacer
    legend_elements.append(plt.Line2D([0], [0], color='none', label='Difference Thresholds:'))
    legend_elements.extend([
        plt.Line2D([0], [0], color='#bc4749', linestyle=':', linewidth=2.2, 
                  label=f'  p-value = {p_thr}'),
        plt.Line2D([0], [0], color='#bc4749', linestyle=':', linewidth=2.2, 
                  label=f'  |log2FC| = ±{df_thr}')
    ])
    
    # Create the legend with optimized layout
    legend = ax8.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=legend_fontsize - 1,  # Slightly smaller for better fit
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        title=r'$\mathbf{Legend}$',
        title_fontsize=legend_fontsize,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=0.4,
        labelspacing=0.3,
        borderpad=0.6
    )
    legend.get_title().set_position((0, 0))

    ax8.axis('off')
    
    # Add comprehensive methodology and figure description in the last row (split into two columns)
    methodology_ax_left = fig.add_subplot(gs[3, :3])   # Left half of last row
    methodology_ax_right = fig.add_subplot(gs[3, 3:])  # Right half of last row
    methodology_ax_left.axis('off')
    methodology_ax_right.axis('off')

    # Left: Statistical methodology
    # Compute maximum sample sizes for each condition
    N1 = int(plot_data['N1'].max()) if 'N1' in plot_data.columns else 0
    N2 = int(plot_data['N2'].max()) if 'N2' in plot_data.columns else 0

    methodology_text_left = (
        r"$\mathbf{STATISTICAL\ TESTING\ METHODOLOGY}$" + "\n\n"
        rf"• $\mathbf{{Equivalence\ Testing}}$: Two One-Sided Tests (TOST) with $|log_2FC| < {eq_thr:.3f}$ as equivalence threshold" +"\n"
        rf"• $\mathbf{{Difference\ Testing}}$: Welch's t-test with $|log_2FC| > {df_thr:.3f}$ as significance threshold" + "\n"
        rf"• $\mathbf{{CV\ Filtering}}$: Features with CV $> {cv_thr:.2f}$ excluded from analysis" + "\n"
        rf"• $\mathbf{{Multiple\ Testing}}$: {correction} correction applied" + "\n"
        rf"• $\mathbf{{Significance\ Level}}$: $\alpha = {p_thr:.3f}$" + "\n"
        "\n"
        rf"$\mathbf{{Data\ Summary}}$: {len(plot_data):,} features analyzed" + "\n"
        rf"{cond_1}: {N1} samples • {cond_2}: {N2} samples"
    )

    # Right: Figure panel description
    methodology_text_right = (
        r"$\mathbf{FIGURE\ PANEL\ DESCRIPTION}$" + "\n\n"
        rf"$\mathbf{{A)}}$ T-test P-values: Distribution of p-values from Welch's t-tests comparing {cond_1} vs {cond_2}" + "\n"
        r"$\mathbf{B)}$ TOST P-values: Distribution of p-values from Two One-Sided Tests for equivalence assessment" + "\n"
        r"$\mathbf{C)}$ P-value Comparison: Scatter plot comparing t-test vs TOST p-values with significance thresholds" + "\n"
        r"$\mathbf{D)}$ Antler's Plot: Effect size (LogFC) vs statistical significance with TOST categories color-coded" + "\n"
        r"$\mathbf{E)}$ MA Plot: Mean expression vs log fold change with significance and equivalence regions highlighted" + "\n"
        r"$\mathbf{F)}$ Category Distribution: Bar chart showing counts of features in each TOST category" + "\n"
        r"$\mathbf{G)}$ Exclusion Matrix: Cross-tabulation of sample filtering status between conditions" + "\n"
        r"$\mathbf{H)}$ Sample Size Comparison: Hexagonal binning of sample sizes between conditions"
    )

    methodology_ax_left.text(
        0.02, 0.98, methodology_text_left,
        transform=methodology_ax_left.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor='#f8f9fa',
            alpha=0.9,
            edgecolor='#343a40',
            linewidth=1
        )
    )

    methodology_ax_right.text(
        -0.1, 0.98, methodology_text_right,
        transform=methodology_ax_right.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor='#f8f9fa',
            alpha=0.9,
            edgecolor='#343a40',
            linewidth=1
        ),
    )
    
    # Add main title with better positioning
    fig.suptitle(
        f"QuEStVar's Testing Summary for Equivalence and Difference Testing\n{title_add}",
        **main_title_styles
    )

    # Finalize the plot
    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

def antlers_with_annotations(
        quest_data,
        p_thr=0.05,
        eq_thr=0.5,
        df_thr=0.75,
        protein_ids: List[str] = None,    
        status_colors: Dict[str, str] = None,

        # Plot parameters
        figsize: Tuple[float, float] = (12, 9),
        id_col: str = 'Protein',
        annot_col: str = 'Gene',
        title_add: str = '',
        cond_1: str = 'Condition 1',
        cond_2: str = 'Condition 2',
        
        # Annotation parameters
        annotation_fontsize: int = 9,
        annotation_fontweight: str = 'bold',
        annotation_alpha: float = 0.5,
        annotation_edge_width: float = 1.5,
        arrow_width: float = 1.5,
        arrow_alpha: float = 0.8,
        
        # adjust_text parameters
        expand_points: Tuple[float, float] = (1.5, 1.5),
        expand_text: Tuple[float, float] = (1.3, 1.3),
        expand_objects: Tuple[float, float] = (1.1, 1.1),
        arrowprops_adjust: dict = None,
        force_points: float = 0.3,
        force_text: float = 0.8,
        force_objects: float = 0.5,
        
        rasterize_scatters=True,

        # Finalize plot parameters
        show: bool = True,
        save: bool = False,
        filename: str = "antlers_annotated",
        fileformats: List[str] = ['png', 'svg', 'pdf'],
        filepath: str = '',
        transparent: bool = True,
        dpi: int = 300,
    ):
    """
        Create an enhanced Antler's plot with high-quality annotations and 
        automatic label positioning.
    
        TODO: docstring
    """
    
    # Default colors optimized for contrast
    if status_colors is None:
        status_colors = {
            'Unexplained': "#C2C0C0",    
            'Excluded': '#565d61',       
            'Upregulated': '#780000',    
            'Downregulated': '#e36414',  
            'Equivalent': '#003049',     
        }
    
    # Default arrow properties for adjust_text
    if arrowprops_adjust is None:
        arrowprops_adjust = dict(
            arrowstyle='->', 
            color='black', 
            lw=arrow_width, 
            alpha=arrow_alpha,
            shrinkA=3, 
            shrinkB=3
        )
    
    # Helper function to determine optimal text color based on background
    def get_optimal_text_color(bg_color):
        """
        Determine optimal text color (black or white) based on background color luminance.
        Uses relative luminance calculation for accessibility.
        """
        # Convert color to RGB if it's a hex string
        if isinstance(bg_color, str) and bg_color.startswith('#'):
            # Remove # and convert to RGB
            bg_color = bg_color.lstrip('#')
            r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            # Assume it's already RGB or a named color - convert via matplotlib
            rgb = mcl.to_rgb(bg_color)
            r, g, b = [int(255 * x) for x in rgb]
        
        # Calculate relative luminance (ITU-R BT.709)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        # Return white text for dark backgrounds, black for light backgrounds
        return 'white' if luminance < 0.5 else 'black'
    
    # Enhanced color palette for annotation boxes (better contrast)
    def get_annotation_colors(status):
        """Get optimized background and text colors for annotations based on status."""
        color_schemes = {
            'Unexplained': {'bg': '#F5F5F5', 'edge': '#C2C0C0'},      # Light gray bg
            'Upregulated': {'bg': '#FFEBEE', 'edge': '#780000'},      # Light red bg
            'Downregulated': {'bg': '#FFF3E0', 'edge': '#e36414'},    # Light orange bg
            'Equivalent': {'bg': '#E3F2FD', 'edge': '#003049'},       # Light blue bg
        }
        
        scheme = color_schemes.get(status, {'bg': '#FFFFFF', 'edge': '#000000'})
        text_color = get_optimal_text_color(scheme['bg'])
        
        return scheme['bg'], scheme['edge'], text_color

    # Plot order for proper layering
    plot_order = ['Unexplained', 'Downregulated', 'Upregulated', 'Equivalent']

    # Data validation
    plot_data = quest_data.copy()
    if 'Status' not in plot_data.columns:
        raise ValueError("The input data must contain a 'Status' column.")

    if 'log2FC' not in plot_data.columns or 'log10(adj_pval)' not in plot_data.columns:
        raise ValueError("The input data must contain 'log2FC' and 'log10(adj_pval)' columns.")

    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points by status with controlled layering
    for status in plot_order:
        if status in plot_data['Status'].unique():
            subset = plot_data[plot_data['Status'] == status]
            if len(subset) > 0:
                ax.scatter(
                    subset['log2FC'], subset['log10(adj_pval)'],
                    c=status_colors[status], label=status, s=45,
                    edgecolor='white', linewidth=0.4, alpha=0.85,
                    rasterized=rasterize_scatters, zorder=5
                )
    
    # Optimize plot limits for better visualization
    x_data = plot_data['log2FC'].dropna()
    y_data = plot_data['log10(adj_pval)'].dropna()

    if len(x_data) > 0 and len(y_data) > 0:
        # X-axis: symmetric around zero with padding
        x_abs_max = np.max(np.abs(x_data))
        x_padding = x_abs_max * 0.15  # More padding for annotations
        ax.set_xlim(-(x_abs_max + x_padding), x_abs_max + x_padding)

        # Y-axis: with padding for annotations
        y_min, y_max = y_data.min(), y_data.max()
        y_padding = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add enhanced threshold lines
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
    
    # Equivalence thresholds (blue dashed) with labels
    eq_color = '#1976D2'
    ax.axhline(y=np.log10(p_thr), color=eq_color, linestyle='--', linewidth=2.5, 
               alpha=0.9, zorder=2, label=f'Equivalence: p={p_thr}')
    ax.axvline(x=eq_thr, color=eq_color, linestyle='--', linewidth=2.5, 
               alpha=0.9, zorder=2)
    ax.axvline(x=-eq_thr, color=eq_color, linestyle='--', linewidth=2.5, 
               alpha=0.9, zorder=2)
    
    # Difference thresholds (red dotted) with labels
    diff_color = '#D32F2F'
    ax.axhline(y=-np.log10(p_thr), color=diff_color, linestyle=':', linewidth=2.5, 
               alpha=0.9, zorder=2, label=f'Difference: p={p_thr}')
    ax.axvline(x=df_thr, color=diff_color, linestyle=':', linewidth=2.5, 
               alpha=0.9, zorder=2)
    ax.axvline(x=-df_thr, color=diff_color, linestyle=':', linewidth=2.5, 
               alpha=0.9, zorder=2)
    
    # Enhanced labels and styling
    ax.set_xlabel(f'log₂ Fold Change ({cond_1} vs {cond_2})', fontsize=14)
    ax.set_ylabel('log₁₀ Adjusted p-value\n(Equivalence |0| -Difference)', fontsize=14)
    
    title = "Antler's Plot: Equivalence + Difference Testing"
    if title_add:
        title += f"\n{title_add}"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='lightgray')
    ax.tick_params(labelsize=12, length=6, width=1.2)
    
    # Improved legend positioned outside plot area
    legend = ax.legend(
        title='Legend', 
        fontsize=11, 
        title_fontsize=12,
        loc='upper left',
        # Position outside plot area
        bbox_to_anchor=(1.025, 1.025),
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='black',
        facecolor='white'
    )
    legend.get_title().set_fontweight('bold')

    # Enhanced annotations with adjust_text (robust for 0, 1, or many annotations)
    if protein_ids is not None and len(protein_ids) > 0:
        annotations = []
        points = []
        for protein_id in protein_ids:
            matching_rows = plot_data[plot_data[id_col] == protein_id]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                x, y = row['log2FC'], row['log10(adj_pval)']
                # Prepare annotation text: prefer annot_col, fallback to id_col
                if annot_col in row and pd.notnull(row[annot_col]) and str(row[annot_col]).strip():
                    annot_text = str(row[annot_col])
                else:
                    annot_text = str(row[id_col])
                if len(annot_text) > 15:
                    annot_text = annot_text[:12] + "..."
                status = row['Status'] if 'Status' in row else 'Unexplained'
                bg_color, edge_color, text_color = get_annotation_colors(status)
                
                # Create annotation (initial offset)
                annotation = ax.annotate(
                    annot_text,
                    xy=(x, y),
                    xytext=(x + 0.2, y + 0.2),
                    fontsize=annotation_fontsize,
                    fontweight=annotation_fontweight,
                    color=text_color,
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor=bg_color,
                        edgecolor=edge_color,
                        linewidth=annotation_edge_width,
                        alpha=annotation_alpha
                    ),
                    arrowprops=arrowprops_adjust,
                    zorder=10
                )
                annotations.append(annotation)
        # # Use adjust_text if there are any annotations
        # if annotations:
        #     adjust_text(
        #         annotations,
        #         expand_points=expand_points,
        #         expand_text=expand_text,
        #         expand_objects=expand_objects,
        #         arrowprops=arrowprops_adjust,
        #         force_points=force_points,
        #         force_text=force_text,
        #         force_objects=force_objects,
        #         ax=ax,
        #         precision=0.1,
        #         save_steps=False,
        #         avoid_self=True,
        #         only_move={'points': 'xy', 'text': 'xy'}
        #     )

    # Add threshold value annotations
    if len(x_data) > 0 and len(y_data) > 0:
        # Equivalence threshold labels
        ax.text(eq_thr + 0.05, ax.get_ylim()[1] * 0.95, f'eq: ±{eq_thr}', 
                rotation=90, ha='left', va='top', fontsize=10, 
                color=eq_color, fontweight='bold', alpha=0.8)
        
        # Difference threshold labels  
        ax.text(df_thr + 0.05, ax.get_ylim()[1] * 0.95, f'diff: ±{df_thr}', 
                rotation=90, ha='left', va='top', fontsize=10, 
                color=diff_color, fontweight='bold', alpha=0.8)

    # Final layout adjustment with space for external legend
    plt.tight_layout()
    # Adjust the subplot to make room for the legend
    plt.subplots_adjust(right=0.82)
    
    # Finalize the plot
    finalize_plot(
        fig, show, save, filename,
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

# ======================================================================================
# ProteoForge - Modelling
# ======================================================================================

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
        show: bool = True,
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

    finalize_plot( 
        fig, show, save, filename,
        # Arguments
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

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
        
        condition_palette: dict = None,
        cluster_palette: dict = None,
        perturbed_peptide_col: Optional[str] = None,
        # Plotting parameters
        figsize: tuple = (12, 9),
        # Saving options
        save: bool = False,
        show: bool = True,
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
        "errorbar": ('ci', 95),
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
    # Early return if no data for this protein
    if plot_data.shape[0] == 0:
        print(f"No data found for protein: {cur_protein}")
        return None
    plot_data['isSignificant'] = plot_data[pvalue_col] <= pThr
    # Guard against zero p-values
    safe_p = plot_data[pvalue_col].replace(0, np.finfo(float).tiny)
    plot_data["-log10(adj.pval)"] = -np.log10(safe_p)
    
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

    # Determine perturbed peptides if column provided
    perturbed_peptides = set()
    if perturbed_peptide_col is not None and perturbed_peptide_col in plot_data.columns:
        # Expect a boolean or indicator per row; collect peptide ids that are perturbed
        mask = plot_data[perturbed_peptide_col].notna() & (plot_data[perturbed_peptide_col].astype(bool))
        perturbed_peptides = set(plot_data.loc[mask, peptide_col].unique())

    ## Initialize the figure
    fig, ax = plt.subplots(
        figsize=figsize, ncols=1, nrows=3, 
        sharex=True, sharey=False,
        gridspec_kw={ "height_ratios": [1, .5, 1], "hspace": 0.01 }
    )
    # Create an explicit ordered categorical for peptide positions so we can draw patches reliably
    ordered_peptides = plot_data[peptide_col].drop_duplicates().tolist()
    plot_data[peptide_col] = pd.Categorical(plot_data[peptide_col], categories=ordered_peptides, ordered=True)
    plot_data = plot_data.sort_values(by=[peptide_col])
    plot_data['_xpos'] = plot_data[peptide_col].cat.codes
    # First Plot shows the adjusted intensity values
    sns.lineplot(
        ax=ax[0],
        data=plot_data,
        x='_xpos',
        y=adjIntensity_col,
        hue=condition_col,
        palette=condition_palette,
        **lineplot_params
    )
    ax[0].set_ylabel("Adjusted Intensity", fontsize=12)
    # Dynamically adjust bbox_to_anchor for legend so that the leftmost side aligns at x=1.1
    legend = ax[0].legend(**legend_params, title="Condition", loc="upper right", bbox_to_anchor=(1.1, 1))
    fig.canvas.draw()  # Needed to get correct legend size
    legend_width = legend.get_window_extent().width / fig.dpi / fig.get_size_inches()[0]  # width in axes fraction
    # Calculate new x anchor so left side is at 1.1
    ncol = legend_params.get("ncol", 1)
    if ncol > 1:
        # For multi-column, width is less predictable, so skip adjustment
        pass
    else:
        # Only adjust for single-column legends
        new_x = 1.1 + legend_width
        legend.set_bbox_to_anchor((1.1 + legend_width, 1))

    # Create a cluster
    pepClusters = plot_data[[
    peptide_col, '-log10(adj.pval)', cluster_col, 'isSignificant', '_xpos'
    ]].drop_duplicates()
    # Second Plot shows the adjusted p-values
    sns.scatterplot(
        ax=ax[1],
        data=pepClusters,
        x='_xpos',
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
    threshold_line = -np.log10(pThr)
    # If the preferred p-value threshold is higher than maxLine, extend maxLine
    if threshold_line > maxLine:
        maxLine = threshold_line + offset
    for i, row in pepClusters.iterrows():
        xpos = int(row['_xpos'])
        cluster_val = row[cluster_col]
        color = cluster_palette.get(cluster_val, '#d3d3d3')
        ax[1].add_patch(
            plt.Rectangle(
                (xpos - 0.5, minLine), 1, maxLine - minLine,
                color=color,
                alpha=0.5, linewidth=1.5, zorder=1, 
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

    # Highlight perturbed peptides on middle plot (vertical line + star marker) if any
    if perturbed_peptides:
        for pep in ordered_peptides:
            if pep in perturbed_peptides:
                xpos = int(plot_data[plot_data[peptide_col] == pep]['_xpos'].iloc[0])
                # vertical line across middle plot
                ax[1].axvline(x=xpos, color='gold', linestyle='-', linewidth=1.5, alpha=0.9, zorder=5)
                # star marker at top of middle plot for visibility
                ytop = maxLine
                ax[1].scatter([xpos], [ytop], marker='*', color='gold', s=200, edgecolor='black', zorder=6)

    # Third Plot shows the log10 intensity values
    sns.lineplot(
        ax=ax[2],
        data=plot_data,
    x='_xpos',
        y=rawIntensity_col,
        hue=condition_col,
        palette=condition_palette,
        legend=False,
        **lineplot_params
    )

    ax[2].set_ylabel("Log10 Raw Intensity", fontsize=12)
    ax[2].set_xlabel(f"PeptideIDs in {cur_protein} (Ordered by Start Position)", fontsize=12)
    # Set xticks to peptide labels (add '*' suffix for perturbed peptides)
    xticks = plot_data['_xpos'].unique().tolist()
    xticklabels = [str(p) + ('*' if (p in perturbed_peptides) else '') for p in ordered_peptides]
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticklabels)

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

    finalize_plot( 
        fig, show, save, filename,
        # Arguments
        filepath=filepath,
        formats=fileformats,
        transparent=transparent,
        dpi=dpi
    )

    # return fig

# ======================================================================================
# ProteoForge - Clustering 
# ======================================================================================

def clustering_check_with_single_heatmap(
    corr_matrix, 
    labels, 
    protein_id,
    ax=None,
    base_size=1,
    min_size=6,
    max_size=20,
    cmap="RdBu_r",
    show_annotations=False,
    linewidth=3,
    boundary_color="black",
    custom_title=None,
    show_cluster_bars=True,
    cluster_bar_width=0.3, 
    vmin=-1,
    vmax=1,
):
    if not labels.size:
        print("No clustering results to visualize")
        return None
    
    if corr_matrix is None or corr_matrix.empty:
        print("No valid correlation matrix for visualization")
        return None
    
    if ax is None:
        n_peptides = len(labels)
        fig_size = max(min_size, min(max_size, n_peptides * base_size))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    sorted_idx = np.argsort(labels)
    sorted_labels = labels[sorted_idx]
    sorted_corr = corr_matrix.iloc[sorted_idx, sorted_idx]
    
    # Create main heatmap
    sns.heatmap(
        sorted_corr.fillna(0),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot=show_annotations,
        fmt=".2f" if show_annotations else None,
        ax=ax,
    )
    
    # Add cluster indicator bars if requested
    if show_cluster_bars:
        n_clusters = len(np.unique(labels))
        n_peptides = len(labels)
        
        # Generate distinct colors for clusters
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        label_to_color = {label: cluster_colors[i] for i, label in enumerate(np.unique(sorted_labels))}
        
        # Create color arrays for the cluster bars
        cluster_color_array = np.array([label_to_color[label] for label in sorted_labels])
        
        # Add horizontal cluster bar (top)
        for i, color in enumerate(cluster_color_array):
            ax.add_patch(plt.Rectangle((i, n_peptides + 0.05), 1, cluster_bar_width, 
                                     facecolor=color, edgecolor='white', linewidth=0.5))
        
        # Add vertical cluster bar (right) - fix alignment with y-axis
        for i, color in enumerate(cluster_color_array):
            ax.add_patch(plt.Rectangle((n_peptides + 0.05, i), cluster_bar_width, 1, 
                                     facecolor=color, edgecolor='white', linewidth=0.5))
        
        # Add cluster labels in the bars
        group_changes = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [n_peptides]])
        for i in range(len(group_changes) - 1):
            start_idx = group_changes[i]
            end_idx = group_changes[i + 1]
            mid_idx = (start_idx + end_idx) / 2
            cluster_label = sorted_labels[start_idx]
            
            # Label on top bar
            ax.text(mid_idx, n_peptides + 0.05 + cluster_bar_width/2, f'C{cluster_label}', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='black')
            
            # Label on right bar - fix alignment with y-axis
            ax.text(n_peptides + 0.05 + cluster_bar_width/2, mid_idx, f'C{cluster_label}', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='black', rotation=90)
    
    # Draw boundary lines between clusters
    group_changes = np.where(np.diff(sorted_labels) != 0)[0] + 1
    for change in group_changes:
        ax.axhline(y=change, color=boundary_color, linewidth=linewidth)
        ax.axvline(x=change, color=boundary_color, linewidth=linewidth)
    
    # Set title
    n_clusters = len(np.unique(labels))
    n_peptides = len(labels)
    if custom_title is not None:
        ax.set_title(custom_title, fontweight="bold")
    else:
        ax.set_title(
            f"Proteoform Clustering - {protein_id}\n"
            f"{n_peptides} peptides, {n_clusters} clusters", 
            fontweight="bold"
        )
    
    # Adjust axis limits to accommodate cluster bars
    if show_cluster_bars:
        ax.set_xlim(0, n_peptides + cluster_bar_width + 0.1)
        ax.set_ylim(0, n_peptides + cluster_bar_width + 0.1)
    
    return ax

# ======================================================================================
# ProteoForge - Summarization 
# ======================================================================================



# ======================================================================================
# ProteoForge - Protein Annotations 
# ======================================================================================

#TODO: Following are pretty early crudely built plotting functions for protein annotations.

def annotations_on_protein(
        target_protein, 
        control_package, 
        target_peptide=None
    ):
    """
    Generates a detailed protein coverage plot with UniProt annotations and peptide traces.

    Parameters:
    -----------
    target_protein : str
        The ID of the protein to visualize (e.g., 'P12345').
    control_package : dict
        A master dictionary containing all data, styles, and configuration objects.
        Expected structure:
            {
                'data': {
                    'uniprot': pd.DataFrame,
                    'detailed': pd.DataFrame,
                    'summary': pd.DataFrame
                },
                'styles': {
                    'feature_colors': dict,
                    'dpf_colors': dict,
                    'category_order': list
                }
                'config': {
                    'is_demo': bool,
                    'figure_path': str,
                    'figure_formats': list,
                    'transparent_bg': bool,
                    'figure_dpi': int,
                    'str_add': str
                }
            }
    target_peptide : str, optional
        ID of a specific peptide to highlight (if needed for future logic).
    """
    
    ## Internal Variables (can be moved to control package later)
    # Placing the UniProt features into categories
    feature_categories = {
        'Protein Processing & Maturation': [
            'INIT_MET', 'SIGNAL', 'TRANSIT', 
            'PROPEP', 'CHAIN', 'PEPTIDE', 'CLEAVAGE'
        ],
        'Co- & Post-Translational Modifications': [
            'MOD_RES', 'CARBOHYD', 'LIPID', 
            'DISULFID', 'CROSSLNK', 'NON_STD'
        ],
        'Sequence Heterogeneity & Isoforms': [
            'VARIANT', 'VAR_SEQ', 'MUTAGEN', 
            'CONFLICT', 'BREAKPOINT'
        ],
        'Functional Domains, Regions & Sites': [
            'DOMAIN', 'REPEAT', 'ZN_FING', 'MOTIF', 
            'REGION', 'ACT_SITE', 'BINDING', 'SITE', 
            'DNA_BIND', 'CA_BIND', 'METAL', 'NP_BIND'
        ],
        'Structure & Topology': [
            'HELIX', 'STRAND', 'TURN', 'COILED', 
            'TRANSMEM', 'INTRAMEM', 'TOPO_DOM', 'COMPBIAS'
        ]
    }

    # Feature category colors for UniProt annotations
    feature_category_colors = {
        'Sequence Heterogeneity & Isoforms': '#e74c3c',
        'Co- & Post-Translational Modifications': '#3498db', 
        'Protein Processing & Maturation': '#2ecc71',
        'Structure, Topology & Sequence Characteristics': '#f39c12',
        'Functional Domains, Regions & Sites': '#9b59b6'
    }

    # Define FIXED ordering for categories and features for consistency across all proteins
    CATEGORY_ORDER = [
        'Sequence Heterogeneity & Isoforms',
        'Co- & Post-Translational Modifications', 
        'Protein Processing & Maturation',
        'Structure, Topology & Sequence Characteristics',
        'Functional Domains, Regions & Sites'
    ]

    # 1. UNPACK CONTROL PACKAGE
    # Use .get() or direct access depending on how strict you want to be
    dfs = control_package['data']
    styles = control_package['styles']
    cfg = control_package['config']

    uniprot_data = dfs['uniprot']
    test_data = dfs['detailed']
    summary_data = dfs['summary']

    # 2. FILTER DATA FOR CURRENT PROTEIN
    # Get UniProt annotations
    current_uniprot = uniprot_data[uniprot_data['Protein'] == target_protein].copy()
    
    # Get detailed peptide data
    current_detailed = test_data[test_data['Protein'] == target_protein].copy()
    
    # Get summary data (contains dPF and trace info)
    current_summary = summary_data[summary_data['Protein'] == target_protein].copy()
    
    if current_summary.empty:
        print(f"Warning: No summary data found for protein {target_protein}")
        return

    # 3. SETUP STYLES & METADATA
    dPF_colors = styles.get('dpf_colors', {})

    # Get protein information
    cur_seqLength = current_summary['Length'].iloc[0]
    cur_protein = current_summary['Protein'].iloc[0]
    cur_description = current_summary['Description'].iloc[0]
    cur_geneName = current_summary['Gene'].iloc[0]
    coverage_pct = current_summary['Coverage'].iloc[0] if 'Coverage' in current_summary.columns else 0
    total_peptides = current_summary['PeptideID'].nunique()

    # 4. CALCULATE DIMENSIONS
    protein_length = cur_seqLength
    if protein_length <= 300:
        fig_width = 10
    elif protein_length <= 600:
        fig_width = 15
    elif protein_length <= 1000:
        fig_width = 20
    else:
        fig_width = 25

    # Calculate figure dimensions based on feature categories and max traces
    # Handle case where trace might be NaN or missing
    max_traces = current_summary['trace'].max() if 'trace' in current_summary.columns else 1
    if pd.isna(max_traces): max_traces = 1

    relevant_categories = []
    category_feature_counts = {}

    if len(current_uniprot) > 0:
        # Use FIXED category order - only include categories that have data
        available_categories = current_uniprot['feature_category'].unique().tolist()
        relevant_categories = [cat for cat in CATEGORY_ORDER if cat in available_categories]
        
        # Calculate number of features per category for dynamic height adjustment
        for category in relevant_categories:
            all_features_in_category = []
            for feat_cat, features in feature_categories.items():
                if feat_cat == category:
                    all_features_in_category = features
                    break
            category_feature_counts[category] = len(all_features_in_category)
    else:
        relevant_categories = []

    n_feature_categories = len(relevant_categories)

    # Calculate heights based on feature counts in each category
    category_heights = []
    for category in relevant_categories:
        feature_count = category_feature_counts.get(category, 1)
        # Base height + extra height based on number of features
        height = 1.0 + (feature_count * 0.15) 
        category_heights.append(height)

    # 5. INITIALIZE FIGURE
    base_height = 1 
    total_category_height = sum(category_heights)
    fig_height = total_category_height + base_height + 5 

    height_ratios = category_heights + [base_height]
    n_subplots = n_feature_categories + 1
    
    fig, axes = plt.subplots(
        nrows=n_subplots, ncols=1, figsize=(fig_width, fig_height), # using calculated fig_width
        gridspec_kw={'height_ratios': height_ratios}
    )

    if n_subplots == 1:
        axes = [axes]

    # Set consistent x-axis ticks
    step = max(1, protein_length // 20)
    positions = range(0, protein_length + 1, step)

    # 6. PLOT FEATURE CATEGORIES (Top Layers)
    for cat_idx, category in enumerate(relevant_categories):
        ax_cat = axes[cat_idx]
        ax_cat.set_xlim(0, protein_length)
        
        cat_annotations = current_uniprot[current_uniprot['feature_category'] == category].copy()
        
        # Get ALL possible features for this category
        all_features_in_category = []
        for feat_cat, features in feature_categories.items():
            if feat_cat == category:
                all_features_in_category = features
                break
        
        # Sort and map Y positions
        all_features_in_category_sorted = sorted(all_features_in_category)
        y_positions = {feat: i for i, feat in enumerate(all_features_in_category_sorted)}
        y_max = len(all_features_in_category_sorted)
        ax_cat.set_ylim(-0.5, y_max + 0.5)
        
        category_color = feature_category_colors.get(category, '#7f8c8d')
        
        if len(cat_annotations) > 0:
            cat_annotations = cat_annotations[cat_annotations['feature'].isin(all_features_in_category_sorted)]
            
            for _, annotation in cat_annotations.iterrows():
                start, end = annotation['start'], annotation['end']
                feature_type = annotation['feature']
                cleaned_note = annotation['note'] if pd.notna(annotation['note']) else ''
                
                if feature_type in y_positions:
                    y_pos = y_positions[feature_type]
                    
                    if start == end:
                        # Single position marker
                        ax_cat.axvline(
                            x=start, ymin=(y_pos-0.3)/(y_max+1), ymax=(y_pos+0.3)/(y_max+1), 
                            color=category_color, linewidth=4, alpha=0.8
                        )
                        ax_cat.scatter(
                            start, y_pos, s=80, c=category_color, alpha=0.8, 
                            edgecolors='white', linewidth=1, zorder=5
                        )
                    else:
                        # Range bar
                        ax_cat.barh(
                            y_pos, end - start, left=start, height=0.7, 
                            color=category_color, alpha=0.8, edgecolor='white', linewidth=1
                        )
                    
                    # Labels
                    feature_width = end - start if end != start else protein_length * 0.02
                    if feature_width > protein_length * 0.03:
                        label_text = f"{feature_type}"
                        if cleaned_note and len(cleaned_note) > 0:
                            label_text += f"\n{cleaned_note[:20]}{'...' if len(cleaned_note) > 20 else ''}"
                        
                        x_pos = start if start == end else start + (end - start)/2
                        ax_cat.text(
                            x_pos, y_pos, label_text, 
                            ha='center', va='center', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                        )
        
        # Formatting for category axis
        ax_cat.set_yticks(list(y_positions.values()))
        ax_cat.set_yticklabels(list(y_positions.keys()), fontsize=9, fontweight='bold')
        
        ax_cat.text(0.02, 0.95, category, transform=ax_cat.transAxes, 
                    fontsize=12, fontweight='bold', color=category_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=category_color),
                    verticalalignment='top', horizontalalignment='left')
        
        ax_cat.set_ylabel('')
        ax_cat.set_xlabel('')
        ax_cat.grid(True, alpha=0.4, axis='both', linestyle='-', linewidth=0.5)
        ax_cat.set_axisbelow(True)
        ax_cat.set_xticks(positions)
        ax_cat.set_xticklabels([])

    # 7. PLOT BOTTOM LAYER (Peptides & Protein)
    ax_combined = axes[-1]
    ax_combined.set_xlim(0, protein_length)

    y_min = -1.8 
    y_max_combined = max_traces + 0.3 
    ax_combined.set_ylim(y_min, y_max_combined)

    # Protein Rectangle
    protein_height = 1.0 
    ax_combined.barh(-0.9, protein_length, height=protein_height, color='#34495e', alpha=0.8, edgecolor='white', linewidth=2)

    protein_text = f'{cur_geneName} ({cur_protein}) - {cur_description[:40]}{"..." if len(cur_description) > 40 else ""}\n{protein_length} AA | Coverage: {coverage_pct:.1f}%'
    ax_combined.text(protein_length/2, -0.9, protein_text, 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#34495e', alpha=0.9, edgecolor='white'))

    # Peptide Traces
    for _, peptide in current_summary.iterrows():
        start, end = peptide['peptide_start'], peptide['peptide_end']
        trace = peptide['trace']
        dpf = peptide['dPF']
        is_sig = peptide['isSignificant']
        peptide_id = peptide['PeptideID']
        
        # Highlight logic (if current_peptide passed in args matches)
        is_target_peptide = (target_peptide is not None) and (peptide_id == target_peptide)
        
        color = dPF_colors.get(dpf, '#7f8c8d')
        
        height = 0.85 if is_sig else 0.7 
        alpha = 0.9 if is_sig else 0.7
        edge_color = 'red' if is_target_peptide else 'white' # Example highlight logic
        line_width = 2.5 if is_target_peptide else 1
        
        y_position = trace * 0.9
        
        ax_combined.barh(y_position, end - start, left=start, height=height, 
                        color=color, alpha=alpha, edgecolor=edge_color, linewidth=line_width)
        
        label = f"{peptide_id}{'*' if is_sig else ''}"
        font_weight = 'bold' if is_sig else 'normal'
        
        ax_combined.text(start + (end - start)/2, y_position, label, 
                        ha='center', va='center', fontsize=9, fontweight=font_weight, color='black')

    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=f'dPF={dpf}' if dpf >= 0 else 'PTM') 
            for dpf, color in dPF_colors.items() if dpf in current_summary['dPF'].values
    ]
    if legend_elements:
        ax_combined.legend(handles=legend_elements, loc='upper right', fontsize=9, title='dPF Categories')

    # Y-Axis Formatting
    y_tick_positions = [-0.9] + [i * 0.9 for i in range(int(max_traces) + 1)]
    ax_combined.set_yticks(y_tick_positions)
    y_labels = ['Protein'] + ['' for _ in range(int(max_traces) + 1)] 
    ax_combined.set_yticklabels(y_labels, fontsize=9, fontweight='bold')

    ax_combined.set_xlabel('Protein Position (AA)', fontsize=12, fontweight='bold')
    ax_combined.grid(True, alpha=0.4, axis='both', linestyle='-', linewidth=0.5)
    ax_combined.set_axisbelow(True)
    ax_combined.set_ylabel('Peptides & Protein', fontsize=12, fontweight='bold', rotation=90)

    # X-Axis Formatting
    ax_combined.set_xticks(positions)
    ax_combined.set_xticklabels(positions, fontsize=10)

    # 8. FINALIZE AND SAVE
    fig.suptitle(
        f'{cur_geneName} ({cur_protein}) - {cur_description}\n'
        f'Length: {cur_seqLength} AA | Coverage: {coverage_pct:.1f}% | Peptides: {total_peptides}',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.08, hspace=0.05)
    
    str_add = cfg.get("str_add", "")
    if str_add:
        str_add = f"_{str_add}"

    finalize_plot( 
        fig, 
        show=True, 
        save=not cfg.get('is_demo', False), 
        filename=f'protein_overview_{cur_geneName}{str_add}',
        filepath=cfg.get('figure_path', './figures'),
        formats=cfg.get('figure_formats', ['png']),
        transparent=cfg.get('transparent_bg', False),
        dpi=cfg.get('figure_dpi', 300)
    )