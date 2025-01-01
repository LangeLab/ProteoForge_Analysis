import numpy as np
import pandas as pd

from gprofiler import GProfiler

from src import utils

def printParams(
        m_pval: float,
        e_pval: float,
        correction: str,
        pval_cap: int,
        organism: str,
        sources: list[str],
        background: list[str], 
        enrichment_sets: dict[str, list[str]],
        analysis_name: str = None
    ):
    """
        Prints the parameters used to run the enrichment analysis.
    """
    print("Enrichment analysis parameters:")
    print(f"  - match p-value threshold: {m_pval}")
    print(f"  - enrichment p-value threshold: {e_pval}")
    print(f"  - correction method: {correction}")
    print(f"  - capping p-value at: {pval_cap}")
    print(f"  - using {organism} as organism")
    print(f"  - getting results from {sources} ")
    if background is None:
        print(f"  - using default background")
    else:
        print(f"  - using {len(background)} proteins as custom background")
    if len(enrichment_sets) > 1:
        print(f"  - running multi-query enrichment with {len(enrichment_sets)} queries")
    for key, value in enrichment_sets.items():
        print(f"      - using {len(value)} proteins for '{key}' enrichment analysis")
    if analysis_name is None:
        print("  - no analysis name provided")
    else:
        print(f"  - will use the '{analysis_name}' as analysis name for id and save files")

def run_gprofiler(
        query: dict[str, list[str]],
        background: list[str],
        organism: str = "hsapiens",
        user_threshold: float = 1,     # Used to get all enriched
        signf_threshold: float = 0.001, # Used to label significant
        correction: str = "g_SCS",
        sources: list[str] = ["GO:BP", "GO:MF", "GO:CC", "REAC"],
        no_evidences: bool = False,
        no_iea: bool = True,
        ordered: bool = False,
        simplify_cols: bool = True,
        pval_cap: float = 10**-20,
        save_path: str = None,
        analysis_name: str = None,
        verbose: bool = True
    ):
    """
        The flexible access to gprofiler's api with gprofiler package.
        Runs gprofiler on a dictionary of lists of proteins.
    """
    # Start the timer
    startTime = utils.getTime()
    # Couple of checks and preparations
    if save_path not in [None, ""]:
        if analysis_name is None:
            analysis_name = "gprofiler"
            if verbose:
                print("No analysis name provided, using 'gprofiler' as default.")
    else:
        raise ValueError("No save_path provided.")

    # Create g:Profiler object
    gp = GProfiler(
        return_dataframe=True
    )

    # Run the enrichment
    gp_res = gp.profile(
        organism=organism,
        query=query, 
        background=background, 
        user_threshold=user_threshold,
        significance_threshold_method=correction,
        sources=sources,
        no_evidences=no_evidences, 
        no_iea=no_iea, 
        ordered=ordered,
    )

    # If user defined pval_cap, cap the p_value for certain plots
    if pval_cap is not None:
        if verbose:
            print("Capping the p_value at", pval_cap)
        gp_res["p_capped"] = gp_res["p_value"].clip(lower=pval_cap)
    else:
        gp_res["p_capped"] = gp_res["p_value"]

    # Update the returned data with more info
    # GORILLA like Enrichment 
    gp_res["Enrichment"] = (
        (gp_res["intersection_size"] / gp_res["query_size"]) /
        (gp_res["term_size"] / gp_res["effective_domain_size"])
    )
    # Calculate GeneRatio
    gp_res["GeneRatio"] = (
        gp_res["intersection_size"] / 
        gp_res["term_size"]
    )
    # negative Log10 of p_value and p_capped
    gp_res["-log10(p_value)"] = -np.log10(gp_res["p_value"])
    gp_res["-log10(p_capped)"] = -np.log10(gp_res["p_capped"])
    
    # Significant
    gp_res["significant"] = (
        gp_res["p_value"] < signf_threshold
    )
    # Remove rows with no parents
    gp_res = gp_res[
        gp_res["parents"].map(len) > 0
    ].reset_index(drop=True)

    if simplify_cols:
        if verbose:
            print("Simplifying the columns.")
        # Simplify the columns
        gp_res = gp_res[
            [
                "query",
                "source",
                "native",
                "name",
                "significant",
                "p_value",
                "p_capped",
                "-log10(p_value)",
                "-log10(p_capped)",
                "Enrichment",
                "GeneRatio",
                "parents",
                # "intersections"
            ]
        ]


    # Save the results if save_path is provided
    if save_path not in [None, ""]:
        gp_res.to_csv(
            (
                save_path + 
                analysis_name + 
                ".csv"
            ),
            index=False
        )

    if verbose:
        # Write detailed report on the analysis
        print(
            "P-value threshold: {} \nCorrection method: {}".format(
                user_threshold,
                correction
            )
        )
        print(
            "Number of queries:", 
            len(query)
        )
        utils.print_series(
            gp_res["query"].value_counts(),
            header="Number of terms per query:",
            tab = 2,
        )
        print(
            "Number of Unique enriched terms:", 
            gp_res["native"].nunique()
        )
        print(
            "Number of total terms saved:",
            len(gp_res)
        )
        print(
            "Enrichment result are prepared in", 
            utils.prettyTimer(utils.getTime()-startTime)
        )
    # Return the results
    else:
        return gp_res

def subset_by_source(
        data: pd.DataFrame,
        source: str, 
        source_col: str = "source",
        query_col: str = "query",
        query_entries: list[str] = None,
        info_cols: list[str] = ["native", "name"],
        exploration_copy: bool = False,
        exploration_col: str = "Enrichment", 
        subset_copy: bool = False,
        save_path: str = None,
        analysis_name: str = None,
        verbose: bool = True
    ):
    """
        Utility function to subset the g:Profiler results by source 
        as well as create a copy in the local folder if enabled.
    """

    # Start the timer
    startTime = utils.getTime()
    # Couple of checks and preparations
    if save_path not in [None, ""]:
        if analysis_name is None:
            analysis_name = "gprofiler"
            if verbose:
                print("No analysis name provided, using 'gprofiler' as default.")
    # Try if the source_col is in the data
    if source_col not in data.columns:
        raise ValueError(f"""Column '{source_col}' not in the data.""")
    # Check if the source is in the data
    if source not in data[source_col].unique():
        raise ValueError(f"""Source '{source}' not in the data.""")
    # Check if the query_col is in the data
    if query_col not in data.columns:
        raise ValueError(f"""Column '{query_col}' not in the data.""")
    # Check if info_cols are in the data
    if not all([col in data.columns for col in info_cols]):
        raise ValueError(f"""Not all info columns are in the data.""")
    # Check if exploration_cols are in the data
    if exploration_copy: 
        if exploration_col not in data.columns:
            raise ValueError(f"""Not all exploration column are in the data.""")
    
    # Subset the data
    subset_data  = data[data[source_col] == source]

    if verbose:
        print(f"Subsetted data by source '{source}'.")
        print(f"Subsetted data has {subset_data.shape[0]} rows and {subset_data.shape[1]} columns.")
        print(f"There are {subset_data['native'].nunique()} unique terms in the subsetted data.")
    
    # Create a copy of the subset data if the subset_copy is enabled
    if subset_copy:
        if save_path:
            if verbose:
                print(f"Saving {source} subsetted data to {save_path}.")
            subset_data.to_csv(
                f"{save_path}{analysis_name}_{source}.csv",
                sep=",", 
                index=False
            )
        else:
            raise ValueError(f"""No save_path provided when copy of subset requested.""")
    
    # Create a copy of the subset data modified 
    # for better readability if the exploration_copy is enabled
    if exploration_copy:
        if save_path is None:
            raise ValueError(f"""No save_path provided when copy of exploration requested.""")
        
        # Create a Wide-format Table for better readability
        pivot_data = subset_data.pivot_table(
            index= [source_col] + info_cols,
            columns=query_col,
            values=exploration_col
        )

        # If query_entries is not None, add the missing columns with NaN values
        if query_entries is not None:
            for col in query_entries:
                if col not in pivot_data.columns:
                    pivot_data[col] = np.nan

        pivot_data = pd.concat(
            [
                pivot_data,
                pivot_data.sum(axis=1).rename(f"Sum({exploration_col})"),
                (~pivot_data.isna()).sum(axis=1).rename("Term In Query")
            ],
            axis=1
        ).sort_values(
            [
                "Term In Query",
                f"Sum({exploration_col})"
            ], 
            ascending=[1,0]
        ).reset_index()

        if verbose:
            print(f"Saving {source} exploration data to {save_path}.")
        
        pivot_data.to_csv(
            f"{save_path}{analysis_name}_{source}_exploration.csv",
            sep=",", 
            index=False
        )
     
    # End the timer
    endTime = utils.getTime()
    if verbose:
        print(
            "Subset by source completed in", 
            utils.prettyTimer(endTime - startTime)
        )

    # Return the subsetted data
    return subset_data