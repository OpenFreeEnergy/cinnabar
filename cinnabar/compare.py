import string
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

from cinnabar import FEMap
from cinnabar.stats import _AVAILABLE_STATS


def compare_and_rank_results(
    femap: FEMap,
    prediction_type: Literal["nodewise", "edgewise"] = "edgewise",
    rank_metric: Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU", "PI"] = "MUE",
    metrics_to_compute: list[Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU","PI"]] | None = None,
    num_bootstraps: int = 1_000,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare and rank multiple result sources on a single FEMap based on the chosen performance metric and return an ordered table of
    results using the compact letter display (CLD) system.

    Parameters
    ----------
    femap : FEMap
        An FEMap instance with results from multiple sources to compare.
    prediction_type: Literal["nodewise", "edgewise"], optional
        The type of prediction to evaluate, by default "edgewise".
    rank_metric : Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU", "PI"], optional
        The metric used to rank the models, by default "MUE".
    metrics_to_compute : list[Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU", "PI"]], optional
        A list of metrics to compute for each model. If None, all metrics appropriate for the `prediction_type` will be computed.
    num_bootstraps : int, optional
        The number of bootstrap samples to use for estimating confidence intervals, by default 1000.
    confidence_level : float, optional
        The confidence level for the intervals, by default 0.95.

    Note
    ----
    - The comparison method uses a joint bootstrapping procedure that generates a distribution of differences in the rank metric and checks for significant differences using a method inspired by. [1]_
    - Each source must be evaluated on the same set of edges.
    - Prediction types "nodewise" and "edgewise" correspond to DGs and edgewise DDGs respectively.
    - When we have more than 2 models, we apply multiple testing correction to the pairwise comparisons using the ``Holm`` method by default.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - The first DataFrame contains the computed metrics for each model.
        - The second DataFrame contains the pairwise comparison results between models.

    References
    ----------
    .. [1] https://www.nature.com/articles/s42004-025-01428-y
    """

    # get the predictions and experimental values from the FEMaps and align them into a DataFrame
    predictions_by_key = defaultdict(dict)
    if prediction_type == "nodewise":
        # make sure we have absolute values
        femap.generate_absolute_values()
        abs_df = femap.get_absolute_dataframe()
        # get the computational sources we want to compare
        sources = abs_df[abs_df["computational"] == True]["source"].unique()
        # for each row add the node prediction
        for _, row in abs_df.iterrows():
            node_label = row["label"]
            if not row["computational"]:
                source_label = "exp"
            else:
                source_label = f"{row["source"]}_calc"
            predictions_by_key[node_label][source_label] = row["DG (kcal/mol)"]

    elif prediction_type == "edgewise":
        rel_df = femap.get_relative_dataframe()
        sources =rel_df[rel_df["computational"] == True]["source"].unique()
        for _, row in rel_df.iterrows():
            key = (row["labelA"], row["labelB"])
            if not row["computational"]:
                source_label = "exp"
            else:
                source_label = f"{row["source"]}_calc"
            predictions_by_key[key][source_label] = row["DDG (kcal/mol)"]
    else:
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    predictions_df = pd.DataFrame(predictions_by_key.values())

    # check that we have experimental values for all edges
    if "exp" not in predictions_df.columns:
        raise ValueError("Experimental values are required to rank the results.")

    # check that we have the same number of values for all sources
    for source in sources:
        # check if any values for this source are missing
        if any(predictions_df[f"{source}_calc"].isna()):
            raise ValueError(f"Missing predictions for source {source}, all sources must have the same number of predictions.")

    # set metrics to compute based on best practices if not provided
    if metrics_to_compute is None:
        if prediction_type == "edgewise":
            metrics_to_compute = ["MUE", "RMSE"]
        elif prediction_type == "nodewise":
            metrics_to_compute = ["MUE", "RMSE", "RAE", "R2", "rho", "KTAU", "PI"]

    # we must compute the rank metric however it is possible to miss it
    if rank_metric not in metrics_to_compute:
        metrics_to_compute.append(rank_metric)

    # check we can compute all requested metrics
    for metric in metrics_to_compute:
        if metric not in _AVAILABLE_STATS:
            raise ValueError(f"Metric {metric} is not available.")

    metrics_by_source = dict((source, dict((metric, []) for metric in metrics_to_compute)) for source in sources)
    pairwise_metrics = {}
    # compute bootstrap metrics for each model using the same bootstrap samples, joint bootstrap
    for _ in range(num_bootstraps):
        bootstrap_sample = predictions_df.sample(frac=1.0, replace=True)
        for source in sources:
            y_true = bootstrap_sample["exp"].values
            y_pred = bootstrap_sample[f"{source}_calc"].values
            for metric in metrics_to_compute:
                value = _AVAILABLE_STATS[metric](y_true, y_pred)
                metrics_by_source[source][metric].append(value)

    # compute pairwise differences for ranking metric
    for i, source_i in enumerate(sources):
        for j, source_j in enumerate(sources):
            if j <= i:
                continue
            diffs = np.array(metrics_by_source[source_i][rank_metric]) - np.array(metrics_by_source[source_j][rank_metric])
            pairwise_metrics[(source_i, source_j)] = diffs

    # summarize all metrics
    summary_data = []
    for source in sources:
        row = {"Model": source}
        for metric in metrics_to_compute:
            # compute the sample metric
            x = predictions_df[f"{source}_calc"].values
            y = predictions_df["exp"].values
            sample_value = _AVAILABLE_STATS[metric](y, x)
            bootstrap_values = np.array(metrics_by_source[source][metric])
            lower = np.percentile(bootstrap_values, (1 - confidence_level) / 2 * 100)
            upper = np.percentile(bootstrap_values, (1 + confidence_level) / 2 * 100)
            row[f"{metric}"] = sample_value
            row[f"{metric}_CI_Lower"] = lower
            row[f"{metric}_CI_Upper"] = upper
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)

    # summarize pairwise comparisons with corrected p-values
    comparison_data = []
    for (source_i, source_j), diffs in pairwise_metrics.items():
        lower = np.percentile(diffs, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(diffs, (1 + confidence_level) / 2 * 100)
        # calculate the p-value as the fraction of bootstrap samples that cross zero
        # use a 2-tailed test
        # inspired by https://www.nature.com/articles/s42004-025-01428-y
        p_value = 2 * min(np.mean(diffs < 0), np.mean(diffs > 0))
        comparison_data.append(
            {
                "Model 1": source_i,
                "Model 2": source_j,
                f"Diff in {rank_metric}": summary_df[summary_df["Model"] == source_i][rank_metric].values[0]
                - summary_df[summary_df["Model"] == source_j][rank_metric].values[0],
                "CI Lower": lower,
                "CI Upper": upper,
                "p-value": p_value,
                "significant": p_value < 0.05,
            }
        )
    comparison_df = pd.DataFrame(comparison_data)

    # if we have more than 2 models, apply multiple testing correction
    if len(sources) > 2:
        from statsmodels.stats.multitest import multipletests

        p_values = comparison_df["p-value"].values
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="holm")
        comparison_df["p-value corrected"] = pvals_corrected
        # add corrected significance
        comparison_df["significant"] = pvals_corrected < 0.05

    # rank the models and apply the CLD
    # the order depends on whether lower or higher is better for the rank metric
    if rank_metric in ["MUE", "RMSE", "RAE"]:
        ascending = True  # lower is better
    else:
        ascending = False  # higher is better
    ordered_labels = summary_df.sort_values(by=rank_metric, ascending=ascending).Model.tolist()
    cld_assignment = _apply_cld(comparison_df, ordered_labels)
    summary_df["CLD"] = summary_df["Model"].map(cld_assignment)

    return summary_df, comparison_df


def _build_significance_lookup(pairwise_df: pd.DataFrame):
    """
    Build a lookup dictionary for significance between model pairs.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame containing pairwise comparison results with a 'significant' column.

    Returns
    -------
    dict[frozenset, bool]
        A dictionary mapping frozensets of model pairs to their significance (True if significant, False otherwise).
    """
    sig = {}
    for _, row in pairwise_df.iterrows():
        m1, m2 = row["Model 1"], row["Model 2"]
        sig[frozenset((m1, m2))] = row.significant
    return sig


def _apply_cld(comparison_df: pd.DataFrame, ordered_labels: list[str]) -> dict[str, str]:
    """
    Apply the Compact Letter Display (CLD) system to the pairwise comparison results using the insert-absorb method.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame containing pairwise comparison results with a 'significant' column.
    ordered_labels : list[str]
        List of model labels sorted by metric performance (best to worst).

    Returns
    -------
    dict[str, str]
        A dictionary mapping each model label to its assigned CLD letters.
    """

    sig = _build_significance_lookup(comparison_df)

    # Each letter is represented as a set of methods which are not significantly different
    letters = []

    def is_compatible(meth, let):
        """Check if method can join this letter set"""
        for other in let:
            if sig.get(frozenset((meth, other)), False):
                return False
        return True

    def absorb(lets):
        """Remove letters that are strict subsets of others"""
        absorbed = []
        for i, a in enumerate(lets):
            redundant = False
            for j, b in enumerate(lets):
                if i != j and a < b:  # a is a strict subset of b
                    redundant = True
                    break
            if not redundant:
                absorbed.append(a)
        return absorbed

    # For each method in order, try to insert into existing letters or create new letter
    for method in ordered_labels:
        inserted = False

        # Try inserting into existing letters
        for letter in letters:
            if is_compatible(method, letter):
                letter.add(method)
                inserted = True

        # If no insertion possible, create new letter
        if not inserted:
            letters.append({method})

        # Absorb step
        letters = absorb(letters)

    # Convert to method → string mapping
    alphabet = string.ascii_lowercase
    if len(letters) > len(alphabet):
        raise ValueError("Too many letters required for CLD")

    result = {m: "" for m in ordered_labels}
    for i, letter in enumerate(letters):
        for m in letter:
            result[m] += alphabet[i]

    return result
