import string

import numpy as np

from cinnabar import FEMap
from typing import Literal
import pandas as pd
from collections import defaultdict
from cinnabar.stats import AVAILABLE_STATS
import itertools



def compare_and_rank_femaps(
    femaps: list[FEMap],
    labels: list[str],
    prediction_type: Literal["nodewise", "edgewise", "pairwise"] = "pairwise",
    rank_metric: Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU"] = "MUE",
    metrics_to_compute: list[Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU"]] | None = None,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare and rank multiple FEMaps based on the chosen performance metric and return an ordered table of
    results using the compact letter display (CLD) system.

    Parameters
    ----------
    femaps : list[FEMap]
        A list of FEMap instances to compare.
    labels : list[str]
        A list of labels corresponding to each FEMap used in the output tables.
    prediction_type: Literal["nodewise", "edgewise", "pairwise"], optional
        The type of prediction to evaluate, by default "pairwise".
    rank_metric : Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU"], optional
        The metric used to rank the models, by default "MUE".
    metrics_to_compute : list[Literal["MUE", "RMSE", "RAE", "R2", "rho", "KTAU"]], optional
        A list of metrics to compute for each model. If None, all metrics appropriate for the `prediction_type` will be computed.
    num_bootstraps : int, optional
        The number of bootstrap samples to use for estimating confidence intervals, by default 1000.
    confidence_level : float, optional
        The confidence level for the intervals, by default 0.95.

    Note
    ----
    - We assume that all FEMaps have been evaluated on the same test set.
    - Prediction types "nodewise", "edgewise", and "pairwise" correspond to DGs, edgewise DDGs and pairwise DDGs (back calculated from nodewise DGs) respectively.
    - When we have more than 2 models, we apply multiple testing correction to the pairwise comparisons using the ``Holm`` method by default.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - The first DataFrame contains the computed metrics for each model.
        - The second DataFrame contains the pairwise comparison results between models.
    """


    # get the predictions and experimental values from the FEMaps and align them into a DataFrame
    predictions_by_key = defaultdict(dict)
    for label, femap in zip(labels, femaps):
        graph = femap.to_legacy_graph()
        if prediction_type == "nodewise":
            #TODO  should we expose a shift or centering option here?
            for node, data in graph.nodes(data=True):
                predictions_by_key[node][f"{label}_calc"] = data["calc_DG"]
                if "exp_DG" not in predictions_by_key[node]:
                    predictions_by_key[node]["exp"] = data["exp_DG"]
        elif prediction_type == "edgewise":
            for a, b, data in graph.edges(data=True):
                # we assume all edges have been run in the same direction
                key = (a, b)
                predictions_by_key[key][f"{label}_calc"] = data["calc_DDG"]
                if "exp_DDG" not in predictions_by_key[key]:
                    predictions_by_key[key]["exp"] = data["exp_DDG"]
        elif prediction_type == "pairwise":
            nodes = list(graph.nodes())
            for a, b in itertools.combinations(nodes, 2):
                exp = graph.nodes[b]["exp_DG"] - graph.nodes[a]["exp_DG"]
                calc = graph.nodes[b]["calc_DG"] - graph.nodes[a]["calc_DG"]
                key = (a, b)
                predictions_by_key[key][f"{label}_calc"] = calc
                if "exp" not in predictions_by_key[key]:
                    predictions_by_key[key]["exp"] = exp
        else:
            raise ValueError(f"Invalid prediction_type: {prediction_type}")

    predictions_df = pd.DataFrame(predictions_by_key.values())

    # set metrics to compute based on best practices if not provided
    if metrics_to_compute is None:
        if prediction_type in ["pairwise", "edgewise"]:
            metrics_to_compute = ["MUE", "RMSE"]
        elif prediction_type == "nodewise":
            metrics_to_compute = ["MUE", "RMSE", "RAE", "R2", "rho", "KTAU"]

    # check we can compute all requested metrics
    for metric in metrics_to_compute:
        if metric not in AVAILABLE_STATS:
            raise ValueError(f"Metric {metric} is not available.")

    metrics_by_label =  dict((label, dict((metric, []) for metric in metrics_to_compute)) for label in labels)
    pairwise_metrics = {}
    # compute bootstrap metrics for each model using the same bootstrap samples, joint bootstrap
    for _ in range(num_bootstraps):
        bootstrap_sample = predictions_df.sample(frac=1.0, replace=True)
        for label in labels:
            y_true = bootstrap_sample["exp"].values
            y_pred = bootstrap_sample[f"{label}_calc"].values
            for metric in metrics_to_compute:
                value = AVAILABLE_STATS[metric](y_true, y_pred)
                metrics_by_label[label][metric].append(value)

    # compute pairwise differences for ranking metric
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if j <= i:
                continue
            diffs = np.array(metrics_by_label[label_i][rank_metric]) - np.array(metrics_by_label[label_j][rank_metric])
            pairwise_metrics[(label_i, label_j)] = diffs

    # summarize all metrics
    summary_data = []
    for label in labels:
        row = {"Model": label}
        for metric in metrics_to_compute:
            # compute the sample metric
            x = predictions_df[f"{label}_calc"].values
            y = predictions_df["exp"].values
            sample_value = AVAILABLE_STATS[metric](y, x)
            bootstrap_values = np.array(metrics_by_label[label][metric])
            lower = np.percentile(bootstrap_values, (1 - confidence_level) / 2 * 100)
            upper = np.percentile(bootstrap_values, (1 + confidence_level) / 2 * 100)
            row[f"{metric}"] = sample_value
            row[f"{metric}_CI_Lower"] = lower
            row[f"{metric}_CI_Upper"] = upper
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)

    # summarize pairwise comparisons with corrected p-values
    comparison_data = []
    for (label_i, label_j), diffs in pairwise_metrics.items():
        lower = np.percentile(diffs, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(diffs, (1 + confidence_level) / 2 * 100)
        # calculate the p-value as the fraction of bootstrap samples that cross zero
        # use a 2-tailed test
        # inspired by https://www.nature.com/articles/s42004-025-01428-y
        p_value = 2 * min(np.mean(diffs < 0), np.mean(diffs > 0))
        comparison_data.append({
            "Model 1": label_i,
            "Model 2": label_j,
            f"Diff in {rank_metric}": summary_df[summary_df["Model"] == label_i][rank_metric].values[0] - summary_df[summary_df["Model"] == label_j][rank_metric].values[0],
            "CI Lower": lower,
            "CI Upper": upper,
            "p-value": p_value,
            "significant": p_value < 0.05
        })
    comparison_df = pd.DataFrame(comparison_data)

    # if we have more than 2 models, apply multiple testing correction
    if len(labels) > 2:
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



