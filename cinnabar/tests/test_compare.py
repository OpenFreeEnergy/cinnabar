import numpy as np
import pytest
from openff.units import unit

from cinnabar import FEMap
from cinnabar.compare import compare_and_rank_results


def test_compare_and_rank_results(fe_map):
    np.random.seed(42)

    compare_map = FEMap()
    for m in fe_map:
        if m.computational:
            # add the result with a new source
            compare_map.add_relative_calculation(
                labelA=m.labelA, labelB=m.labelB, value=m.DG, uncertainty=m.uncertainty, source="original"
            )
            # add the data again under a second source with some noise added
            compare_map.add_relative_calculation(
                labelA=m.labelA,
                labelB=m.labelB,
                value=np.random.normal(m.DG.m, m.uncertainty.m) * unit.kilocalorie_per_mole,
                uncertainty=m.uncertainty,
                source="perturbed",
            )
            # add a third set of data with a lot of noise to get a significant difference
            compare_map.add_relative_calculation(
                labelA=m.labelA,
                labelB=m.labelB,
                value=np.random.normal(m.DG.m, 12.0 * m.uncertainty.m) * unit.kilocalorie_per_mole,
                uncertainty=m.uncertainty,
                source="noisy",
            )
        else:
            # add the experimental data
            compare_map.add_measurement(m)

    summary_df, comparison_df = compare_and_rank_results(
        compare_map,
        prediction_type="edgewise",
        rank_metric="MUE",
        metrics_to_compute=["MUE", "RMSE"],
    )

    # make sure the MUE and RMSE have been calculated and recorded in the summary table
    for metric in ["MUE", "RMSE"]:
        assert metric in summary_df.columns
        for ci in ["Upper", "Lower"]:
            assert f"{metric}_CI_{ci}" in summary_df.columns
    # check that noisy is ranked worst
    assert summary_df[summary_df["Model"] == "noisy"]["CLD"].values[0] == "b"
    # check that original and perturbed are ranked better
    assert summary_df[summary_df["Model"] == "original"]["CLD"].values[0] == "a"
    assert summary_df[summary_df["Model"] == "perturbed"]["CLD"].values[0] == "a"
    # check that the comparison table has all three models and corrected p-values
    assert len(comparison_df) == 3
    assert "p-value corrected" in comparison_df.columns


def test_nodewise_comparison(fe_map):
    np.random.seed(42)

    compare_map = FEMap()
    for m in fe_map:
        if m.computational:
            # add the result with a new source
            compare_map.add_relative_calculation(
                labelA=m.labelA, labelB=m.labelB, value=m.DG, uncertainty=m.uncertainty, source="original"
            )
            # add the data again under a second source with some noise added
            compare_map.add_relative_calculation(
                labelA=m.labelA,
                labelB=m.labelB,
                value=np.random.normal(m.DG.m, m.uncertainty.m) * unit.kilocalorie_per_mole,
                uncertainty=m.uncertainty,
                source="perturbed",
            )
        else:
            # add the experimental data
            compare_map.add_measurement(m)

    # add the absolute values to the map
    compare_map.generate_absolute_values()
    # a simple test to make sure it runs
    summary_df, comparison_df = compare_and_rank_results(
        compare_map,
        prediction_type="nodewise",
        rank_metric="PI",
    )
    for metric in ["MUE", "RMSE", "R2", "rho", "PI", "KTAU", "RAE"]:
        assert metric in summary_df.columns
        for ci in ["Upper", "Lower"]:
            assert f"{metric}_CI_{ci}" in summary_df.columns
        # check that original and perturbed are ranked the same
        assert summary_df[summary_df["Model"] == "MLE(original)"]["CLD"].values[0] == "a"
        assert summary_df[summary_df["Model"] == "MLE(perturbed)"]["CLD"].values[0] == "a"
        # check that the comparison table has a single comparison between the two models
        assert len(comparison_df) == 1
        # as we have two methods we do not need to correct the p-values
        assert "p-value corrected" not in comparison_df.columns


def test_invalid_prediction_type(fe_map):
    with pytest.raises(ValueError, match="Invalid prediction_type: pairwise"):
        compare_and_rank_results(
            fe_map,
            prediction_type="pairwise",
            rank_metric="MUE",
            metrics_to_compute=["MUE", "RMSE"],
        )


def test_missing_experimental_data(fe_map):
    new_map = FEMap()
    for m in fe_map:
        if m.computational:
            new_map.add_measurement(m)

    with pytest.raises(ValueError, match="Experimental values are required to rank the results."):
        compare_and_rank_results(
            new_map,
        )


def test_missing_source_data(fe_map):
    new_map = FEMap()
    for i, m in enumerate(fe_map):
        if not m.computational:
            new_map.add_measurement(m)
        else:
            new_map.add_relative_calculation(
                labelA=m.labelA, labelB=m.labelB, value=m.DG, uncertainty=m.uncertainty, source="original"
            )
            # for even I add the second source as well
            if i % 2 == 0:
                new_map.add_relative_calculation(
                    labelA=m.labelA, labelB=m.labelB, value=m.DG, uncertainty=m.uncertainty, source="perturbed"
                )

    with pytest.raises(
        ValueError,
        match="Missing predictions for source perturbed, all sources must have the same number of predictions.",
    ):
        compare_and_rank_results(
            new_map,
        )


def test_bad_metric(fe_map):
    with pytest.raises(ValueError, match="Metric bad_metric is not available."):
        compare_and_rank_results(
            fe_map,
            rank_metric="bad_metric",
        )


def test_missing_rank_metric(fe_map):
    _, _ = compare_and_rank_results(
        fe_map,
        rank_metric="MUE",
        metrics_to_compute=["RMSE"],  # miss the rank metric from the compute list and it should still work
    )


def test_missing_node_values(fe_map):
    with pytest.raises(ValueError, match="The FEMap contains no computed absolute values. "):
        _, _ = compare_and_rank_results(
            fe_map,
            prediction_type="nodewise",
        )


def test_low_bootstrap_samples(fe_map):
    with pytest.raises(ValueError, match="num_bootstraps must be an integer >= 1."):
        _, _ = compare_and_rank_results(fe_map, num_bootstraps=0)


@pytest.mark.parametrize("ci", [0, 1, "one"])
def test_bad_ci(fe_map, ci):
    with pytest.raises(ValueError, match="confidence_level must be a number between 0 and 1"):
        _, _ = compare_and_rank_results(
            fe_map,
            confidence_level=ci,
        )


@pytest.mark.parametrize("alpha", [0, 1, "one"])
def test_bad_alpha(fe_map, alpha):
    with pytest.raises(ValueError, match="alpha must be a number between 0 and 1"):
        _, _ = compare_and_rank_results(
            fe_map,
            alpha=alpha,
        )
