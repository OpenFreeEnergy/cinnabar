from cinnabar.classification_metrics import _compute_overlap_coefficient, _create_2d_histogram,compute_fraction_best_ligands
import numpy as np
import pytest

def test_2d_histogram_wrong_shape():
    with pytest.raises(ValueError, match="same length"):
        _create_2d_histogram([1, 2, 3], [1, 2])


def test_2d_histogram_simple():
    # test creating a histogram with simple data
    # the perfect case with a diagonal histogram
    hist, xedges, yedges = _create_2d_histogram([1, 2, 3], [1, 2, 3])
    assert hist.shape == (3, 3)
    assert np.all(hist.diagonal() == 1)
    # there should only be 3 non-zero entries
    assert np.sum(hist) == 3
    # check bin edges
    ref_edges = [1, 1.5, 2.5, 3]
    assert np.allclose(xedges, ref_edges)
    assert np.allclose(yedges, ref_edges)

    # test a case with non-diagonal histogram
    hist, xedges, yedges = _create_2d_histogram([1, 3.2, 3], [1, 2, 3])
    assert hist.shape == (3, 3)
    # there should only be 3 non-zero entries
    assert np.sum(hist) == 3
    assert hist[0, 0] == 1
    assert hist[1, 2] == 1
    assert hist[2, 1] == 1


def test_overlap_coefficient_wrong_ranking():
    hist = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="greater than 0"):
        _compute_overlap_coefficient(hist, 0)
    with pytest.raises(ValueError, match="less than the number of ligands"):
        _compute_overlap_coefficient(hist, 3)


def test_overlap_coefficient_simple():
    # perfect overlap
    hist = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    overlap = _compute_overlap_coefficient(hist, 3)
    assert overlap == 1.0

    # half overlap
    hist = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    overlap = _compute_overlap_coefficient(hist, 2)
    assert overlap == 0.5

    # no overlap
    hist = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    overlap = _compute_overlap_coefficient(hist, 1)
    assert overlap == 0.0


def test_fraction_best_ligands_bad_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_fraction_best_ligands([1, 2, 3], [1, 2, 3], fraction=1.5)


def test_fraction_best_ligands_simple():
    # perfect prediction
    fraction = compute_fraction_best_ligands([1, 2, 3, 4], [1, 2, 3, 4], fraction=0.5)
    assert fraction == 1.0

    # 50% correct prediction
    fraction = compute_fraction_best_ligands([1, 2, 3, 4], [2.5, 2, 4, 3], fraction=0.5)
    assert fraction == 0.5

    # 75% correct prediction
    fraction = compute_fraction_best_ligands([1, 2, 3, 4], [1, 4, 2, 3], fraction=0.5)
    assert fraction == 0.75

    # no correct prediction
    fraction = compute_fraction_best_ligands([1, 2, 3, 4], [4, 3, 2, 1], fraction=0.5)
    assert fraction == 0.0

def test_fraction_best_ligands_regression(fe_map):
    # regression test for a real dataset
    fe_map.generate_absolute_values()

    # we need to compare the absolute experimental and calculated values so generate them
    abs_dataframe = fe_map.get_absolute_dataframe()
    exp_df = abs_dataframe[~abs_dataframe['computational']]
    calc_df = abs_dataframe[abs_dataframe['computational']]

    # get the calculated and experimental values in the same order
    merged = exp_df.merge(calc_df, on='label', suffixes=('_exp', '_calc'))
    y_true = merged['DG (kcal/mol)_exp'].values
    y_pred = merged['DG (kcal/mol)_calc'].values
    fraction = compute_fraction_best_ligands(y_true, y_pred, fraction=0.5)
    assert fraction == pytest.approx(0.7216416707838275)