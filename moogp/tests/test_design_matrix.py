import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from moogp.design import make_G, build_Gy, vecF, unvecF


def sample_data():
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
    return {'X_scaled': X}


def test_make_G_no_interactions():
    data = sample_data()
    G, names = make_G(data, terms=[None, 1, 3], one_based=True, return_names=True)
    expected = np.column_stack([np.ones(3), data['X_scaled'][:, 0], data['X_scaled'][:, 2]])
    assert G.shape == (3, 3)
    assert_array_equal(G, expected)
    assert names == ['1', 'x1', 'x3']


def test_make_G_interaction():
    data = sample_data()
    G, names = make_G(data, terms=[None, (1, 2)], one_based=True, return_names=True)
    expected_interaction = data['X_scaled'][:, 0] * data['X_scaled'][:, 1]
    assert_array_equal(G[:, 0], np.ones(3))
    assert_array_equal(G[:, 1], expected_interaction)
    assert names == ['1', 'x1*x2']


def test_make_G_repeated_index_interaction():
    data = sample_data()
    G, names = make_G(data, terms=[(1, 1)], one_based=True, return_names=True)
    expected = (data['X_scaled'][:, 0] ** 2).reshape(3, 1)
    assert_array_equal(G, expected)
    assert names == ['x1*x1']


def test_make_G_empty_terms_returns_empty_matrix():
    data = sample_data()
    G = make_G(data, terms=[])
    assert G.shape == (3, 0)


def test_make_G_single_index_out_of_range_raises():
    data = sample_data()
    with pytest.raises(ValueError):
        make_G(data, terms=[10], one_based=True)


def test_make_G_interaction_length_one_raises():
    data = sample_data()
    with pytest.raises(ValueError):
        make_G(data, terms=[(2,)], one_based=True)


def test_make_G_malformed_term_raises():
    data = sample_data()
    # a string is not a valid term; should raise ValueError
    with pytest.raises(ValueError):
        make_G(data, terms=['not_a_term'])


def test_build_Gy_kron_structure_and_shape():
    G = np.array([[1, 2],
                  [3, 4]])
    p = 3
    Gy = build_Gy(G, p)
    expected = np.kron(np.eye(p), G)
    assert_array_equal(Gy, expected)
    assert Gy.shape == (G.shape[0] * p, G.shape[1] * p)


def test_vecF_and_unvecF():
    # Create A with shape (3,2) filled column-wise
    A = np.arange(1, 7).reshape((3, 2), order='F')  # [[1,4],[2,5],[3,6]]
    v = vecF(A)
    # column-major stacking yields [1,2,3,4,5,6]
    assert_array_equal(v, np.array([1, 2, 3, 4, 5, 6]))
    A_rec = unvecF(v, 3, 2)
    assert_array_equal(A_rec, A)
