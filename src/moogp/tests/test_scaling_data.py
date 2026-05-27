import numpy as np

from src.moogp.datasets import (
    tstd2theta,
    xstd2x,
    generate_borehole_data_nd,
    generate_forrester_data,
    generate_borehold_data_1d,
)



def test_tstd2theta():
    # From "Constructing a simulation surrogate with partially observed output" Supplementary materials 
    # theta_1 = [990,1110]
    # theta_2 = [0.074, 1.12]
    # theta_3 = [0.05, 0.5],
    # theta_4 = [-0.5,0.5]
    tstd = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])

    theta = tstd2theta(tstd)
    assert theta.shape == (2, 4)

    bounds = np.array([
        [990, 7.46666667e-02, 0.05, -0.5],
        [1110, 1.12, 0.5, 0.5],
                    ])
    
    assert np.isclose(theta, bounds,atol=1e-6, rtol=1e-6).all()

def test_xstd2x():
    # x_1 = [0.05, 0.5]
    # x_2 = [700, 820]
    xstd = np.array([
        [0.0,0.0],
        [1.0,1.0]
    ])
    x = xstd2x(xstd)
    assert x.shape == (2,2)

    bounds = np.array([
        [0.05, 700],
        [0.5, 820],
    ])

    assert np.isclose(x, bounds,atol=1e-6, rtol=1e-6).all()
    
def test_generate_borehole_nd():
    p = 5
    data1 = generate_borehole_data_nd(n=10, p=p, seed=123)
    data2 = generate_borehole_data_nd(n=10, p=p, seed=123)

    X1 = data1["X_scaled"]
    Y1 = data1["Y"]
    loc1 = data1["locations_phys"]

    assert X1.shape == (10, 4)       # 4 theta dims
    assert Y1.shape == (10, p)
    assert loc1.shape == (p, 2)      # physical (rw, Hl)

    assert np.allclose(X1, data2["X_scaled"])
    assert np.allclose(Y1, data2["Y"])

def test_generate_forrester():
    data = generate_forrester_data(n=11, seed=42)
    assert data["X"].shape == (11, 1)
    assert data["X_scaled"].shape == (11, 1)
    assert data["y"].shape == (11, 3)


def test_generate_borehole_1d():
    data = generate_borehold_data_1d(n=8, seed=7)
    assert data["X_phys"].shape == (8, 8)
    assert data["X_scaled"].shape == (8, 8)
    assert data["y"].shape == (8,)

