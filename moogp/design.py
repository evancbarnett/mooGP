import numpy as np

def make_G(data, terms, one_based=True, return_names=False):
    """
     Creates n x r design matrix of the form
    
    Input terms as list:
    e.g. ->  terms = [None, 1, 2, 3]
    This results in g(x) = [1, x1, x2, x3]
    """
    
    

    X = np.asarray(data['X_scaled'])
    n, d = X.shape
    
    cols = []
    names = []

    def add_col(name, col):
            cols.append(col.reshape(n, 1))
            names.append(name)

    for t in terms:
        # None -> Intercept
        if t is None:
            add_col('1', np.ones(n))
            continue

        # int -> Single Effect
        if isinstance(t, (int, np.integer)):
            j = t - 1 if one_based else t
            if j < 0 or j >= d:
                raise ValueError(f"Index {t} out of range for d={d} (one_based={one_based}).")
            add_col(f'x{j+1}', X[:, j])
            continue

        # (int, int) -> Interaction
        else:
            try:
                idxs = [(i - 1) if one_based else int(i) for i in t]
            except Exception as e:
                raise ValueError(f"Unrecognized term: {t}") from e

        if len(idxs) < 2:
            raise ValueError(f"Interactions must have length >= 2; got {t}")
        if any((i < 0 or i >= d) for i in idxs):
            raise ValueError(f"Some indices in {t} are out of range for d={d}.")


        prod = np.prod(X[:, idxs], axis=1)
        name = '*'.join([f'x{i+1}' for i in idxs])
        add_col(name, prod)

    G = np.hstack(cols) if cols else np.empty((n, 0))
    return (G, names) if return_names else G


def build_Gy(G,p):
    return np.kron(np.eye(p), G)

def vecF(A):
    """vec with column-stacking"""
    return A.reshape(-1, order='F')

def unvecF(v, r, p):
    """inverse of vecF."""
    return v.reshape((r, p), order='F')
