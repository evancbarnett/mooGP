import numpy as np


def parse_terms_to_index_sets(terms, d, one_based=True, allow_repeated=False):
    """Parse a list of regression *terms* into index-sets.

    This helper is shared by both the design-matrix builder :func:`make_G` and the
    orthogonal-kernel utilities in :mod:`kernels`.

    Parameters
    ----------
    terms : list
        Term encoding:
          - ``None``           -> intercept (empty index set)
          - ``int j``          -> main effect for coordinate ``j``
          - ``(i, j, ...)``    -> interaction among coordinates
        Indices are interpreted as 1-based if ``one_based=True``.
    d : int
        Input dimension.
    one_based : bool
        Whether to interpret integer indices as 1-based.

    Returns
    -------
    list[tuple[int, ...]]
        Each element is a sorted tuple of coordinate indices in ``{0, ..., d-1}``.
        The intercept maps to ``()``.
    """

    J_sets = []
    for t in terms:
        if t is None:
            J_sets.append(tuple())
            continue

        if isinstance(t, (int, np.integer)):
            j = int(t) - 1 if one_based else int(t)
            if not (0 <= j < d):
                raise ValueError(f"Index {t} out of range for d={d} (one_based={one_based}).")
            J_sets.append((j,))
            continue

        # Otherwise treat as an interaction specification
        try:
            idxs = [(int(i) - 1) if one_based else int(i) for i in t]
        except Exception as e:
            raise ValueError(f"Unrecognized term: {t}") from e

        if len(idxs) < 2:
            raise ValueError(f"Interactions must have length >= 2; got {t}")
        if any((i < 0 or i >= d) for i in idxs):
            raise ValueError(f"Some indices in {t} are out of range for d={d}.")
        if (not allow_repeated) and (len(set(idxs)) != len(idxs)):
            raise ValueError(f"Duplicate indices in interaction {t}")

        J_sets.append(tuple(sorted(idxs)))

    return J_sets

def make_G(data, terms, one_based=True, return_names=False):
    """
     Creates n x r design matrix of the form
    
    Input terms as list:
    e.g. ->  terms = [None, 1, 2, 3]
    This results in g(x) = [1, x1, x2, x3]
    """
    
    

    X = np.asarray(data['X_scaled'])
    n, d = X.shape

    # Validate `terms` early (also ensures consistent parsing w.r.t. kernels.py)
    J_sets = parse_terms_to_index_sets(terms, d, one_based=one_based, allow_repeated=True)
    
    cols = []
    names = []

    for idxs in J_sets:
        if len(idxs) == 0:
            # Intercept
            cols.append(np.ones((n, 1)))
            names.append('1')
            
        elif len(idxs) == 1:
            # Single Effect
            j = idxs[0]
            cols.append(X[:, j].reshape(n, 1))
            names.append(f'x{j + 1}' if one_based else f'x{j}')
            
        else:
            # Interaction
            prod = np.prod(X[:, idxs], axis=1).reshape(n, 1)
            name = '*'.join([f'x{i+1}' if one_based else f'x{i}' for i in idxs])
            cols.append(prod)
            names.append(name)

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
