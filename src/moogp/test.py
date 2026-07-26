def test(level=0):
    """
    Run full set of surmise tests.

    Parameters
    ----------
    level : int
        Smaller values indicate less logging

    Returns
    -------
    bool
        True if all tests passed; False, otherwise.
    """
    VERBOSITY = [0, 1, 2]

    if level not in VERBOSITY:
        raise ValueError(f"level must be in {VERBOSITY}")

    try:
        import pytest
    except ImportError as exc:
        raise ImportError(
            "moogp.test() requires pytest; install the project's test dependencies first."
        ) from exc

    cmd = [f"--verbosity={level}", "--pyargs", "moogp.tests"]

    return (pytest.main(cmd) == 0)
