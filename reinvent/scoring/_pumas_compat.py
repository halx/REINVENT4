try:
    import pumas  # noqa: F401

    PUMAS_AVAILABLE = True
except ImportError:
    PUMAS_AVAILABLE = False


def require_pumas() -> None:
    if not PUMAS_AVAILABLE:
        raise ImportError(
            "use_pumas=True requires the 'pumas' package: " "pip install 'reinvent[pumas]'"
        )
