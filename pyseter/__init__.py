"""
Pyseter
Open source photo-identification in Python
"""

__version__ = "0.3.2"

# Lazy imports to avoid loading extract module at package import time
def __getattr__(name):
    if name == "verify_pytorch":
        from pyseter.extract import verify_pytorch
        return verify_pytorch
    elif name == "get_best_device":
        from pyseter.extract import get_best_device
        return get_best_device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Define what gets imported with "from your_package import *"
__all__ = ["verify_pytorch", "get_best_device"]