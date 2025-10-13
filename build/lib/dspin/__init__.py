
from . import annotate
from . import compute
from . import plot
from . import dspin
from importlib.metadata import version 

# Import main classes for easy access
from .dspin import DSPIN, GeneDSPIN, ProgramDSPIN, AbstractDSPIN

__all__ = [
    'DSPIN',
    'GeneDSPIN', 
    'ProgramDSPIN',
    'AbstractDSPIN',
    'annotate',
    'compute',
    'plot'
]

__version__ = version("DSPIN")