
from tensorflow import keras
try:
    __version__ = keras.__version__
except AttributeError:
    __version__ = "unknown"
