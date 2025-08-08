from torch.optim import SGD, Adam
from .anytime_sgd import AnyTimeSGD
from .mu2sgd import Mu2SGD
from .storm import STORM


OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'momentum': SGD,
    'adam': Adam,
    'anytime_sgd': AnyTimeSGD,
    'mu2sgd': Mu2SGD,
    'storm': STORM
}


