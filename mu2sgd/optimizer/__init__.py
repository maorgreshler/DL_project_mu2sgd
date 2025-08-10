from torch.optim import SGD, Adam
from .anytime_sgd import AnyTimeSGD
from .mu2sgd import Mu2SGD
from .storm import STORM
from .adaptive_mu2sgd import AdaptiveMu2SGD


OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'momentum': SGD,
    'adam': Adam,
    'adaptive_mu2sgd': AdaptiveMu2SGD,
    'anytime_sgd': AnyTimeSGD,
    'mu2sgd': Mu2SGD,
    'storm': STORM
}


