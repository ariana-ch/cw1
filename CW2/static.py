from enum import Enum, auto

class DataSplit(Enum):
    TRAIN   = 'train',
    TEST    = 'test',
    VAL     = 'validation',
    ALL     = ('train', 'validation', 'test')

class DataClass(Enum):
    TENSORFLOW  = auto()
    TORCH       = auto()

