# Libraries import #
from pprint import pprint
from torch import nn as nn


def make_train_tag(net_class: nn.Module,
                   lr: float,
                   aug: bool,
                   aug_p: float,
                   batch_size: int,
                   num_classes: int,
                   ):
    # Training parameters and tag
    tag_params = dict(net=net_class.__name__,
                      lr=lr,
                      aug=aug,
                      aug_p=aug_p,
                      batch_size=batch_size,
                      num_classes=num_classes
                      )
    print('Parameters')
    pprint(tag_params)
    tag = ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    print('Tag: {:s}'.format(tag))
    return tag