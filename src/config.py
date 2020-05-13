
'''
import numpy as np

data_directory = '/home/reyhaneh/VGG/data/'
num_cls = 10
rgb_mean = [0,0,0]
batch_size = None
epoch = None
CHECKPOINT_DIR = '/home/reyhaneh/VGG/checkpoint/'
'''

import os.path  as osp
import numpy    as np
from easydict import EasyDict as edict



__C = edict()
cfg = __C
__C.DATA_SETS_TYPE='cifar-10'

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
__C.DATA_SETS_DIR=osp.join(__C.ROOT_DIR, 'data/')

__C.CHECKPOINT_DIR=osp.join(__C.ROOT_DIR,'checkpoint')
__C.LOG_DIR=osp.join(__C.ROOT_DIR,'log')

#__C.num_class = 10
__C.num_out = 3321
__C.image_size = [32,32,1]
__C.img_channels = 1
__C.degree = 80
__C.rgb_mean = [0.0,0.0,0.0]



# if timer is needed.
__C.TRAINING_TIMER = True
__C.TRACKING_TIMER = True
__C.DATAPY_TIMER = False

__C.USE_CLIDAR_TO_TOP = True

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

if __name__ == '__main__':
    print('__C.ROOT_DIR = '+__C.ROOT_DIR)