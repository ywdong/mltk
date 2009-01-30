"""\
the TDT2 corpus (Nist Topic Detection and Tracking corpus) consists of
data collected during the first half of 1998 and taken from 6 sources,
including 2 newswires (APW, NYT), 2 radio programs (VOA, PRI) and 2
television programs (CNN, ABC). It constis of 11201 on-topic documents
which are classified into 96 semantic categories. 

In this subset, those documents appearing in two or more categories were
removed, and only the largest 30 categories were kept, thus leaving us with
9394 documents in total.
"""

from random import sample
from os.path import join, dirname
from scipy.io import loadmat
from numpy import hstack, array

__docformat__ = 'restructuredtext'
__all__ = ['get_dataset']

tdt2 = None
category_count = 30

def get_dataset(categories=None, drop_zeros=True):
    global tdt2
    if tdt2 is None:
        tdt2 = loadmat(join(dirname(__file__), 'tdt2.mat'))
    if categories is None:
        return tdt2['fea'].copy, tdt2['gnd'].copy()
    else:
        if isinstance(categories, int):
            categories = sample(range(category_count), categories)
        else:
            if max(categories) >= category_count or \
                    min(categories) < 0:
                raise ValueError('categories should be within [0, %d)' %
                        category_count)
        gnd = tdt2['gnd']
        idx = hstack([(gnd == categories[i]).nonzero()[0] \
                for i in range(len(categories))])
        fea = tdt2['fea'][idx]
        if drop_zeros:
            fea = fea[:, array(fea.sum(axis=0).nonzero()[0])[0]]
        return fea, gnd[idx]

