from __future__ import absolute_import, division, print_function
import os 
import warnings
import random
import uproot

import numpy as np

from energyflow.utils.data_utils import _get_filepath, _pad_events_axis1

__all__ = ['load']

def load(num_data=100000, generator='pythia', pad=True, with_bc=True, cache_dir='/home/liaolb/energyflow/dataset'):

    # obtain files
    Xs, ys = [], []
    cache_dir = "/home/liaolb/energyflow/dataset"
    try:
#        filename = "cepc_qq.npz" 
        filename = "zqq.root"
        fpath = cache_dir+"/"+filename 
        print(fpath)
    except Exception as e:
        print(str(e))

    f = uproot.open(fpath)["cmb"] #open the tree
    if f:
        print(f.show()) # show tuples
    
    #------- get tuples --------
    energy = f['Energy'].array()
    cosT = f['CosT'].array()
    PHI = f['PHI'].array()
    PDGID = f['PDGID'].array()
    BCL = f['BCL'].array()

    #------- combine 2-d arrays to a 3-d array -------
    comb = np.array([energy,cosT,PHI,PDGID])

    #------- transpose the axis to y-z-x -------
    outcomb = comb.transpose((1,2,0))
#    f = np.load(fpath)
    print(f)
    Xs.append(outcomb)
    ys.append(BCL)

#    try:
#        filename = "cepc_cc.npz" 
#        fpath = cache_dir+"/"+filename 
#    except Exception as e:
#        print(str(e))
#
#    f = np.load(fpath)
#    Xs.append(f['X'])
#    ys.append(f['Y'])
#    f.close()

    # get X array
    if pad:
        max_len_axis1 = max([X.shape[1] for X in Xs])
        X = np.vstack([_pad_events_axis1(x, max_len_axis1) for x in Xs])
    else:
        X = np.asarray([x[x[:,0]>0] for X in Xs for x in X], dtype='O')

    # get y array
    y = np.concatenate(ys)

    # chop down to specified amount of data
    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y
    
