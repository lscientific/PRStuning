'''
Some string processing functions

@author: Wei Jiang (w.jiang@yale.edu)
'''

import numpy as np

def isnumeric(array):
    if isinstance(array,str):
        try:
            array = float(array)
        except:
            return(False)
    _NUMERIC_KINDS = set('buifc')
    if isinstance(array, (list,int ,float)):
        array = np.asarray(array)
    try:
        return(array.dtype.kind in _NUMERIC_KINDS)
    except:
        return(False)

#Return index of elements in aList within aDict, and also their corresponding values
def listInDict(aList, aDict):
    idx = []
    val = []
    for i in range(len(aList)):
        tmpVal = aDict.get(aList[i])
        if tmpVal is not None:
            idx.append(i)
            val.append(tmpVal)
    return np.array(idx), np.array(val)

