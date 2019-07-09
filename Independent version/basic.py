# -*- coding: utf-8 -*-
'''
This module inculdes several basic functions and operators in quantum mechanics.
1) Pauli matrices:
sigmaz = |1><1|-|0><0|=|e><e|-|g><g|, where |0>=|g> is the ground state and |1>=e> is the excited state.
'''

__all__ = ['sigmax','sigmay','sigmaz','sigmaplus','sigmaminus']

import numpy as np

#===============================================================================


def sigmax():
    '''
    sigma_x = |1><0|＋|0><1|=|e><g|＋|g><e|.
    '''
    result = np.array([[0,1],[1,0]])
    return result
    
def sigmay():
    '''
    sigma_y = -i|1><0|＋i|0><1|=-i|e><g|＋i|g><e|.
    '''
    result = np.array([[0,-1.j],[1.j,0]])
    return result
        
def sigmaz():
    '''
    sigma_z = |1><1|-|0><0|＝|e><e|-|g><g|.
    '''
    result = np.array([[1.,0.],[0.,-1.]])
    return result    
    
def sigmaplus():
    '''
    sigma_{+} = |0><1|＝|e><g|.
    '''
    result = np.array([[0.,1.],[0.,0.]])
    return result
    
def sigmaminus():
    '''
    sigma_{-} = |1><0|＝|g><e|.
    '''
    result = np.array([[0.,0.],[1.,0.]])
    return result           
#===============================================================================

if __name__ == '__main__':
   print(__doc__)
