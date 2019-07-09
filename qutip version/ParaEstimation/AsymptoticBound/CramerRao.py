# -*- coding: utf-8 -*-
__all__ = ['CFI', 'CFIM', 'QFI', 'QFIM']

import numpy as np
from qutip import *

def CFI(rho, drho, M):
    '''
    Calculation of classical Fisher information for a density matrix.
    Input:
    1) rho:
       TYPE: Qobj (matrix)
       DESCRIPTION: density matrix
    2) drho:
       TYPE: Qobj (matrix)
       DESCRIPTION: derivative of density matrix on the parameter to be estimated.
    3) M:
       TYPE: list of Qobj (matrix)
       DESCRIPTION: POVM measurement. It takes the form [M1, M2, ...], where M1, M2 ...
                    are matrices and satisfies M1 + M2 + ... = identity matrix.
    Output:
    1) classical Fisher information
       TYPE: real number
       DESCRIPTION: classical Fisher information
    Notice: here we assume the measurement M is independent on the parameter
            to be estimated.
    '''
    m_num = len(M)
    p = [0. for k in range(0, m_num)]
    dp = [0. for k in range(0, m_num)]
    cf = 0.
    for pi in range(0, m_num):
        mp = M[pi]
        p[pi] = (rho * mp).tr()
        dp[pi] = (drho * mp).tr()
        cadd = 0.
        if p[pi] != 0:
           cadd = (dp[pi]**2)/p[pi]
        cf += cadd

    return np.real(cf)

def CFIM(rho, drho, M):
    '''
    Calculation of classical Fisher information matrix for a density matrix.
    Input:
    1) rho:
       TYPE: Qobj (matrix)
       DESCRIPTION: density matrix
    2) drho:
       TYPE: list of Qobj (matrix)
       DESCRIPTION: derivatives of density matrix on all parameters to be estimated.
    3) M:
       TYPE: list of Qobj (matrix)
       DESCRIPTION: POVM measurement. It takes the form [M1,M2,...], where M1, M2 ...
                    are matrices satisfying M1+M2+... = identity matrix.
    Output:
    1) classical Fisher information matrix:
       TYPE: Qobj (matrix)
       DESCRIPTION: classical Fisher information matrix
    Notice:
    1) here we assume the measurement M is independent on the parameter under estimation.
    2) drho here contains the derivative on all parameters, for example, drho[0] is the
       derivative vector on the first parameter.
    '''
    if type(drho) != list:
       raise TypeError('Please make sure drho is a list since this is a multiparameter case!')

    m_num = len(M)
    para_num = len(drho)
    cfim_res = np.array([[0. for i in range(0, para_num)] for k in range(0, para_num)])

    for pi in range(0, m_num):
        mp = M[pi]
        p = (rho * mp).tr()
        cadd = np.array([[0. for i in range(0, para_num)] for k in range(0, para_num)])
        if p != 0:
           for para_i in range(para_num):
               drho_i = drho[para_i]
               dp_i = (drho_i * mp).tr()
               for para_j in range(para_i, para_num):
                   drho_j = drho[para_j]
                   dp_j = (drho_j * mp).tr()
                   cadd[para_i][para_j] = np.real(dp_i * dp_j / p)
                   cadd[para_j][para_i] = cadd[para_i][para_j]
        cfim_res += cadd

    return Qobj(cfim_res)

def SLD(rho, drho):
    '''
    Calculation of the symmetric logarithmic derivative for a density matrix.
    Input:
    1) rho:
       TYPE: Qobj (matrix)
       DESCRIPTION: density matrix
    2) drho:
       TYPE: Qobj (matrix) or a list of Qobj (matrix)
       DESCRIPTION: derivatives of density matrix on all parameters to be estimated.
    Output:
    1) symmetric logarithmic derivative:
       TYPE: same with drho
       DESCRIPTION: SLD operator (list of SLDs) for single (multi-) parameter estimation.
    '''
    #--------------------------
    # multi-parameter scenario
    #--------------------------
    if type(drho) == list:
       purity = (rho * rho).tr()

       if np.abs(1-purity) < 1e-8:
          SLD_res = [2*drho[i] for i in range(0, len(drho))]
       else:
          SLD_res = [[] for i in range(0, len(drho))]
          dim = rho.dims[0][0]
          val, vec = rho.eigenstates()
          vec_mat = Qobj([[0. + 0.j for k in range(0, dim)] for i in range(0, dim)])
          for si in range(0, dim):
              vec_mat += vec[si] * basis(dim, si).dag()

          for para_i in range(0, len(drho)):
              SLD_tp = np.array([[0.+0.*1.j for i in range(0, dim)] for k in range(0, dim)])
              for fi in range (0, dim):
                  for fj in range (0, dim):
                      coeff = 2./(val[fi]+val[fj])
                      SLD_tp[fi][fj] = coeff * (vec[fi].dag() * (drho[para_i] * vec[fj])).full().item()
              SLD_tp[SLD_tp == np.inf] = 0.
              SLD_tp = Qobj(SLD_tp)
              SLD_res[para_i] = vec_mat * (SLD_tp * vec_mat.dag())
    #---------------------------
    # single-parameter scenario
    #---------------------------
    else:
       purity = (rho * rho).tr()
       if np.abs(1-purity) < 1e-8:
          SLD_res = 2 * drho
       else:
          dim = rho.dims[0][0]
          val, vec = rho.eigenstates()
          vec_mat = Qobj([[0. + 0.j for k in range(0, dim)] for i in range(0, dim)])
          for si in range(0, dim):
              vec_mat += vec[si] * basis(dim, si).dag()

          SLD_res = np.array([[0.+0.*1.j for i in range(0, dim)] for i in range(0, dim)])
          for fi in range(0, dim):
              for fj in range(0, dim):
                  coeff = 2 / (val[fi]+val[fj])
                  SLD_res[fi][fj] = coeff * (vec[fi].dag() * (drho * vec[fj])).full().item()

          SLD_res[SLD_res == np.inf] = 0.
          SLD_res = Qobj(SLD_res)
          SLD_res = vec_mat * (SLD_res * vec_mat.dag())

    return SLD_res

def QFI(rho, drho):
    '''
    Calculation of quantum Fisher information for a density matrix.
    Input:
    1) rho:
       TYPE: Qobj (matrix)
       DESCRIPTION: density matrix
    2) drho:
       TYPE: Qobj (matrix)
       DESCRIPTION: derivative of density matrix on parameter.
    Output:
    1) quantum Fisher information:
       TYPE: real number
       DESCRIPTION: quantum Fisher information
    '''
    SLD_tp = SLD(rho, drho)
    SLD2_tp = SLD_tp * SLD_tp
    F = (rho * SLD2_tp).tr()

    return np.real(F)

def QFIM(rho, drho):
    '''
    Calculation of quantum Fisher information matrix for a density matrix.
    Input:
    1) rho:
       TYPE: Qobj (matrix)
       DESCRIPTION: density matrix
    2) drho:
       TYPE: list of Qobj
       DESCRIPTION: list of derivative of density matrix on all parameters to be estimated.
    Output:
    1) quantum Fisher information matrix:
       TYPE: Qobj (matrix)
       DESCRIPTION: quantum Fisher information matrix (the basis is the order of drho)
    Notice: drho here contains the derivative on all parameters, for example,
            drho[0] is the derivative vector on the first parameter.
    '''
    if type(drho) != list:
       raise TypeError('Multiple derivatives of density matrix are required for QFIM')

    QFIM_res = np.array([[0. for i in range(0,len(drho))] for i in range(0,len(drho))])
    SLD_tp = SLD(rho, drho)
    for para_i in range(0, len(drho)):
        for para_j in range(para_i, len(drho)):
            SLD_anti = SLD_tp[para_i] * SLD_tp[para_j] + SLD_tp[para_j] * SLD_tp[para_i]
            QFIM_res[para_i][para_j] = np.real(0.5*(rho * SLD_anti).tr())
            QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]

    return Qobj(QFIM_res)

if __name__ == '__main__':
   print(CFI.__doc__)
   print(CFIM.__doc__)
   print(QFI.__doc__)
   print(QFIM.__doc__)