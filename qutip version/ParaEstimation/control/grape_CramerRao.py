# -*- coding: utf-8 -*-
__all__ = ['control']

import numpy as np
from qutip import *
from qutip.superoperator import liouvillian, mat2vec, vec2mat
import ParameterEstimation.AsymptoticBound.CramerRao as CR

class control:
      '''
      control generation with gradient ascent pulse engineering (GRAPE)
      Input:
      1) H0:
      2) Lvec:
      3) gamma:
      4) rho_initial:
      5) times:
         TYPE: np.array
         DESCRIPTION: time array for the evolution
      6) dH:
      7) Hc_opt:
      8) Hc_coeff:
      '''
      def __init__(self, H0, rho_initial, times, Lvec=[], gamma=[], dH=[], Hc_opt=[], Hc_coeff=[], epsilon=0.01):
          self.freeHamiltonian = H0
          self.Liouville_operator = Lvec
          self.gamma = gamma
          self.rho_initial = rho_initial
          self.times = times
          self.Hamiltonian_derivative = dH
          self.control_Hamiltonian = Hc_opt
          self.control_coefficients = Hc_coeff
          self.epsilon = epsilon

          self.rho = None
          self.rho_derivative = None
          self.propagator = None

      def evolution(self, t):
          '''
          internal function
          '''
          H0 = self.freeHamiltonian
          tj = (t-self.times[0])/(self.times[1]-self.times[0])
          dt = self.times[1]-self.times[0]

          for hn in range(0, len(self.control_Hamiltonian)):
              Hc_temp = self.control_coefficients[hn]
              H0 += self.control_Hamiltonian[hn] * Hc_temp[tj]

          Ld = dt * Qobj(liouvillian(H0, c_ops=self.Liouville_operator).full())
          return Ld.expm()

#=================================================================
#                     Single parameter scenario
#=================================================================
      def propagation_single(self):
          '''
          This function will save all the propators during the evolution, which may be memory consuming.
          '''
          if len(self.Hamiltonian_derivative) != 1:
             raise TypeError('this is single parameter scenario, the length of dH has to be 1!')

          num = len(self.times)
          dim = self.freeHamiltonian.dims[0][0]
          dH = self.Hamiltonian_derivative
          dL = Qobj(liouvillian(dH[0]).full())
          dt = self.times[1]-self.times[0]

          D = [[[] for i in range(0,num+1)] for k in range(0,num+1)]
          rhovec = [[] for i in range(0,num)]
          drhovec = [[] for i in range(0,num)]

          rhovec[0] = Qobj(mat2vec(self.rho_initial.full()))
          drhovec[0] = dt * dL * rhovec[0]
          D[0][0] = self.evolution(0)
          D[1][0] = qeye(dim**2)

          for di in range(1,num):
              tnow = self.times[0] + dt * di
              D[di+1][di] = qeye(dim**2)
              D[di][di] = self.evolution(tnow)
              D[0][di] = D[di][di] * D[0][di-1]
              rhovec[di] = D[di][di] * rhovec[di-1]
              drho_temp = dt * dL * rhovec[di]

              for dj in range(1, di):
                  D[di-dj][di] = D[di-dj+1][di] * D[di-dj][di-dj]
                  drho_temp += dt * D[di-dj+1][di] * dL * rhovec[di-dj]
              drhovec[di] = drho_temp
          #Data saving:
          self.rho = rhovec
          self.rho_derivative = drhovec
          self.propagator = D

      def gradient_QFI(self):
          '''
          Output:
          1) updated values of self.control_coefficients
          Notice: To run this function, the function 'propagation_single()' has to run first.
          '''
          num = len(self.times)
          dH0 = 1.j * Qobj(liouvillian(self.Hamiltonian_derivative[0]).full())
          rhomat_final = Qobj(vec2mat(self.rho[num-1].full()))
          drhomat_final = Qobj(vec2mat(self.rho_derivative[num-1].full()))
          SLD_final = CR.SLD(rhomat_final, drhomat_final)

          dim = self.freeHamiltonian.dims[0][0]
          dt = self.times[1] - self.times[0]
          Hc_coeff = self.control_coefficients
          D = self.propagator

          for ki in range(0, len(self.control_Hamiltonian)):
              Hk = 1.j * Qobj(liouvillian(self.control_Hamiltonian[ki]).full())
              Hc_ki = Hc_coeff[ki]
              for ti in range(0, num):
                  Mj1 = 1.j * D[ti+1][num-1] * Hk * self.rho[ti]

                  Mj2 = Qobj(np.zeros((dim*dim, 1)))
                  for ri in range(0, ti+1):
                      Mj2 += D[ti+1][num-1] * Hk * D[ri+1][ti] * dH0 * self.rho[ri]

                  Mj3 = Qobj(np.zeros((dim*dim,1)))
                  for ri in range(ti+1, num):
                      Mj3 += D[ri+1][num-1] * dH0 * D[ti+1][ri] * Hk * self.rho[ti]
                  Mj1mat = Qobj(vec2mat(Mj1.full()))
                  Mj2mat = Qobj(vec2mat(Mj2.full()))
                  Mj3mat = Qobj(vec2mat(Mj3.full()))

                  term1 = dt * (SLD_final * SLD_final * Mj1mat).tr()
                  term2 = -2 * (dt * dt) * (SLD_final * (Mj2mat + Mj3mat)).tr()
                  delta = np.real(term1+term2)
                  Hc_ki[ti] += self.epsilon * delta
              Hc_coeff[ki] = Hc_ki

          self.control_coefficients = Hc_coeff

      def gradient_CFI(self, M):
          '''
          Input:
          1) M:
             TYPE: list of Qobj (matrix)
             DESCRIPTION: merasurement. It takes the form [M1,M2,...], where M1, M2 ...
             are matrices and satisfies M1+M2+... = identity matrix.
          Output:
          1) updated values of self.control_coefficients.
          Notice: To run this function, the function 'propagation_single()' has to run first.
          '''
          num = len(self.times)
          dH0 = 1.j * Qobj(liouvillian(self.Hamiltonian_derivative[0]).full())

          dim = self.freeHamiltonian.dims[0][0]
          dt = self.times[1] - self.times[0]
          Hc_coeff = self.control_coefficients
          D = self.propagator

          rhoT_vec = self.rho[num-1]
          drhoT_vec = self.rho_derivative[num-1]
          rhoT = Qobj(vec2mat(rhoT_vec.full()))
          drhoT = Qobj(vec2mat(drhoT_vec.full()))

          L1 = Qobj(np.array([[0. for i in range(0,dim)] for k in range(0,dim)]))
          L2 = Qobj(np.array([[0. for i in range(0,dim)] for k in range(0,dim)]))
          for mi in range(0, len(M)):
              ptemp = (rhoT * M[mi]).tr()
              dptemp = (drhoT * M[mi]).tr()
              if ptemp != 0:
                 L1 += (dptemp / ptemp) * M[mi]
                 L2 += ((dptemp / ptemp)**2) * M[mi]

          for ki in range(0, len(self.control_Hamiltonian)):
              Hk = 1.j * Qobj(liouvillian(self.control_Hamiltonian[ki]).full())
              Hc_ki = Hc_coeff[ki]
              for ti in range(0,num):
                  Mj1 = 1.j * D[ti+1][num-1] * Hk * self.rho[ti]

                  Mj2 = Qobj(np.zeros((dim*dim, 1)))
                  for ri in range(0, ti+1):
                      Mj2 += D[ti+1][num-1] * Hk * D[ri+1][ti] * dH0 * self.rho[ri]

                  Mj3 = Qobj(np.zeros((dim*dim, 1)))
                  for ri in range(ti+1, num):
                      Mj3 += D[ri+1][num-1] * dH0 * D[ti+1][ri] * Hk * self.rho[ti]
                  Mj1mat = Qobj(vec2mat(Mj1.full()))
                  Mj2mat = Qobj(vec2mat(Mj2.full()))
                  Mj3mat = Qobj(vec2mat(Mj3.full()))

                  term1 = dt * (L2 * Mj1mat).tr()
                  term2 = -2 * (dt * dt) * (L1 * (Mj2mat+Mj3mat)).tr()
                  delta = np.real(term1 + term2)

                  Hc_ki[ti] += Hc_ki[ti] + self.epsilon * delta

              Hc_coeff[ki] = Hc_ki

          self.control_coefficients = Hc_coeff

#=================================================================
#                     Multi-parameter scenario
#=================================================================
      def propagation_multiple(self):
          '''
          This function will save all the propators during the evolution, which may be memory consuming.
          '''
          if len(self.Hamiltonian_derivative) < 2:
              raise TypeError('this is a multiparameter scenario, the length of dH has to be larger than 1!')

          num = len(self.times)
          dim = self.freeHamiltonian.dims[0][0]
          dt = self.times[1] - self.times[0]
          dL = [Qobj((self.Hamiltonian_derivative[i]).full()) for i in range(0, len(self.Hamiltonian_derivative))]
          D = [[[] for i in range(0, num + 1)] for k in range(0, num + 1)]

          rhovec = [[] for i in range(0, num)]
          drhovec = [[[] for k in range(0, len(self.Hamiltonian_derivative))] for i in range(0, num)]
          rhovec[0] = Qobj(mat2vec(self.rho_initial.full()))

          for para_i in range(0, len(self.Hamiltonian_derivative)):
              drhovec[0][para_i] = dt * dL[para_i] * rhovec[0]

          D[0][0] = self.evolution(0)
          D[1][0] = qeye(dim ** 2)
          for di in range(1, num):
              tnow = self.times[0] + dt * di
              D[di+1][di] = qeye(dim ** 2)
              D[di][di] = self.evolution(tnow)
              D[0][di] = D[di][di] * D[0][di-1]
              rhovec[di] = D[di][di] * rhovec[di-1]

              for para_i in range(0, len(self.Hamiltonian_derivative)):
                  drho_temp = dt * dL[para_i] * rhovec[di]
                  for dj in range(1, di):
                      D[di - dj][di] = D[di-dj+1][di] * D[di-dj][di-dj]
                      drho_temp += dt * D[di-dj+1][di] * dL[para_i] * rhovec[di - dj]
                  drhovec[di][para_i] = drho_temp

          self.rho = rhovec
          self.rho_derivative = drhovec
          self.propagator = D

      def gradient_CFIM(self, M, obj_fun):
          '''
          Input:
          1) M:
             TYPE: list of Qobj (matrix)
             DESCRIPTION: measurement. It takes the form [M1,M2,...], where M1, M2 ...
                          are matrices and satisfies M1+M2+... = identity matrix.
          3) obj_fun
             TYPE: string
             DESCRIPTION: obj_fun = {'f0','f1','exact'}.
             Different with the single-parameter case, the GRAPE here uses two alternative objective functions:
             'f0': $f_{0}=\sum_{\alpha} 1/F_{\alpha\alpha}$.
             'f1': the lower bound $f_{1}=d^2/TrF$.
             Notice both above functions only use the gradient of diagonal entries of CFIM.
             'exact': exact gradient for TrF^{-1}, however, it is ONLY valid for two-parameter systems.
          Output:
          1) updated values of self.control_coefficients.
          Notice: To run this function, the function 'propagation_multiple()' has to be run first.
          '''
          if len(self.control_Hamiltonian) > 2 and obj_fun == 'exact':
              raise TypeError('the "exact" mode is only valid for two-parameter estimations!')

          num = len(self.times)
          dim = self.freeHamiltonian.dims[0][0]
          dt = self.times[1] - self.times[0]
          Hc_coeff = self.control_coefficients
          D = self.propagator

          rhoTvec = self.rho[num-1]
          rhoTmat = Qobj(vec2mat(rhoTvec.full()))
          drhoTvec = self.rho_derivative[num-1]
          drhoTmat = [Qobj(vec2mat(drhoTvec[i].full())) for i in range(0, len(self.Hamiltonian_derivative))]

          #Generation of L1 and L2 for diagonal entries of CFIM (i.e., alpha = beta):
          L1 = [Qobj(np.zeros((dim, dim))) for i in range(0, len(self.Hamiltonian_derivative))]
          L2 = [Qobj(np.zeros((dim, dim))) for i in range(0, len(self.Hamiltonian_derivative))]
          for para_i in range(0, len(self.Hamiltonian_derivative)):
              for mi in range(0, len(M)):
                  ptemp = (rhoTmat * M[mi]).tr()
                  dptemp = (drhoTmat[para_i] * M[mi]).tr()
                  if ptemp != 0:
                     L1[para_i] += (dptemp / ptemp) * M[mi]
                     L2[para_i] += ((dptemp / ptemp) ** 2) * M[mi]

          # Generation L2 for off-diagonal entries of CFIM in two-parameter estimation:
          if len(self.Hamiltonian_derivative) == 2:
             L2_offdiag = Qobj(np.zeros((dim, dim)))
             for mi in range(0, len(M)):
                 ptp_2para = (rhoTmat * M[mi]).tr()
                 dptp0_2para = (drhoTmat[0] * M[mi]).tr()
                 dptp1_2para = (drhoTmat[1] * M[mi]).tr()
                 L2_offdiag += (dptp0_2para * dptp1_2para / ptp_2para / ptp_2para) * M[mi]

          # Generation of CFIM at the target time
          CFIM_temp = CR.CFIM(rhoTmat, drhoTmat, M)
          norm_f0 = 0.
          for ci in range(0, len(self.Hamiltonian_derivative)):
              norm_f0 += 1 / CFIM_temp[ci, ci]
          norm_f0 = norm_f0**2
          M2_2para = [[] for i in range(0, 2)]
          M3_2para = [[] for i in range(0, 2)]

          #update
          for ki in range(0, len(self.control_Hamiltonian)):
              Hk = 1.j * Qobj(liouvillian(self.control_Hamiltonian[ki]).full())
              Hc_ki = Hc_coeff[ki]
              for ti in range(0, num):
                  Mj1 = 1.j * D[ti+1][num-1] * Hk * self.rho[ti]
                  Mj1mat = Qobj(vec2mat(Mj1.fulll()))
                  delta = 0.

                  for para_i in range(0, len(self.control_Hamiltonian)):
                      dH0_i = 1.j * Qobj(liouvillian(self.Hamiltonian_derivative[para_i]).full)
                      Mj2 = Qobj(np.zeros((dim*dim, 1)))
                      for ri in range(0,ti+1):
                          Mj2 += D[ti+1][num-1] * Hk * D[ri+1][ti] * dH0_i * self.rho[ri]

                      Mj3 = Qobj(np.zeros((dim*dim,1)))
                      for ri in range(ti+1, num):
                          Mj3 += D[ri+1][num-1] * dH0_i * D[ti+1][ri] * Hk * self.rho[ti]
                      Mj2mat = Qobj(vec2mat(Mj2.full()))
                      Mj3mat = Qobj(vec2mat(Mj3.full()))

                      term1 = dt * (L2[para_i] * Mj1mat).tr()
                      term2 = -2 * (dt ** 2) * (L1[para_i] * (Mj2mat + Mj3mat)).tr()
                      if obj_fun == 'f0':
                         delta += np.real(term1+term2)/((CFIM_temp[para_i, para_i])**2)/norm_f0
                      elif obj_fun == 'f1':
                         delta += np.real(term1+term2)/(float(len(self.control_Hamiltonian))**2)
                      elif obj_fun == 'exact':
                           delta += np.real(term1+term2)*(CFIM_temp[1-para_i, 1-para_i]**2 + CFIM_temp[0, 1]**2) \
                                    /(CFIM_temp.tr()**2)
                           M2_2para[para_i] = Mj2mat
                           M3_2para[para_i] = Mj3mat

                  if obj_fun == 'exact':
                     grad_offdiag = dt * (L2_offdiag * Mj1mat).tr() \
                                    - (dt**2) * (L1[1] * (M2_2para[0] + M3_2para[0])).tr() \
                                    - (dt**2) * (L1[0] * (M2_2para[1] + M3_2para[1])).tr()
                     delta += - np.real(2 * grad_offdiag * CFIM_temp[0, 1]/CFIM_temp.tr())

                  Hc_ki[ti] += Hc_ki[ti] + self.epsilon * delta
              Hc_coeff[ki] = Hc_ki
          self.control_coefficients = Hc_coeff


      def Run(self, statement, M=[], obj_fun=None):
          '''
          Input:
          1) statement:
             TYPE: string.
             DESCRIPTION: 'classical': classical parameter estimation and
                          'quantum': quantum parameter estimation
          '''
          num = len(self.times)
          if statement == 'classical':
             print('classical parameter estimation')
             if len(M) < 1:
                raise TypeError('measurement is required for CFI')

             if len(self.Hamiltonian_derivative) == 1:
                print('single parameter estimation scenario')
                self.propagation_single()
                rho0 = Qobj(vec2mat(self.rho[num - 1].full()))
                drho0 = Qobj(vec2mat(self.rho_derivative[num - 1].full()))
                cfi_ini = CR.CFI(rho0, drho0, M)
                while True:
                    self.gradient_CFI(M)
                    self.propagation_single()
                    rho1 = Qobj(vec2mat(self.rho[num - 1].full()))
                    drho1 = Qobj(vec2mat(self.rho_derivative[num - 1].full()))
                    cfi_now = CR.CFI(rho1, drho1, M)
                    if cfi_now > cfi_ini and cfi_now - cfi_ini < 1e-4:
                       print('Iteration over, data saved.')
                       print('Final CFI is ' + str(cfi_now))
                       qsave(self.control_coefficients, 'controls')
                       qsave(self.times, 'time_span')
                       break
                    else:
                       cfi_ini = cfi_now
                       print('current CFI is ' + str(cfi_now))

             elif len(self.Hamiltonian_derivative) > 1:
                 print('multiparameter estimation scenario')
                 if obj_fun is None:
                    print('objective function has to be clarified in multiparameter scenario')

                 self.propagation_multiple()
                 rho0 = Qobj(vec2mat(self.rho[num - 1].full()))
                 drho0 = Qobj(vec2mat(self.rho_derivative[num - 1].full()))
                 cfim_ini = CR.CFIM(rho0, drho0, M)
                 if obj_fun == 'f0':
                    obj_ini = 1.0 / np.trace(1.0 / cfim_ini.full())
                 elif obj_fun == 'f1':
                    obj_ini = cfim_ini.tr()/(len(self.Hamiltonian_derivative)**2)
                 elif obj_fun == 'exact':
                    obj_ini = np.linalg.det(cfim_ini.full())/cfim_ini.tr()

                 while True:
                    self.gradient_CFIM(M, obj_fun)
                    self.propagation_multiple()
                    rho1 = Qobj(vec2mat(self.rho[num - 1].full()))
                    drho1 = Qobj(vec2mat(self.rho_derivative[num - 1].full()))
                    cfim_now = CR.CFIM(rho1, drho1, M)
                    if obj_fun == 'f0':
                        obj_now = 1.0 / np.trace(1.0 / cfim_now.full())
                    elif obj_fun == 'f1':
                        obj_now = cfim_now.tr() / (len(self.Hamiltonian_derivative) ** 2)
                    elif obj_fun == 'exact':
                        obj_now = np.linalg.det(cfim_now.full()) / cfim_now.tr()

                    if obj_now > obj_ini and obj_now - obj_ini < 1e-4:
                       print('Iteration over, data saved.')
                       print('Final '+obj_fun+' value is ' + str(obj_now))
                       print('Final Tr(CFIM^{-1}) is'+str(np.linalg.det(cfim_now.full()) / cfim_now.tr()))
                       qsave(self.control_coefficients, 'controls')
                       qsave(self.times, 'time_span')
                       break
                    else:
                       obj_ini = obj_now
                       print('current ' + obj_fun + ' value is ' + str(obj_now))
                       print('current Tr(CFIM^{-1}) is'+str(np.linalg.det(cfim_now.full()) / cfim_now.tr()))

          elif statement == 'quantum':
             print('quantum parameter estimation')

             if len(self.Hamiltonian_derivative) == 1:
                print('single parameter estimation scenario')
                self.propagation_single()
                rho0 = Qobj(vec2mat(self.rho[num - 1].full()))
                drho0 = Qobj(vec2mat(self.rho_derivative[num - 1].full()))
                qfi_ini = CR.QFI(rho0, drho0)
                while True:
                    self.gradient_QFI()
                    self.propagation_single()
                    rho1 = Qobj(vec2mat(self.rho[num - 1].full()))
                    drho1 = Qobj(vec2mat(self.rho_derivative[num - 1].full()))
                    qfi_now = CR.QFI(rho1, drho1)
                    if qfi_now > qfi_ini and qfi_now - qfi_ini < 1e-4:
                       print('Iteration over, data saved.')
                       print('Final QFI is ' + str(qfi_now))
                       qsave(self.control_coefficients, 'controls')
                       qsave(self.times, 'time_span')
                       break
                    else:
                       qfi_ini = qfi_now
                       print('current QFI is ' + str(qfi_now))
             elif len(self.Hamiltonian_derivative) > 1:
                 print('using QFIM is not a good choice for multiparameter scenario')

          else:
             print('Please only use "classical" or "quantum" as the statement input')

if __name__ == '__main__':
   print(control_grape.__doc__)