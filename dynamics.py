# -*- coding: utf-8 -*-
__all__ = ['Lindblad_Dynamics','Cramer_Rao_bound']

import numpy as np
from numpy import linalg as nplin
from scipy import linalg as scylin

class Lindblad_Dynamics:
  '''
  general dynamics of density matrices in the form of time local Lindblad master equation
  '''

  def __init__(self, H, Lvec, gamma, rho_initial, delta_t,t_final,dH,Hc_control,\
               Hc_coeff,control_switch):
      self.freeHamiltonian = H
      self.Liouville_operator = Lvec
      self.gamma = gamma
      self.rho_initial = rho_initial
      self.delta_t = delta_t
      self.tfinal = t_final
      self.Hamiltonian_derivative = dH#for single-parameter estimation, it is a matrix. For multiparameter estimation, it is a list.
      self.control_Hamiltonian = Hc_control
      self.control_coefficients = Hc_coeff
      self.control_switch = control_switch

      self.dimension = len(self.freeHamiltonian)
      self.Liouvillenumber = len(self.Liouville_operator)
      self.num = int(self.tfinal/self.delta_t)
      self.tspan = np.linspace(self.delta_t,self.tfinal,self.num)
      self.ctrlnum = len(self.control_Hamiltonian)
      if self.control_switch == True:
          length1 = self.ctrlnum
          length2 = np.shape(self.control_coefficients)[0]
          if length1 > length2 :
             raise TypeError('there are not enough coefficients sequences: exit the program')
          elif length1 < length2:
             raise TypeError('there are too many coefficients sequences: exit the program')
      elif self.control_switch == False:
          pass
      else:
         raise TypeError('the control switch has to be set as logical True or False.')

      self.ctrlnum_total = self.ctrlnum
      self.control_coeff_total = self.control_coefficients
      self.rho = None
      self.rho_derivative = None
      self.propagator_save = None
      #self.ctrlH_Liou = None
      self.environment_assisted_order = None
      self.environmentstate = False

      if type(self.Hamiltonian_derivative) == list:
          self.freeHamiltonian_derivative_Liou = []
          for para_i in range(0,len(self.Hamiltonian_derivative)):
              dH0_temp = Lindblad_Dynamics.Liouville_commu(self,self.Hamiltonian_derivative[para_i])
              self.freeHamiltonian_derivative_Liou.append(dH0_temp)
      else:
          self.freeHamiltonian_derivative_Liou = self.Liouville_commu(self.Hamiltonian_derivative)

      #Generation of the Liouville representation of control Hamiltonians: self.ctrlH_Liou.
      Hk_Liou = [[] for i in range(0,self.ctrlnum)]
      for hi in range(0,self.ctrlnum):
          Htemp = self.Liouville_commu(self.control_Hamiltonian[hi])
          Hk_Liou[hi] = Htemp
      self.ctrlH_Liou = Hk_Liou

  def general_information(self):
      print('==================================')
      print('General information:')
      show_dimension = 'dimension of Hamiltonian: '+str(self.dimension)
      print(show_dimension)
      show_Liou = 'number of Liouville operators: '+str(self.Liouvillenumber)
      print(show_Liou)
      show_num = 'number of time step: '+str(self.num)
      print(show_num)
      show_ctrl = 'number of controls: '+str(self.ctrlnum_total)
      print(show_ctrl)
      show_cswitch = 'Control switch is '+str(self.control_switch)
      print(show_cswitch)
      print('==================================')

  def Liouville_commu(self,A):
      dim = len(A)
      result = [[0. for i in range(0,dim*dim)] for k in range(0,dim*dim)]
      for bi in range(0,dim):
          for bj in range(0,dim):
              for bk in range(0,dim):
                  ni = dim*bi+bj
                  nj = dim*bk+bj
                  nk = dim*bi+bk

                  result[ni][nj] = A[bi][bk]
                  result[ni][nk] = -A[bk][bj]
                  result[ni][ni] = A[bi][bi]-A[bj][bj]
      result = np.array(result)
      return result

  def Liouville_dissip(self,A):
      dim = len(A)
      result = [[0. for i in range(0,dim*dim)] for k in range(0,dim*dim)]
      for bi in range(0,dim):
          for bj in range(0,dim):
              ni = dim*bi+bj
              for bk in range(0,dim):
                  for bl in range(0,dim):
                      nj = dim*bk+bl
                      L_temp = A[bi][bk]*np.conj(A[bj][bl])
                      for bp in range(0,dim):
                          L_temp = L_temp-0.5*float(bk==bi)*A[bp][bj]*np.conj(A[bp][bl])\
                                   -0.5*float(bl==bj)*A[bp][bk]*np.conj(A[bp][bi])
                      result[ni][nj] = L_temp

      result = np.array(result)
      result[np.abs(result) < 1e-10] = 0.
      return result

#============================================

  def free_evolution(self):
      H = self.freeHamiltonian
      result = -1.j*self.Liouville_commu(H)
      return result

  def Dissipation(self,tj):
      '''
      The Dissipation part of Lindblad form master equation:
      La.rho.La^{\dagger}-0.5(rho.La^{\dagger}.La+La^{\dagger}.La.rho).
      \gamma can either be a scale parameter (i.e., time-independent) or a vector
      (time-dependent).
      '''
      dim = self.dimension
      Loper = self.Liouville_operator
      Lnum = self.Liouvillenumber

      ga = [0. for i in range(0,Lnum)]
      for gi in range(0,Lnum):
          gtest = self.gamma[gi]
          if type(gtest) == float:
             ga[gi] = gtest
          elif type(gtest) != float:
             ga[gi] = gtest[tj]

      result = [[0. for i in range(0,dim*dim)] for k in range(0,dim*dim)]
      for bi in range(0,dim):
          for bj in range(0,dim):
              ni = dim*bi+bj
              for bk in range(0,dim):
                  for bl in range(0,dim):
                      nj = dim*bk+bl
                      L_temp = 0.
                      for Ln in range(0,Lnum):
                          Lc = Loper[Ln]
                          L_temp = L_temp + ga[Ln]*Lc[bi][bk]*np.conj(Lc[bj][bl])
                          for bp in range(0,dim):
                              L_temp = L_temp-0.5*ga[Ln]*float(bk==bi)*Lc[bp][bj]*np.conj(Lc[bp][bl])\
                                       -0.5*ga[Ln]*float(bl==bj)*Lc[bp][bk]*np.conj(Lc[bp][bi])
                      result[ni][nj] = L_temp

      result = np.array(result)
      result[np.abs(result) < 1e-10] = 0.
      return result

  def evolution(self,tj):
      if self.control_switch == False:
         Ld = self.free_evolution()+self.Dissipation(tj)
         result = scylin.expm(self.delta_t*Ld)

      elif self.control_switch == True:
         Hc = self.control_Hamiltonian
         Hc_coeff = self.control_coefficients
         if type(tj) != int:
            raise TypeError('input variable has to be the number of time point')
         else:
            Htot = self.freeHamiltonian
            for hn in range(0,self.ctrlnum):
                Hc_temp = None
                Hc_temp = Hc_coeff[hn]
                Htot = Htot+Hc[hn]*Hc_temp[tj]
            freepart = self.Liouville_commu(Htot)
            Ld = -1.j*freepart+self.Dissipation(tj)
            result = scylin.expm(self.delta_t*Ld)

      return result

  def propagator(self,tstart,tend):
      if type(tstart) != int and type(tend) != int:
         raise TypeError('inputs are the number of time interval')
      else:
         if tstart > tend:
            result = np.eye(self.dimension*self.dimension)
         elif tstart == tend:
             result = self.evolution(tstart)
         else:
             result = self.evolution(tstart)
             for pi in range(tstart+1,tend-tstart):
                 Ltemp = None
                 Ltemp = self.evolution(pi)
                 result = np.dot(Ltemp,result)
         return result

  def evolved_state(self,tj,state):
      dim = self.dimension
      rho_temp = np.reshape(self.rho_initial,(dim*dim,1))
      propa = self.propagator(0,tj)
      rho_tvec = np.dot(propa,rho_temp)
      if state == 'matrix':
         rho_t = np.reshape(rho_tvec,(dim,dim))
      elif state == 'vector':
         rho_t = rho_tvec
      return rho_t

#===============================================================================
#Subclass: metrology
#===============================================================================
class Cramer_Rao_bound(Lindblad_Dynamics):
  '''
  Subclass: calculation of quantum Fisher information and control gradient.
  '''

  def CFI(self, rho_input, drho_input, M):
      '''
      Input: 1) rho: density matrix
             2) drho: derivative of density matrix on parameter.
             3) M: measurement. It takes the form [M1,M2,...], where M1, M2 ...
                are matrices and satisfies M1+M2+... = identity matrix.
      Notice: here we assume the measurement M is independent on the parameter
              under estimation.
      '''
      dim = self.dimension
      if len(rho_input) == dim*dim:
         rho = np.reshape(rho_input,(dim,dim))
         drho = np.reshape(drho_input,(dim,dim))
      else:
         rho = rho_input
         drho = drho_input

      m_num = len(M)
      p = [0. for i in range(0,m_num)]
      dp = [0. for i in range(0,m_num)]
      CF = 0.
      for pi in range(0,m_num):
          Mp = M[pi]
          p[pi] = np.trace(np.dot(rho,Mp))
          dp[pi] = np.trace(np.dot(drho,Mp))
          Cadd = 0
          if p[pi] != 0:
             Cadd = (dp[pi]**2)/(p[pi])
          CF = CF+Cadd
      return (CF, p, dp)

  def CFIM(self,rho_input,drho_input,M):
      '''
      Input: 1) rho: density matrix
             2) drho: list of derivative of density matrix on all parameters.
             3) M: measurement. It takes the form [M1,M2,...], where M1, M2 ...
                are matrices and satisfies M1+M2+... = identity matrix.
      Notice: 1) here we assume the measurement M is independent on the parameter
                 under estimation.
              2) drho here contains the derivative on all parameters, for example,
                 drho[0] is the derivative vector on the first parameter.

      '''
      if type(drho_input) != list:
         raise TypeError('Please make sure drho is a list!')

      dim = self.dimension
      drho = [[] for i in range(0,len(drho_input))]
      if len(rho_input) == dim*dim:
         rho = np.reshape(rho_input,(dim,dim))
         for para_i in range(0,len(drho_input)):
             drho[para_i] = np.reshape(drho_input[para_i],(dim,dim))
      else:
         rho = rho_input
         drho = drho_input

      m_num = len(M)
      para_num = len(drho)
      CFIM_res = [[0. for i in range(0,para_num)] for k in range(0,para_num)]
      CFIM_res = np.array(CFIM_res)

      #----------------
      #for test
      #----------------
      #p_test = [0. for i in range(0,m_num)]
      #dp_test = [[] for i in range(0,m_num)]
      #----------------

      for pi in range(0,m_num):
          Mp = M[pi]
          p = np.trace(np.dot(rho,Mp))
          #---------------
          #p_test[pi] = p#for test
          #---------------
          Cadd = [[0. for i in range(0,para_num)] for k in range(0,para_num)]
          dptemp = [0. for i in range(0,para_num)]
          if p != 0:
             for para_i in range(0,para_num):
                 drho_i = drho[para_i]
                 dp_i = np.trace(np.dot(drho_i,Mp))
                 #--------------
                 #dptemp[para_i] = dp_i#for test
                 #--------------
                 for para_j in range(para_i,para_num):
                     drho_j = drho[para_j]
                     dp_j = np.trace(np.dot(drho_j,Mp))
                     CFIM_temp = np.real(dp_i*dp_j/p)
                     Cadd[para_i][para_j] = CFIM_temp
                     Cadd[para_j][para_i] = CFIM_temp
          #dp_test[pi] = dptemp#for test
          Cadd = np.array(Cadd)
          CFIM_res = CFIM_res+Cadd
      return CFIM_res
      #return (CFIM, p_test, dp_test)#for test

  def SLD(self,rho_input,drho_input):
      dim = self.dimension
      drho = [[] for i in range(0,len(drho_input))]
      #---------------------
      #for multi-parameter:
      #---------------------
      if type(drho_input) == list:
         if len(rho_input) == dim*dim:
            rho = np.reshape(rho_input,(dim,dim))
            for para_i in range(0,len(drho_input)):
                drho[para_i] = np.reshape(drho_input[para_i],(dim,dim))
         else:
            rho = rho_input
            drho = drho_input

         SLD_res = [[] for i in range(0,len(drho))]
         purity = np.trace(np.dot(rho,rho))
         if np.abs(1-purity) < 1e-8:
            for para_i in range(0,len(drho)):
                SLD_res[para_i] = 2*drho[para_i]
         else:
            val, vec = nplin.eigh(rho)
            for para_i in range(0,len(drho)):
                SLD_tp = [[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)]
                SLD_tp = np.array(SLD_tp)
                for fi in range (0,dim):
                    for fj in range (0,dim):
                        coeff = 2/(val[fi]+val[fj])
                        vectemp = vec[:,fi].conj().transpose()
                        SLD_tp[fi][fj] = coeff*np.dot(vectemp,np.dot(drho[para_i],vec[:,fj]))
                SLD_tp[SLD_tp == np.inf] = 0.
                vec_dagger = vec.conj().transpose()
                SLD_original_basis = np.dot(vec,np.dot(SLD_tp,vec_dagger))
                SLD_res[para_i] = SLD_original_basis
      #---------------------
      #for single-parameter:
      #---------------------
      else:
         if len(rho_input) == dim*dim:
            rho = np.reshape(rho_input,(dim,dim))
            drho = np.reshape(drho_input,(dim,dim))
         else:
            rho = rho_input
            drho = drho_input
         
         purity = np.trace(np.dot(rho,rho))
         if np.abs(1-purity) < 1e-8:
            SLD_res = 2*drho
         else:
            val, vec = nplin.eigh(rho)
            SLD_res = [[0.+0.*1.j for i in range(0,dim)] for i in range(0,dim)]
            for fi in range (0,dim):
                for fj in range (0,dim):
                    coeff = 2/(val[fi]+val[fj])
                    vectemp = vec[:,fi].conj().transpose()
                    SLD_res[fi][fj] = coeff*np.dot(vectemp,np.dot(drho,vec[:,fj]))
            SLD_res = np.array(SLD_res)
            SLD_res[SLD_res == np.inf] = 0.
            vec_dagger = vec.conj().transpose()
            SLD_res = np.dot(vec,np.dot(SLD_res,vec_dagger))
      return SLD_res

  def QFI(self,rho_input,drho_input):
      '''
      Input: 1) rho_input: density matrix
             2) drho_input: derivative of density matrix on parameter.
      Notice: the output SLD is in the normal space, not the eigenspace of rho.
      '''
      dim = self.dimension
      if len(rho_input) == dim*dim:
         rho = np.reshape(rho_input,(dim,dim))
         drho = np.reshape(drho_input,(dim,dim))
      else:
          rho = rho_input
          drho = drho_input

      SLD_tp = self.SLD(rho_input,drho_input)
      SLD2 = np.dot(SLD_tp,SLD_tp)
      F = np.trace(np.dot(rho,SLD2))
      return F

  def QFIM(self,rho_input,drho_input):
      '''
          Input:
          1) rho_input: density matrix
          2) drho_input: list of derivative of density matrix on all parameters.
          Notice: drho here contains the derivative on all parameters, for example,
          drho[0] is the derivative vector on the first parameter.
      '''
      if type(drho_input) != list:
         raise TypeError('Multiple derivatives of density matrix are required for QFIM')
      else:
         QFIM_res = [[0. for i in range(0,len(drho_input))] for i in range(0,len(drho_input))]
         QFIM_res = np.array(QFIM_res)

      dim = self.dimension
      if len(rho_input) == dim*dim:
         rho = np.reshape(rho_input,(dim,dim))
      else:
         rho = rho_input

      SLD_tp = self.SLD(rho_input,drho_input)
      for para_i in range(0,len(drho_input)):
          for para_j in range(para_i,len(drho_input)):
              SLD_anticommu = np.dot(SLD_tp[para_i],SLD_tp[para_j])+np.dot(SLD_tp[para_j],SLD_tp[para_i])
              QFIM_res[para_i][para_j] = np.real(0.5*np.trace(np.dot(rho,SLD_anticommu)))
              QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]

      return QFIM_res

  def Holevo_bound_trace(self,rho_input,drho_input,cost_function):
      '''
         Input:
         1) rho_input: density matrix
         2) drho_input: list of derivative of density matrix on all parameters.
         Notice: drho here contains the derivative on all parameters, for example,
         drho[0] is the derivative vector on the first parameter.
         3) cost_function: cost_function in the Holevo bound.
      '''
      if type(drho_input) != list:
         raise TypeError('Multiple derivatives of density matrix are required for Holevo bound')
      else: pass

      dim = self.dimension
      para_dim = len(drho_input)
      CostG = cost_function
      if len(rho_input) == dim*dim:
         rho_matrix = np.reshape(rho_input,(dim,dim))
      else:
         rho_matrix = rho_input

      QFIM_temp = self.QFIM(rho_input,drho_input)
      QFIMinv = nplin.inv(QFIM_temp)
      SLD_temp = self.SLD(rho_input,drho_input)

      V = np.array([[0.+0.j for i in range(0,para_dim)] for k in range(0,para_dim)])
      for para_i in range(0,para_dim):
          for para_j in range(0,para_dim):
              Vij_temp = 0.+0.j
              for ki in range(0,para_dim):
                  for mi in range(0,para_dim):
                      SLD_ki = SLD_temp[ki]
                      SLD_mi = SLD_temp[mi]
                      Vij_temp = Vij_temp+QFIMinv[para_i][ki]*QFIMinv[para_j][mi]\
                                 *np.trace(np.dot(np.dot(rho_matrix,SLD_ki),SLD_mi))
              V[para_i][para_j] = Vij_temp

      real_part = np.dot(CostG,np.real(V))
      imag_part = np.dot(CostG,np.imag(V))
      Holevo_trace = np.trace(real_part)+np.trace(scylin.sqrtm(np.dot(imag_part,np.conj(np.transpose(imag_part)))))

      return Holevo_trace

  def data_generation(self,state):
      '''
      This function will save all the propators during the evolution, which may be memory consuming.
      Input: state: statement of output. If it is 'vector', the output state is in vector form, and if
                    it is 'matrix', the output state is in matrix form.
      '''
      num = self.num
      dim = self.dimension
      dH = self.Hamiltonian_derivative
      dL = -1.j*Lindblad_Dynamics.Liouville_commu(self,dH)
      dt = self.delta_t
      D = [[[] for i in range(0,num+1)] for i in range(0,num+1)]
      rhovec = [[] for i in range(0,num)]
      drhovec = [[] for i in range(0,num)]
      rhomat = [[] for i in range(0,num)]
      drhomat = [[] for i in range(0,num)]

      rhovec[0] = Lindblad_Dynamics.evolved_state(self,0,'vector')
      drhovec[0] = dt*np.dot(dL,rhovec[0])
      rhomat[0] = Lindblad_Dynamics.evolved_state(self,0,'matrix')
      drhomat[0] = np.reshape(dt*np.dot(dL,rhovec[0]),(dim,dim))
      D[0][0] = Lindblad_Dynamics.evolution(self,0)
      D[1][0] = np.eye(dim*dim)

      for di in range(1,num):
          D[di+1][di] = np.eye(dim*dim)
          D[di][di] = Lindblad_Dynamics.evolution(self,di)
          D[0][di] = np.dot(D[di][di],D[0][di-1])
          rhovec[di] = np.array(np.dot(D[di][di],rhovec[di-1]))
          rhomat[di] = np.reshape(rhovec[di],(dim,dim))

          drho_temp = dt*np.dot(dL,rhovec[di])
          for dj in range(1,di):
              D[di-dj][di] = np.dot(D[di-dj+1][di],D[di-dj][di-dj])
              drho_temp = drho_temp+dt*np.dot(D[di-dj+1][di],np.dot(dL,rhovec[di-dj]))
          drhovec[di] = np.array(drho_temp)
          drhomat[di] = np.reshape(np.array(drho_temp),(dim,dim))
      #Data saving:
      if state == 'vector':
         self.rho = rhovec
         self.rho_derivative = drhovec
         self.propagator_save = D
         self.state = 'vector'
      elif state == 'matrix':
          self.rho = rhomat
          self.rho_derivative = drhomat
          self.propagator_save = D
          self.state = 'matrix'
      print('Data saved.')

  def data_generation_multiparameter(self,state):
      '''
      This function will save all the propators during the evolution, which may be memory consuming.
      Input: state: statement of output. If it is 'vector', the output state is in vector form, and if
                    it is 'matrix', the output state is in matrix form.
      Notice: the difference between this function and 'data_generation' is that this function generates
              all the derivatives of rho on all parameters to be estimated.
      '''
      num = self.num
      dim = self.dimension
      dH = self.Hamiltonian_derivative#Here self.Hamiltonian_derivative should be a list
      para_len = len(dH)
      dL = [[] for i in range(0,para_len)]
      for para_i in range(0,para_len):
          dL_temp = -1.j*Lindblad_Dynamics.Liouville_commu(self,dH[para_i])
          dL[para_i] = dL_temp
      dt = self.delta_t
      D = [[[] for i in range(0,num+1)] for i in range(0,num+1)]
      rhovec = [[] for i in range(0,num)]
      drhovec = [[[] for k in range(0,para_len)] for i in range(0,num)]
      rhomat = [[] for i in range(0,num)]
      drhomat = [[[] for k in range(0,para_len)] for i in range(0,num)]

      rhovec[0] = Lindblad_Dynamics.evolved_state(self,0,'vector')
      rhomat[0] = Lindblad_Dynamics.evolved_state(self,0,'matrix')
      for para_i in range(0,para_len):
          drhovec_temp = dt*np.dot(dL[para_i],rhovec[0])
          drhovec[0][para_i] = drhovec_temp
          drhomat[0][para_i] = np.reshape(drhovec_temp,(dim,dim))
      D[0][0] = Lindblad_Dynamics.evolution(self,0)
      D[1][0] = np.eye(dim*dim)

      for di in range(1,num):
          D[di+1][di] = np.eye(dim*dim)
          D[di][di] = Lindblad_Dynamics.evolution(self,di)
          D[0][di] = np.dot(D[di][di],D[0][di-1])
          rhovec[di] = np.array(np.dot(D[di][di],rhovec[di-1]))
          rhomat[di] = np.reshape(rhovec[di],(dim,dim))

          for para_i in range(0,para_len):
              drho_temp = dt*np.dot(dL[para_i],rhovec[di])
              for dj in range(1,di):
                  D[di-dj][di] = np.dot(D[di-dj+1][di],D[di-dj][di-dj])
                  drho_temp = drho_temp+dt*np.dot(D[di-dj+1][di],np.dot(dL[para_i],rhovec[di-dj]))
              drhovec[di][para_i] = np.array(drho_temp)
              drhomat[di][para_i] = np.reshape(np.array(drho_temp),(dim,dim))
      #Data saving:
      if state == 'vector':
         self.rho = rhovec
         self.rho_derivative = drhovec
         self.propagator_save = D
         self.state = 'vector'
      elif state == 'matrix':
           self.rho = rhomat
           self.rho_derivative = drhomat
           self.propagator_save = D
           self.state = 'matrix'
      print('Data saved.')

  def environment_assisted_state(self,statement,Dissipation_order):
      '''
      If the dissipation coefficient can be manually manipulated, it can be updated via GRAPE.
      This function is used to clarify which dissipation parameter can be updated.
      Input: 1) statement: True: the dissipation parameter is involved in the GRAPE.
                           False: the dissipation parameter is not involved in the GRAPE.
             2) Dissipation_order: number list contains the number of dissipation parameter to be updated.
                                   For example, [3] means the 3rd Liouville operator can be updated and
                                   [3, 5] means the 3rd and 5th Liouville operators can be updated.
      '''
      if  statement == True:
          newnum = int(self.ctrlnum+len(Dissipation_order))
          Hk_Liou = [[] for i in range(0,newnum)]
          for hi in range(0,self.ctrlnum):
              Htemp = Lindblad_Dynamics.Liouville_commu(self,Hc[hi])
              Hk_Liou[hi] = Htemp
          for hi in range(0,len(Dissipation_order)):
              hj = int(self.ctrlnum+hi)
              hnum = Dissipation_order[hi]
              Htemp = 1.j*Lindblad_Dynamics.Liouville_dissip(self,self.Liouville_operator[hnum])
              Hk_Liou[hj] = Htemp
              ga = self.gamma[hnum]
              ctrl_coeff = self.control_coeff_total
              ctrl_coeff.append(ga)
              self.control_coeff_total = ctrl_coeff
          self.ctrlnum_total = newnum
          self.ctrlH_Liou = Hk_Liou
          self.environment_assisted_order = Dissipation_order
          self.environmentstate = statement

  #=================================================================
  #                           GRAPE
  #=================================================================
  def GRAPE_QFI(self,epsilon):
      '''
      Input: 1) epsilon: step to update the control coefficients.
             2) environment_assisted: True: environment parameter is involved in GRAPE
                                      False: environment parameter is not involved in GRAPE.
      Output: updated values of self.control_coefficients.
      Notice: To run this funtion, the function 'Data_generation('vector')' and 'environment_assisted_state()' has to run first.
      '''
      if self.state == 'matrix':
         raise TypeError('Please change the input string in Data_generation as "vector".')
      else:
         num = self.num
         rho = self.rho
         dH0 = self.freeHamiltonian_derivative_Liou#Lindblad_Dynamics.Liouville_commu(self,self.Hamiltonian_derivative)
         Ffinal, SLD_final = self.QFI(rho[num-1],self.rho_derivative[num-1])
         dim = self.dimension
         dt = self.delta_t
         Hc_coeff = self.control_coeff_total
         #Hc = self.control_Hamiltonian
         D = self.propagator_save

         for ti in range(0,num):
             for ki in range(0,self.ctrlnum_total):
                 Hk = self.ctrlH_Liou[ki]
                 Mj1 = 1.j*np.dot(D[ti+1][num-1],np.dot(Hk,rho[ti]))
                 #-------
                 Mj2 = np.zeros((dim*dim,1))
                 for ri in range(0,ti+1):
                     Mj2_temp = np.dot(D[ri+1][ti],np.dot(dH0,rho[ri]))
                     Mj2_temp = np.dot(D[ti+1][num-1],np.dot(Hk,Mj2_temp))
                     Mj2 = Mj2+Mj2_temp
                 #-------
                 Mj3 = np.zeros((dim*dim,1))
                 for ri in range(ti+1,num):
                     Mj3_temp = np.dot(D[ti+1][ri],np.dot(Hk,rho[ti]))
                     Mj3_temp = np.dot(D[ri+1][num-1],np.dot(dH0,Mj3_temp))
                     Mj3 = Mj3+Mj3_temp
                 Mj1mat = np.reshape(Mj1,(dim,dim))
                 Mj2mat = np.reshape(Mj2,(dim,dim))
                 Mj3mat = np.reshape(Mj3,(dim,dim))

                 SLD2 = np.dot(SLD_final,SLD_final)
                 term1 = dt*np.trace(np.dot(SLD2,Mj1mat))
                 term2 = -2*(dt**2)*np.trace(np.dot(SLD_final,Mj2mat+Mj3mat))
                 delta = np.real(term1+term2)
                 #---------------------------------
                 #update the control coefficients:
                 #---------------------------------
                 Hc_kiti = Hc_coeff[ki]
                 Hc_kiti[ti] = Hc_kiti[ti]+epsilon*delta
                 Hc_coeff[ki] = Hc_kiti
         if self.environmentstate == False:
            self.control_coeff_total = Hc_coeff
            self.control_coefficients = self.control_coeff_total
         elif  self.environmentstate == True:
             self.control_coefficients = Hc_coeff[0:self.ctrlnum]
             for ei in range(0,len(self.environment_assisted_order)):
                 gam_num = self.environment_assisted_order[ei]
                 self.gamma[gam_num] = Hc_coeff[self.ctrlnum+ei]

  def GRAPE_CFI(self, epsilon, M):
      '''
      Input: 1) epsilon: step to update the control coefficients.
             2) M: merasurement. It takes the form [M1,M2,...], where M1, M2 ...
                   are matrices and satisfies M1+M2+... = identity matrix.
      Output: updated values of self.control_coefficients.
      Notice: To run this funtion, the function 'Data_generation('vector')' has to be run first.
      '''
      if self.state == 'matrix':
         raise TypeError('Please change the input string in Data_generation as "vector".')
      else:
         num = self.num
         rho = self.rho
         dH0 = self.freeHamiltonian_derivative_Liou#Lindblad_Dynamics.Liouville_commu(self,self.Hamiltonian_derivative)
         dim = self.dimension
         dt = self.delta_t
         Hc_coeff = self.control_coeff_total
         Hc = self.control_Hamiltonian
         D = self.propagator_save

         Mnum = len(M)
         rhoT_vec = rho[num-1]
         drhoT_vec = self.rho_derivative[num-1]
         rhoT = np.reshape(rhoT_vec,(dim,dim))
         drhoT = np.reshape(drhoT_vec,(dim,dim))
         L1 = [[0. for i in range(0,dim)] for i in range(0,dim)]
         L2 = [[0. for i in range(0,dim)] for i in range(0,dim)]
         L1 = np.array(L1)
         L2 = np.array(L2)
         for mi in range(0,Mnum):
             ptemp = np.trace(np.dot(rhoT,M[mi]))
             dptemp = np.trace(np.dot(drhoT,M[mi]))
             if ptemp != 0:
                L1 = L1+(dptemp/ptemp)*M[mi]
                L2 = L2+((dptemp/ptemp)**2)*M[mi]
             elif ptemp == 0:
                L1 = L1
                L2 = L2

         for ti in range(0,num):
             for ki in range(0,self.ctrlnum_total):
                 Hk = self.ctrlH_Liou[ki]
                 #Hk = Lindblad_Dynamics.Liouville_commu(self,Hc[ki])
                 Mj1 = 1.j*np.dot(D[ti+1][num-1],np.dot(Hk,rho[ti]))
                 #-------
                 Mj2 = np.zeros((dim*dim,1))
                 for ri in range(0,ti+1):
                     Mj2_temp = np.dot(D[ri+1][ti],np.dot(dH0,rho[ri]))
                     Mj2_temp = np.dot(D[ti+1][num-1],np.dot(Hk,Mj2_temp))
                     Mj2 = Mj2+Mj2_temp
                 #-------
                 Mj3 = np.zeros((dim*dim,1))
                 for ri in range(ti+1,num):
                     Mj3_temp = np.dot(D[ti+1][ri],np.dot(Hk,rho[ti]))
                     Mj3_temp = np.dot(D[ri+1][num-1],np.dot(dH0,Mj3_temp))
                     Mj3 = Mj3+Mj3_temp
                 Mj1mat = np.reshape(Mj1,(dim,dim))
                 Mj2mat = np.reshape(Mj2,(dim,dim))
                 Mj3mat = np.reshape(Mj3,(dim,dim))

                 term1 = dt*np.trace(np.dot(L2,Mj1mat))
                 term2 = -2*(dt**2)*np.trace(np.dot(L1,Mj2mat+Mj3mat))
                 delta = np.real(term1+term2)
                 #---------------------------------
                 #update the control coefficients:
                 #---------------------------------
                 Hc_kiti = Hc_coeff[ki]
                 Hc_kiti[ti] = Hc_kiti[ti]+epsilon*delta
                 Hc_coeff[ki] = Hc_kiti
         if  self.environmentstate == False:
             self.control_coeff_total = Hc_coeff
             self.control_coefficients = self.control_coeff_total
         elif  self.environmentstate == True:
             self.control_coefficients = Hc_coeff[0:self.ctrlnum]
             for ei in range(0,len(self.environment_assisted_order)):
                 gam_num = self.environment_assisted_order[ei]
                 self.gamma[gam_num] = Hc_coeff[self.ctrlnum+ei]


  def GRAPE_CFIM_diagonal(self,epsilon,M,obj_fun):
      '''
      Input: 1) epsilon: step to update the control coefficients.
             2) M: merasurement. It takes the form [M1,M2,...], where M1, M2 ...
                are matrices and satisfies M1+M2+... = identity matrix.
      Output: updated values of self.control_coefficients.
      Notice: 1) To run this funtion, the function 'Data_generation_multiparameter('vector')' has to be run first.
              2) maximize is always more accurate than the minimize in this code.
              3) obj_fun = {'f0','f1','exact'}.
                 Different with the single-parameter case, the GRAPE here uses two alternative objective functions:
                 'f0': $f_{0}=\sum_{\alpha} 1/F_{\alpha\alpha}$.
                 'f1': the lower bound $f_{1}=d^2/TrF$.
                 Notice both above functions only use the gradient of diagonal entries of CFIM.
                 'exact': exact gradient for TrF^{-1}, however, it is ONLY valid for two-parameter systems.
      '''
      if self.state == 'matrix':
         raise TypeError('Please change the input string in Data_generation as "vector".')
      else:
          num = self.num
          rho = self.rho
          drho = self.rho_derivative
          paralen = len(self.Hamiltonian_derivative)
          dH0 = self.freeHamiltonian_derivative_Liou
          dim = self.dimension
          dt = self.delta_t
          Hc_coeff = self.control_coeff_total
          Hc = self.control_Hamiltonian
          D = self.propagator_save

          Mnum = len(M)
          rhoT_vec = rho[num-1]
          rhoT = np.reshape(rhoT_vec,(dim,dim))
          drhoT_vec = drho[num-1]
          drhoT = [[] for i in range(0,paralen)]
          for para_i in range(0,paralen):
              drhoT[para_i] = np.reshape(drhoT_vec[para_i],(dim,dim))

          #Generation of L1 and L2 for diagonal entries of CFIM (i.e., alpha = beta):
          L1 = [[] for i in range(0,paralen)]
          L2 = [[] for i in range(0,paralen)]
          for para_i in range(0,paralen):
              L1[para_i] = np.zeros((dim,dim))
              L2[para_i] = np.zeros((dim,dim))
          for para_i in range(0,paralen):
              for mi in range(0,Mnum):
                  ptemp = np.trace(np.dot(rhoT,M[mi]))
                  dptemp = np.trace(np.dot(drhoT[para_i],M[mi]))
                  L1[para_i] = L1[para_i]+(dptemp/ptemp)*M[mi]
                  L2[para_i] = L2[para_i]+((dptemp/ptemp)**2)*M[mi]
          #Generation L2 for off-diagonal entries of CFIM in two-parameter estimation:
          if paralen == 2:
             L2_offdiag = np.zeros((dim,dim))
             for mi in range(0,Mnum):
                 ptp_2para = np.trace(np.dot(rhoT,M[mi]))
                 dptp0_2para = np.trace(np.dot(drhoT[0],M[mi]))
                 dptp1_2para = np.trace(np.dot(drhoT[1],M[mi]))
                 L2_offdiag = L2_offdiag+(dptp0_2para*dptp1_2para/ptp_2para/ptp_2para)*M[mi]


          #----------------------------------------
          # Generation of CFIM at the target time
          #----------------------------------------
          CFIM_temp = self.CFIM(rhoT,drhoT,M)
          norm_f0 = 0.
          for ci in range(0,paralen):
              norm_f0 = norm_f0+1/CFIM_temp[ci][ci]
          norm_f0 = norm_f0**2
          M2_2para = [[] for i in range(0,2)]
          M3_2para = [[] for i in range(0,2)]

          for ti in range(0,num):
              #------------------------
              #calculation of gradient
              #------------------------
              for ki in range(0,self.ctrlnum_total):
                  Hk = self.ctrlH_Liou[ki]
                  Mj1 = 1.j*np.dot(D[ti+1][num-1],np.dot(Hk,rho[ti]))
                  Mj1mat = np.reshape(Mj1,(dim,dim))
                  delta = 0.

                  for para_i in range(0,paralen):
                      #-------
                      Mj2 = np.zeros((dim*dim,1))
                      for ri in range(0,ti+1):
                          Mj2_temp = np.dot(D[ri+1][ti],np.dot(dH0[para_i],rho[ri]))
                          Mj2_temp = np.dot(D[ti+1][num-1],np.dot(Hk,Mj2_temp))
                          Mj2 = Mj2+Mj2_temp
                      #-------
                      Mj3 = np.zeros((dim*dim,1))
                      for ri in range(ti+1,num):
                          Mj3_temp = np.dot(D[ti+1][ri],np.dot(Hk,rho[ti]))
                          Mj3_temp = np.dot(D[ri+1][num-1],np.dot(dH0[para_i],Mj3_temp))
                          Mj3 = Mj3+Mj3_temp

                      Mj2mat = np.reshape(Mj2,(dim,dim))
                      Mj3mat = np.reshape(Mj3,(dim,dim))

                      term1 = dt*np.trace(np.dot(L2[para_i],Mj1mat))
                      term2 = -2*(dt**2)*np.trace(np.dot(L1[para_i],Mj2mat+Mj3mat))
                      if obj_fun == 'f0':
                         delta = delta+np.real(term1+term2)/((CFIM_temp[para_i][para_i])**2)/norm_f0
                      elif obj_fun == 'f1':
                         delta = delta+np.real(term1+term2)/float(paralen)/float(paralen)
                      elif obj_fun == 'exact':
                           if paralen > 2:
                              raise TypeError('the "exact" mode is only valid for two-parameter systems, the current\
                                              parameter number is '+str(paralen))
                           elif paralen == 2:
                                delta = delta+np.real(term1+term2)*((CFIM_temp[1-para_i][1-para_i])**2+(CFIM_temp[0][1])**2)\
                                              /((np.trace(CFIM_temp))**2)
                                M2_2para[para_i] = Mj2mat
                                M3_2para[para_i] = Mj3mat

                  if obj_fun == 'exact':
                     gradient_offdiag = dt*np.trace(np.dot(L2_offdiag, Mj1mat))-(dt**2)*np.trace(np.dot(L1[1], M2_2para[0]+M3_2para[0]))\
                                        -(dt**2)*np.trace(np.dot(L1[0], M2_2para[1]+M3_2para[1]))
                     delta = delta-np.real(2*gradient_offdiag*CFIM_temp[0][1]/np.trace(CFIM_temp))

                  #---------------------------------
                  #update the control coefficients:
                  #---------------------------------
                  Hc_kiti = Hc_coeff[ki]
                  #print(Hc_kiti[ti])
                  #print(delta)
                  Hc_kiti[ti] = Hc_kiti[ti]+epsilon*delta
                  Hc_coeff[ki] = Hc_kiti
          if  self.environmentstate == False:
              self.control_coeff_total = Hc_coeff
              self.control_coefficients = self.control_coeff_total
          elif  self.environmentstate == True:
                self.control_coefficients = Hc_coeff[0:self.ctrlnum]
                for ei in range(0,len(self.environment_assisted_order)):
                    gam_num = self.environment_assisted_order[ei]
                    self.gamma[gam_num] = Hc_coeff[self.ctrlnum+ei]


if __name__ == '__main__':
   print(Lindblad_Dynamics.__doc__)
   print(Cramer_Rao_bound.__doc__)
