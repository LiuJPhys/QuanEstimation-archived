# -*- coding: utf-8 -*-
import numpy as np
import os
from QuantumPython.dynamics import *
from QuantumPython import basic 
#import matplotlib.pyplot as plt
from os import path

def Determine_fun(Fini,Ffinal,Hc_coeff,iteration,error):
    diff_F = Ffinal-Fini;
    if diff_F < 0:
       print ('Warning: the value of QFI is decreasing!')
       print ('The iteration is forced to stop!')
       filede = 'ctrl_coeff'+str(iteration)+'.npy';
       os.remove(filede);
       index = 0;
    elif diff_F > 0 and diff_F < error:
         print ('Congratulations, you have it!')
         filename = 'ctrl_coeff_final.npy';
         np.save(filename,Hc_coeff);
         index = 0;
    else:
        index = 1;
        print ('The value of QFI is: '+str(Ffinal)+'')
        print ('increase of QFI: '+str(diff_F)+'')
        print ('Loop number: '+str(iteration)+'')
        print ('======================')
        #saving the control coefficients:
        filename = 'ctrl_coeff'+str(iteration)+'.npy';
        np.save(filename,Hc_coeff);
        if iteration > 2:
           filename_delete = 'ctrl_coeff'+str(iteration-2)+'.npy';
           os.remove(filename_delete);
    return index

#=========================================================
#                 Main program
#=========================================================

#--------------------------------------
sx = basic.sigmax();
sy = basic.sigmay();
sz = basic.sigmaz();
sp = basic.sigmaplus();
sm = basic.sigmaminus();
#--------------------------------------
control_switch = True;
gradient_switch = False;
#--------------------------------------
w0 = 1.0;
H = 0.5*w0*sz;
dH = H/w0;
Lvec = [sx];
gamma = [0.05];

rho_initial = np.array([[0.5,0.5],[0.5,0.5]]);

delta_t = 0.01;
t_final = 20.0;
num = int(t_final/delta_t);

epsilon = 0.01;
error = 0.0001;
               
Hc_control = [sx,sy,sz]; 

filename = 'ctrl_coeffT'+str(int(t_final))+'.npy';
if os.path.exists(filename) == 1:
   print('================================')
   print('find the control file.');
   Hc_coeff = np.load(filename); 
else:   
   sx_coeff = [0. for i in range(0,num)];
   sy_coeff = [0. for i in range(0,num)];
   sz_coeff = [-0.4999 for i in range(0,num)];
   sx_coeff = np.array(sx_coeff);
   sy_coeff = np.array(sy_coeff);
   sz_coeff = np.array(sz_coeff);

   #sx_coeff = np.random.randn(num);
   #sy_coeff = np.random.randn(num);
   #sz_coeff = np.random.randn(num);
   Hc_coeff = [sx_coeff,sy_coeff,sz_coeff];
   print('================================')
   print('controls are generated manually or randomly.')
  
#-------------------------------------

CR = Cramer_Rao_bound(H,Lvec,gamma,rho_initial,delta_t,t_final,\
     dH,Hc_control,Hc_coeff,control_switch);  

CR.general_information();
iteration = 1;
tspan = CR.tspan;   

#------------------------------            
          
if control_switch == False and gradient_switch == False: 
   Fno = [0. for i in range(0,num)];
   CR.data_generation('vector');
   for fi in range(0,num):
       Ftemp, _ = CR.QFI(CR.rho[fi],CR.rho_derivative[fi]);
       Fno[fi] = np.real(Ftemp);                    
   Fno = np.array(Fno);
   print(Fno[num-1]);
   print(CR.rho[num-1])
   #fig, = plt.plot(tspan,Fno,'b-',linewidth=2.0,label='no controls');
   #plt.show();
elif control_switch == True and gradient_switch == False:
     Fno = [0. for i in range(0,num)];
     CR.data_generation('vector');
     for fi in range(0,num):
         Ftemp, _ = CR.QFI(CR.rho[fi],CR.rho_derivative[fi]);
         Fno[fi] = np.real(Ftemp);                    
     Fno = np.array(Fno);
     print(Fno[num-1]);

elif control_switch == True and gradient_switch == True:
     CR.data_generation('vector');
     F, _ = CR.QFI(CR.rho[num-1],CR.rho_derivative[num-1]);
     while True:
         CR.GRAPE_QFI(epsilon);
         CR.data_generation('vector');
         Fup, _ = CR.QFI(CR.rho[num-1],CR.rho_derivative[num-1]);
      
         index = Determine_fun(F,Fup,CR.control_coefficients,iteration,error);
         if index == 0:
            break
         else:
           F = Fup;
           iteration = iteration+1;   
   
   
#---------------------------------------------------------
#              Evolution test
#---------------------------------------------------------
#rho_test = [[] for i in range(0,num)]; 
#L = CR.evolution(0);
#rho_ini_vec = np.reshape(rho_initial,(4,1));
#rhoxx = np.dot(L,rho_ini_vec);
#rho00 = [0. for i in range(0,num)];
#rho01 = [0. for i in range(0,num)];
#rho00[0] = rhoxx[0];
#rho01[0] = rhoxx[2];
#for i in range(1,num):
#    rhoxx = np.dot(L,rhoxx);
#    rho00[i] = rhoxx[0];
#    rho01[i] = rhoxx[2];    
#fig = plt.figure()
#fig, = plt.plot(tspan,rho00,'k-',linewidth=4.0);
#fig, = plt.plot(tspan,np.real(rho01),'b-',linewidth=4.0);
#plt.show()
#-----------------------------------------------------------     



