# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:45:47 2019

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps=1e-3
t=np.arange(-1,1+steps,steps)
def my_step(t):
    y = np.zeros(t.shape)
    for i in range (len(t)) :
        if t[i] >= 0: 
            y[i]=1
        else: 
            y[i]=0
    return y

def my_ramp(t):
#   y = np.zeros((len(t),1)) 
   y = np.zeros(t.shape)
   for i in range (len(t)) :
        if t[i] >= 0: 
            y[i]=t[i]
        else: 
            y[i]=0
   return y

def f1(t):
    y= my_step(t-2)-my_step(t-9)
    return y
    
def f2(t):
    y= np.exp(-t)
    return y

def f3(t):
    y= (my_ramp(t-2)*(my_step(t-2)-my_step(t-3))+
        my_ramp(4-t)*(my_step(t-3)-my_step(t-4)))
    return y

steps=1e-2
t=np.arange(0,20+steps,steps)


plt.figure()                #Plots of three functions
plt.figure(figsize=(7,6))
plt.subplot(3,1,1)          #f1(t)
plt.title('f1(t), f2(t), f3(t)')
plt.plot(t,f1(t))
plt.grid(True)
plt.ylabel('f1(t)')
plt.xlabel('t')
plt.xlim([0,10])
plt.ylim([0,1.5])
plt.subplot(3,1,2)          #f2(t)
plt.plot(t,f2(t))
plt.grid(True)
plt.ylabel('f2(t)')
plt.ylim([0,1.5])
plt.subplot(3,1,3)          #f3(t)
plt.plot(t,f3(t))
plt.grid(True)
plt.ylabel('f3(t)')
plt.xlim([0,10])
plt.ylim([0,1.5])
plt.show()

def my_conv(f1,f2):         #Custom Convolution Function
    nf1= len(f1)            #array for length of the first input function
    nf2= len(f2)            #array for length of the second input function
    f1ext= np.append(f1,np.zeros((1,nf2-1)))    #extension of arrays
    f2ext= np.append(f2,np.zeros((1,nf1-1)))
    result= np.zeros(f1ext.shape)               #result variable cleared
    for i in range(nf2+nf1-2):                  #indexing of both functions
        result[i]=0
        for j in range(nf1):
            if (i-j+1 > 0):
                try:
                    result[i]=result[i]+f1ext[j]*f2ext[i-j+1]   #Convolution
                except:
                    print(i,j)
    return result           #return the convolution
 
steps=1e-2
t=np.arange(0,20+steps,steps)
nn= len(t)
tExt=np.arange(0,2*t[nn-1],steps)

           
c1= my_conv(f1(t),f2(t))*steps          #Custom Built Convolution
c2= my_conv(f2(t), f3(t))*steps
c3= my_conv(f1(t),f3(t))*steps

K1= sig.convolve(f1(t),f2(t))*steps     #Built-in Convolution
K2= sig.convolve(f2(t),f3(t))*steps
K3= sig.convolve(f1(t),f3(t))*steps

plt.figure()               #Convolution 1 Plot
plt.figure(figsize=(7,6))
plt.plot(tExt,c1, label = 'Custom Made Convolution')
plt.plot(tExt,K1,'--', label='Built-in Convolution')
plt.grid(True)
plt.title('conv(f1(t)*f2(t))')
plt.ylabel('f1(t)*f2(t)')
plt.xlabel('t')
plt.xlim([0,20])
plt.ylim([0,1.5])
plt.legend()
plt.show

plt.figure()               #Convolution 2 Plot
plt.figure(figsize=(7,6))
plt.plot(tExt,c2, label = 'Custom Made Convolution')
plt.plot(tExt,K2, '--', label='Built-in Convolution')
plt.grid(True)
plt.title('conv(f2(t)*f3(t))')
plt.ylabel('f2(t)*f3(t)')
plt.xlabel('t')
plt.xlim([0,20])
plt.ylim([0,1])
plt.legend()
plt.show

plt.figure()               #Convolution 3 Plot
plt.figure(figsize=(7,6))
plt.plot(tExt,c3, label = 'Custom Made Convolution')
plt.plot(tExt,K3,'--', label='Built-in Convolution')
plt.grid(True)
plt.title('conv(f1(t)*f3(t))')
plt.ylabel('f1(t)*f3(t)')
plt.xlabel('t')
plt.xlim([0,20])
plt.ylim([0,1.5])
plt.legend()
plt.show

       

        