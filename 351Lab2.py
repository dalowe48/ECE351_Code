# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:40:00 2019

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt


# Part 1
t=np.arange(0,4*np.pi,np.pi/32)
def func1(t):
    y=np.cos(t)
    return y
#f = plt.figure()
y=func1(t)
plt.plot(t,y)
plt.ylabel('y(t)')
plt.xlabel('t')
plt.grid(True)
plt.show()
#f.savefig("Part1plot.pdf", bbox_inches='tight')

# Part 2
def my_step(t):
    y = np.zeros((len(t),1))
    for i in range (len(t)) :
        if t[i] >= 0: 
            y[i]=1
        else: 
            y[i]=0
    return y

def my_ramp(t):
   y = np.zeros((len(t),1))    
   for i in range (len(t)) :
        if t[i] >= 0: 
            y[i]=t[i]
        else: 
            y[i]=0
   return y

steps=1e-3
t=np.arange(-1,1+steps,steps)

#f = plt.figure()
plt.figure()
plt.subplot(2,1,1)
plt.title('Step and Ramp functions')
plt.plot(t,my_step(t))
plt.grid(True)
plt.ylabel('Step y(t)')
plt.xlabel('t')
plt.xlim([-1,1])
plt.ylim([-.5,1.5])
plt.subplot(2,1,2)
plt.plot(t,my_ramp(t))
plt.grid(True)
plt.ylabel('Ramp y(t)')
plt.xlim([-1,1])
plt.ylim([-.5,1.5])
plt.show()
#f.savefig("Part2plot1.pdf", bbox_inches='tight')

def func2(t):
   return(my_ramp(t)-my_ramp(t-3)+5*my_step(t-3)-2*my_step(t-6)-2*my_ramp(t-6))
   

steps=1e-3
t=np.arange(-5,12+steps,steps)
f = plt.figure()
plt.plot(t,func2(t))
plt.title('Custom Function')
plt.grid(True)
plt.ylabel('Custom y(t)')
plt.xlabel('t')
plt.xticks(range(-5,10))
plt.yticks(range(-5,10))
plt.xlim([-5,10])
plt.ylim([-3,10])
plt.show()
f.savefig("Part2plot2.pdf", bbox_inches='tight')

# Part 3
steps=1e-3   #Time Reversal
t=np.arange(-12,7+steps,steps)
#f = plt.figure()
plt.plot(t,func2(-t))
plt.title('Custom Function Time Reversal')
plt.grid(True)
plt.ylabel('Custom y(t)')
plt.xlabel('t')
plt.xlim([-12,5])
plt.xticks(range(-12,5))
plt.ylim([-3,10])
plt.yticks(range(-3,10))
plt.show()
#f.savefig("Part3plot1.pdf", bbox_inches='tight')

steps=1e-3   #Shifted by 4
t=np.arange(-5,16+steps,steps)
#f = plt.figure()
plt.plot(t,func2(t-4))
plt.title('Custom Function Time Shifted (t-4)')
plt.grid(True)
plt.ylabel('Custom y(t)')
plt.xlabel('t')
plt.xlim([-5,16])
plt.ylim([-3,10])
plt.show()
#f.savefig("Part3plot2.pdf", bbox_inches='tight')

steps=1e-3   #Time Reversal and shifted by 4
t=np.arange(-15,11+steps,steps)
#f = plt.figure()
plt.plot(t,func2(-t-4))
plt.title('Custom Function Time Reversal and Shift (-t-4)')
plt.grid(True)
plt.ylabel('Custom y(t)')
plt.xlabel('t')
plt.xlim([-15,5])
plt.ylim([-3,10])
plt.show()
#f.savefig("Part3plot3.pdf", bbox_inches='tight')

steps=1e-3
t=np.arange(-5,25+steps,steps)
#f = plt.figure()
plt.plot(t,func2(t/2))
plt.title('Custom Function Time Scale t/2')
plt.grid(True)
plt.ylabel('Custom y(t)')
plt.xlabel('t')
plt.xlim([-5,23])
plt.ylim([-3,10])
plt.show()
#f.savefig("Part3plot4.pdf", bbox_inches='tight')

steps=1e-3
t=np.arange(-5,12+steps,steps)
#f = plt.figure()
plt.plot(t,func2(2*t))
plt.title('Custom Function Time Scale 2t')
plt.grid(True)
plt.ylabel('Custom y(t)')
plt.xlabel('t')
plt.xlim([-5,10])
plt.ylim([-3,10])
plt.show()
#f.savefig("Part3plot5.pdf", bbox_inches='tight')

#Derivative of Custom Function
steps=1e-2
t=np.arange(-5,10+steps,steps)
f = plt.figure()
dt= np.diff(t)
y= func2(t)
dy= np.diff(y,axis=0)/dt
f = plt.figure()
plt.plot(t[range(len(dy))],dy[:,0])
plt.title('Time Derivative of Custom Function')
plt.grid(True)
plt.ylabel('dy(t)/dt')
plt.xlabel('t')
plt.axis([-2,10,-5,5])
#plt.xlim([-5,10])
#plt.ylim([-3,10])
plt.show()
f.savefig("Part3derivative.pdf", bbox_inches='tight')
