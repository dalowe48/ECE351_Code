# Section 1
t = 1
print(t)
print("t =",t)
print('t =',t,"seconds")
print('t is now =',t/3,'\n...and can be rounded using `round()`',round(t/3,4))
print(3**2)

# Section 2
# This is a comment, and the following statement is not executed:
# print(t+5)

# Section 3
import numpy

# Section 4
list1 = [0,1,2,3]
print('list1:',list1)
list2 = [[0],[1],[2],[3]]
print('list2:',list2)
list3 = [[0,1],[2,3]]
print('list3:',list3)
array1 = numpy.array([0,1,2,3])
print('array1:',array1)
array2 = numpy.array([[0],[1],[2],[3]])
print('array2:',array2)
array3 = numpy.array([[0,1],[2,3]])
print('array3:',array3)

import numpy
print(numpy.pi)
import numpy as np
print(np.pi)

from numpy import pi
print(pi)

# Section 5
import numpy as np
print(np.arange(4),'\n',
np.arange(0,2,0.5),'\n',
np.linspace(0,1.5,4))

# Section 6
list1 = [1,2,3,4,5]
array1 = np.array(list1) # definition of a numpy array using a list
print('list1 :',list1[0],list1[4])
print('array1:',array1[0],array1[4])
array2 = np.array([[1,2,3,4,5],
[6,7,8,9,10]])
list2 = list(array2)
print('array2:',array2[0,2],array2[1,4])
print('list2 :',list2[0],list2[1]) # it is best to use numpy arrays
# for indexing specific values
# in multi-dimensional arrays
print(array2[:,2],
array2[0,:])

# Section 7
print('1x3:',np.zeros(3))
print('2x2:',np.zeros((2,2)))
print('2x3:',np.ones((2,3)))

# Section 1.3.2
# Sample code for Lab 1
import numpy as np
import matplotlib.pyplot as plt
# Define variables
steps = 0.1 # step size
x = np.arange(-2,2+steps,steps) # notice the final value is
# `2+steps` to include `2`
y1 = x + 2
y2 = x**2
# Code for plots
plt.figure(figsize=(12,8)) # start a new figure, with
# a custom figure size
plt.subplot(3,1,1) # subplot 1
plt.plot(x,y1)
plt.title('Sample Plots for Lab 1') # title for entire figure
# (all three subplots)
plt.ylabel('Subplot 1') # label for subplot 1
plt.grid(True)
plt.subplot(3,1,2) # subplot 2
plt.plot(x,y2)
plt.ylabel('Subplot 2') # label for subplot 2
plt.grid(which='both') # use major and minor grids
# (minor grids not available
# since plot is small)
plt.subplot(3,1,3) # subplot 3
plt.plot(x,y1,'--r',label='y1')
plt.plot(x,y2,'o',label='y2')
plt.axis([-2.5, 2.5, -0.5, 4.5])
plt.grid(True)
plt.legend(loc='lower right')
plt.xlabel('x') # x-axis label for all
# three subplots (entire figure)
plt.ylabel('Subplot 3') # label for subplot 3
plt.show() ### --- This MUST be included to view your plots! --- ###

# Section 1.3.3
import numpy as np
cRect = 2 + 3j
print(cRect)
cPol = abs(cRect) * np.exp(1j*np.angle(cRect))
print(cPol) # notice Python will store this in rectangular form
cRect2 = np.real(cPol) + 1j*np.imag(cPol)
print(cRect2) # converting from polar to rectangular
print(numpy.sqrt(3*5 - 5*5 + 0j))

# Section 1.3.4
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import pandas as pd
import control
import time
from scipy.fftpack import fft, fftshift
#range() # create a range of numbers (nice for `for` loops)
#np.arange() # create a numpy array that is a range of number with a defined
# step size
#np.append() # add values to the end of a numpy array
#np.insert() # add values to the beginning of a numpy array
#np.concatenate() # combine two numpy arrays
#np.linspace() # create a numpy array that contains a specified (linear) range
# of values with a specified number of elements
#np.logspace() # create a numpy array that contains a specified (logarithmic, base
# specified) range of values with a specified number of elements
#np.reshape() # reshape a numpy array
#np.transpose() # transpose a numpy array
#len() # return the number of elements in an array (horizontal)
#.size # return the number of elements in an array (vertical)
#.shape # return the dimensions of an array
#.reshape # reshape the dimensions of an array (similar to numpy.reshape() above)
#.T # transpose an array (similar to np.transpose() above)