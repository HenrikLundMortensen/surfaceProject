import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import calcenergy as ce
N=5
image = ce.randSurface(N)
print(image)
"""
f(I,a,f)
"""
x=np.arange(1,101)
y1= 20 + 3 * x + np.random.normal(0, 60, 100)
y2= 20 + 3 * x + np.random.normal(0, 60, 100)

plt.figure(1)
plt.plot(x,y1,"o")
plt.figure(2)
plt.plot(x,y2,"x")
plt.show()

def plotFeatures(f):
    d=1
