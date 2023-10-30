import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize']=[16,8]
plt.rcParams.update({'font.size':18})

H = np.loadtxt(os.path.join('.', 'DATA', 'housing.data'))
b=H[:,-1] #housing values in 1000s
A=H[:,:-1] #other factors

#pad with ones for nonzero offset
A = np.pad(A, [(0,0),(0,1)], mode='constant', constant_values=1)

#solve Ax=b using SVD

U,S,VT= np.linalg.svd(A, full_matrices=0)
x = VT.T@np.linalg.inv(np.diag(S))@U.T@b

fig=plt.figure()
ax1=fig.add_subplot(121)

plt.plot(b, color='k', linewidth=2, label='Housing Value')
plt.plot(A@x, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.legend()

plt.show()