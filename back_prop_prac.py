import numpy as np
import math

x = 3
y = -4


sigy = 1.0 / (1 + math.exp(-y))
num = x + sigy
sigx = 1.0 / (1 + math.exp(-x))
xpy = x + y
xpys = xpy**2
den = sigx + xpys
invden = 1.0 / den
f = num * invden

dnum = invden
dinvden = num
dden = (-1.0 / (den**2)) * dinvden
dsigx = dden
dxpys = dden
dxpy = 2 * xpy * dxpys
dx = dxpy
dy = dxpy
dx += ((1 - sigx) * sigx) * dsigx
dx += dnum
dsigy = dnum
dy += ((1 - sigy) * sigy) * dsigy

# VECTOR EXAMPLE

W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

dD = np.random.randn(*D.shape)
dW = dD.dot(X.T)
dX = W.T.dot(dD)
