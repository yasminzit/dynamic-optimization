import numpy as np
from scipy.integrate import solve_bvp

a1,a2,a3=9,10,20
b1,b2,b3=33,19,4
c1,c2,c3=9,10,20

def ode(t,y):
    return np.vstack((y[1], b1*y[0] + (c1/2)*np.exp(2*t)))

def bc(ya,yb):
    return np.array([ya[0]+b2, yb[0]-b3])

t=np.linspace(0,2,400)
y0=np.vstack((np.linspace(-b2,b3,t.size), np.zeros(t.size)))
sol=solve_bvp(ode,bc,t,y0,max_nodes=10000)

tf=np.linspace(0,2,1000)
y=sol.sol(tf)[0]
yp=sol.sol(tf)[1]
V=np.trapz(yp**2+a1*y*yp+b1*y**2+c1*y*np.exp(2*tf),tf)

def V_curve(yv):
    ypv=np.gradient(yv,tf)
    return np.trapz(ypv**2+a1*yv*ypv+b1*yv**2+c1*yv*np.exp(2*tf),tf)

ns=[1,2,3,5,10,20,40]
Vs=[V_curve(y+np.sin(n*np.pi*tf/2)) for n in ns]

print("status:",sol.status,sol.message)
print("V_min:",V)
print("V(y_min + sin(n*pi*t/2)):",dict(zip(ns,Vs)))

# графики

import matplotlib.pyplot as plt
import numpy as np

# 1) y*(t) (минимум)
plt.figure()
plt.plot(tf, y)
plt.title(f"y*(t), V_min = {V:.6f}")
plt.xlabel("t"); plt.ylabel("y(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig("task1_min.png", dpi=200)
plt.show()

# 2) разбег функционала
eps = 0.03
ns = np.array([1,2,3,5,10,20,40,80,160])
Vs = np.array([V_curve(y + eps*np.sin(n*np.pi*tf/2)) for n in ns])

plt.figure()
plt.plot(ns, Vs, marker="o")
plt.title("max не найден: V растёт при увеличении частоты n")
plt.xlabel("n"); plt.ylabel("V")
plt.grid(True)
plt.tight_layout()
plt.savefig("task1_max.png", dpi=200)
plt.show()

print("V_min:", V)
print("V(y_min + eps*sin(n*pi*t/2)):", dict(zip(ns, Vs)))

