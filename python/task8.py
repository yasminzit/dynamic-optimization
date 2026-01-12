import numpy as np
import matplotlib.pyplot as plt

T=10.0; dt=0.01
t=np.arange(0,T+dt,dt)

param_sets=[
    (0.6,0.2,0.8),
    (1.0,0.2,0.8),
    (1.0,0.5,0.8),
    (1.0,0.2,1.2),
    (1.5,0.2,0.8),
]

Ugrid=np.linspace(0,1,51)

def simulate(alpha,beta,gamma,u0):
    x=np.zeros_like(t); y=np.zeros_like(t)
    x[0]=1; y[0]=0
    for i in range(1,len(t)):
        x[i]=x[i-1]+(alpha*u0-beta)*x[i-1]*dt
        y[i]=y[i-1]+gamma*(1-u0)*x[i-1]*dt
    return x,y

for a,b,g in param_sets:
    best=None
    for u0 in Ugrid:
        x,y=simulate(a,b,g,u0)
        val=y[-1]
        if best is None or val>best[0]:
            best=(val,u0,x,y)

    yT,u_star,x_star,y_star=best
    print(f"α={a}, β={b}, γ={g}:  u*≈{u_star:.2f},  y(T)≈{yT:.6f},  x(T)≈{x_star[-1]:.6f}")

    plt.figure()
    plt.plot(t,x_star)
    plt.title(f"x(t), α={a}, β={b}, γ={g}, u*≈{u_star:.2f}")
    plt.xlabel("t"); plt.ylabel("x(t)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"task8_x_a{a}_b{b}_g{g}.png",dpi=200)
    plt.show()

    plt.figure()
    plt.plot(t,y_star)
    plt.title(f"y(t), α={a}, β={b}, γ={g}, u*≈{u_star:.2f}")
    plt.xlabel("t"); plt.ylabel("y(t)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"task8_y_a{a}_b{b}_g{g}.png",dpi=200)
    plt.show()
