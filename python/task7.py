import numpy as np
import matplotlib.pyplot as plt

T=2.0; dt=0.002
t=np.arange(0,T+dt,dt)

umax=3/8; umin=-umax
p=0.5*((t-1)**2-1)

def solve(case):
    if case=='A':
        u=np.clip(-p,umin,umax)
        f=u-t
    else:
        den=1+2*p
        u_free=np.empty_like(t)
        m=np.abs(den)>1e-12
        u_free[m]=-p[m]/den[m]
        u_free[~m]=np.sign(-p[~m])*1e6
        u=np.clip(u_free,umin,umax)
        f=u+u*u-t

    y=np.zeros_like(t)
    y[1:]=np.cumsum(f[:-1])*dt
    J=np.trapezoid(0.5*u*u-t*y+y,t)
    return y,u,J

for c in ['A','B']:
    y,u,J=solve(c)

    plt.figure()
    plt.plot(t,y)
    plt.title(f"y(t), case {c}, J ≈ {J:.6f}")
    plt.xlabel("t"); plt.ylabel("y(t)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"task7_y_{c}.png",dpi=200)
    plt.show()

    plt.figure()
    plt.step(t,u,where="post")
    plt.title(f"u(t), case {c}")
    plt.xlabel("t"); plt.ylabel("u(t)")
    plt.ylim(umin-0.05,umax+0.05)
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"task7_u_{c}.png",dpi=200)
    plt.show()

    print(f"case {c}: J =",J)
