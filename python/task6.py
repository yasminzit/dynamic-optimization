import numpy as np
import matplotlib.pyplot as plt

def candidates(x0,v0):
    out=[]
    D=0.5*v0*v0-x0
    if D>=0:
        t1=-v0+np.sqrt(D); t2=v0+t1
        if t1>=0 and t2>=0: out.append((+1.0,t1,t2,t1+t2))
    D=x0+0.5*v0*v0
    if D>=0:
        s=np.sqrt(D)
        for t1 in (v0+s, v0-s):
            t2=-v0+t1
            if t1>=0 and t2>=0: out.append((-1.0,t1,t2,t1+t2))
    return out

def solve_min_time(x0,v0,dt=0.001):
    c=candidates(x0,v0)
    if not c: raise ValueError("No 1-switch bang-bang solution")
    u1,t1,t2,T=min(c, key=lambda z: z[3])

    t=np.arange(0,T+dt,dt)
    x=np.empty_like(t); v=np.empty_like(t); u=np.empty_like(t)

    m=t<=t1
    tt=t[m]
    v[m]=v0+u1*tt
    x[m]=x0+v0*tt+0.5*u1*tt*tt
    u[m]=u1

    v1=v0+u1*t1
    x1=x0+v0*t1+0.5*u1*t1*t1

    ss=t[~m]-t1
    v[~m]=v1-u1*ss
    x[~m]=x1+v1*ss-0.5*u1*ss*ss
    u[~m]=-u1

    return t,x,v,u,T,t1,u1

def plot_case(i, x0, v0, dt=0.001, save_u=False):
    t,x,v,u,T,t1,u1=solve_min_time(x0,v0,dt=dt)

    print(f"case {i}: (x0,v0)=({x0:.3f},{v0:.3f}) | T≈{T:.6f} | t*≈{t1:.6f} | u: {u1:+.0f}->{-u1:+.0f} | x(T)≈{x[-1]:.2e}, v(T)≈{v[-1]:.2e}")

    plt.figure()
    plt.plot(t,x)
    plt.axvline(t1, linestyle="--")
    plt.title(f"x(t), case {i}: x0={x0:.2f}, v0={v0:.2f}, T≈{T:.3f}")
    plt.xlabel("t"); plt.ylabel("x(t)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"task6_case{i}_x.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(t,v)
    plt.axvline(t1, linestyle="--")
    plt.title(f"v(t), case {i}: x0={x0:.2f}, v0={v0:.2f}, T≈{T:.3f}")
    plt.xlabel("t"); plt.ylabel("v(t)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"task6_case{i}_v.png", dpi=200)
    plt.show()

    if save_u:
        plt.figure()
        plt.step(t,u,where="post")
        plt.axvline(t1, linestyle="--")
        plt.title(f"u(t), case {i}: x0={x0:.2f}, v0={v0:.2f}")
        plt.xlabel("t"); plt.ylabel("u(t)")
        plt.ylim(-1.2,1.2)
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"task6_case{i}_u.png", dpi=200)
        plt.show()

def sample_initial_conditions(n=3, seed=42, x_range=(-10,10), v_range=(-5,5)):
    rng=np.random.default_rng(seed)
    xs=rng.uniform(x_range[0], x_range[1], size=n)
    vs=rng.uniform(v_range[0], v_range[1], size=n)
    return list(zip(xs,vs))

cases = [(5,2), (-7,3), (4,-6)]

for i,(x0,v0) in enumerate(cases, start=1):
    plot_case(i, x0, v0, dt=0.001, save_u=False)
