import numpy as np, math
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt

a1,a3=9,20
b1,b2=33,19
umin,umax=-0.09,0.1
T=2.0

# момент переключения
ts=(40-math.log(1+19*20/33))/20

def y_piecewise(t,u1,u2):
    t=np.asarray(t)
    y=np.empty_like(t,dtype=float)
    m=t<=ts
    y0=a1
    y[m]=(y0+u1/20.0)*np.exp(20*t[m])-u1/20.0
    yts=(y0+u1/20.0)*math.exp(20*ts)-u1/20.0
    y[~m]=(yts+u2/20.0)*np.exp(20*(t[~m]-ts))-u2/20.0
    return y

def u_piecewise(t,u1,u2):
    t=np.asarray(t)
    return np.where(t<=ts,u1,u2)

getcontext().prec=80
Tdec=Decimal("2")
ts_dec=Decimal(str(ts))

def J_piecewise(u1,u2):
    def seg_int_y(y0,u,dt):
        exp=(Decimal(20)*dt).exp()
        return (y0+u/Decimal(20))/Decimal(20)*(exp-1) - u/Decimal(20)*dt
    def seg_end_y(y0,u,dt):
        exp=(Decimal(20)*dt).exp()
        return (y0+u/Decimal(20))*exp - u/Decimal(20)
    y0=Decimal(a1)
    dt1=ts_dec
    dt2=Tdec-ts_dec
    Iy1=seg_int_y(y0,u1,dt1)
    yts=seg_end_y(y0,u1,dt1)
    Iy2=seg_int_y(yts,u2,dt2)
    return Decimal(b1)*(Iy1+Iy2) - Decimal(b2)*(u1*dt1 + u2*dt2)

Jmax=J_piecewise(Decimal(str(umax)),Decimal(str(umin)))
Jmin=J_piecewise(Decimal(str(umin)),Decimal(str(umax)))

t=np.linspace(0,T,2001)

ymax=y_piecewise(t,umax,umin)
ymin=y_piecewise(t,umin,umax)
umax_t=u_piecewise(t,umax,umin)
umin_t=u_piecewise(t,umin,umax)

# графики

# y(t) — максимум
plt.figure()
plt.plot(t,ymax)
plt.axvline(ts,linestyle="--")
plt.title(f"y(t) max, V_max ≈ {float(Jmax):.3e}")
plt.xlabel("t"); plt.ylabel("y(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig("task2_y_max.png",dpi=200)
plt.show()

# y(t) - min
plt.figure()
plt.plot(t,ymin)
plt.axvline(ts,linestyle="--")
plt.title(f"y(t) min, V_min ≈ {float(Jmin):.3e}")
plt.xlabel("t"); plt.ylabel("y(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig("task2_y_min.png",dpi=200)
plt.show()

# u(t) - max
plt.figure()
plt.step(t,umax_t,where="post")
plt.axvline(ts,linestyle="--")
plt.title("u(t) - max")
plt.xlabel("t"); plt.ylabel("u(t)")
plt.ylim(umin-0.02,umax+0.02)
plt.grid(True)
plt.tight_layout()
plt.savefig("task2_u_max.png",dpi=200)
plt.show()

# u(t) - min
plt.figure()
plt.step(t,umin_t,where="post")
plt.axvline(ts,linestyle="--")
plt.title("u(t) - min")
plt.xlabel("t"); plt.ylabel("u(t)")
plt.ylim(umin-0.02,umax+0.02)
plt.grid(True)
plt.tight_layout()
plt.savefig("task2_u_min.png",dpi=200)
plt.show()

print("t* =",ts)
print("V_max ≈",float(Jmax))
print("V_min ≈",float(Jmin))
