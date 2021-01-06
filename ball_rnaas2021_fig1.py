#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit, fsolve
from scipy.special import hyp2f1

pl.style.use('style.mpl')

# Load Hopf functions by Trampedach et al. (2014)
try:
    datafile = open('TtauFeH0.dat', 'r')
except FileNotFoundError:
    import urllib.request as request
    with request.urlopen('http://cdsarc.u-strasbg.fr/ftp/J/MNRAS/442/805/TtauFeH0.dat') as remote:
        with open('TtauFeH0.dat', 'wb') as local:
            local.write(remote.read())

    datafile = open('TtauFeH0.dat', 'r')

datafile.readline() # skip first line
Teff = np.array(list(map(float, datafile.readline().split()[1:])))
logg = np.array(list(map(float, datafile.readline().split()[2:])))
FeH = np.array(list(map(float, datafile.readline().split()[1:])))
alpha = np.array(list(map(float, datafile.readline().split()[2:])))
sigma = np.array(list(map(float, datafile.readline().split()[2:])))

data = np.loadtxt(datafile.readlines()).T
datafile.close()

x = data[0]
tau = 10.0**x
y = data[2]
dy_dx = np.gradient(y, x)

# Eq. (2): dq/dx = (c₁+exp((x-a)/v))/(1+exp((x-b)/w))
def df_dx(x, *p):
    c1, a, v, b, w = p
    return (c1 + np.exp((x-a)/v))/(1+np.exp((x-b)/w))

p0 = np.array([0.07, 1.0, 1.0, 0.25, 1/7])
popt, pcov = curve_fit(df_dx, x, dy_dx, p0=p0)

# Eq. (3): q = c₀ + c₁(x-w*log(exp(b/w)+exp(x/w))) + v*exp((x-a)/v)*₂F₁(1,w/v;1+w/v;-exp((x-b)/w))
def f(x, *p):
    c0, c1, a, v, b, w = p
    V = np.exp((x-a)/v)
    W = np.exp((x-b)/w)
    return c0 + c1*(x-w*np.log(np.exp(b/w)+np.exp(x/w))) \
        + v*V*hyp2f1(1, w/v, 1+w/v, -W)

q0 = np.hstack([0.65, popt])
qopt, qcov = curve_fit(f, x, y, p0=q0)

# Eq. (4): q = c₀ + c₁*(x-b) + v*exp((x-a)/v)
def f_approx(x, *p):
    c0, c1, a, v, b, w = p
    return c0 + c1*(x-b) + v*np.exp((x-a)/v)

# Standard form that other relations use
def standard(x, *p):
    return p[0] + p[1]*np.exp(-p[2]*10.**x) + p[3]*np.exp(-p[4]*10.**x)

r_Edd = np.array([2./3., 0., 0., 0., 0.])                # Eddington
r_KS = np.array([1.39,   -0.815,  2.54,  -0.025,  30.])  # Krishna Swamy (1966)
r_SH = np.array([1.0361, -0.3134, 2.448, -0.2959, 30.])  # Vernazza, Avrett & Loeser (1981), aka VAL-C

# To find τ_eff, set T = Teff ⇒ solve 3/4(τ + q(τ)) = 1
def tau_eff(q, args):
    return fsolve(lambda tau: 0.75*(tau+q(np.log10(tau), *args))-1, 0.5)

# Print some information
print('\nCoefficients:')
for row in zip(['c0', 'c1', 'a', 'v', 'b', 'w'], qopt):
    print('%2s = %18.16f' % row)

print('\ntau_eff:')
print('%18.16f' % tau_eff(f, qopt))

print('\nMax. fractional error in solar model:')
max_err = np.abs(f(x, *qopt)/data[2]-1).max()
print(max_err)

I = (Teff > 4400) & (logg > Teff/1000 - 2.2)
print('\nMax. fractional error excluding models with Teff < 4400 K and logg < Teff/1000 K - 2.2:')
print(np.abs(f(x, *qopt)/data[1:][I]-1).max())

# This was determined by inspecting err_approx vs x
x_ok = x[57]
print('\nMax. x for which approximate formula is similarly accurate:')
print(x_ok)

print('\nMax. fraction error of approximate formulae for x <= %.4f in solar model:' % x_ok)
print(np.abs(f_approx(x, *qopt)/data[2]-1)[x<=x_ok].max())

print()

# Plot figure
xx = np.linspace(np.min(x), np.max(x), 401) # for analytic relations

pl.plot(x, data[1], '-', label="Trampedach et al. (2014, all)", alpha=0.2, c='#808080');
pl.plot(x, data[2:].T, '-', alpha=0.2, c='#808080');
pl.plot(x, data[2], 'k-', label='Trampedach et al. (2014, solar)')

pl.plot(xx, f(xx, *qopt), label='This work (eq. 3)', c='C1')
pl.plot(xx, f_approx(xx, *qopt), '--', label='This work (eq. 4)', c='C1')
t = np.log10(tau_eff(f, qopt))
pl.plot(t, f(t, *qopt), 'o', c='C1')

pl.plot(xx, xx*0 + 2/3, label='Eddington', c='C0')
pl.plot(np.log10(2/3), 2/3, 'o', c='C0')

pl.plot(xx, standard(xx, *r_KS), label='Krishna Swamy (1966)', c='C2')
t = np.log10(tau_eff(standard, r_KS))
pl.plot(t, standard(t, *r_KS), 'o', c='C2')
        
pl.plot(xx, standard(xx, *r_SH), label='VAL-C', c='C3')
t = np.log10(tau_eff(standard, r_SH))
pl.plot(t, standard(t, *r_SH), 'o', c='C3')

pl.plot([np.nan], [np.nan], 'ko', label=r"$\tau_\mathrm{eff}$")

pl.xlim(np.min(x), 1.2)
pl.ylim(np.min(data[1:])*0.95, r_KS[0]*1.05)
pl.legend(ncol=1, borderpad=0.2, labelspacing=0.4, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.4, frameon=False, fontsize='small')
pl.xlabel(r"$\log_{10}\tau\,(=x)$")
pl.ylabel(r"$q(\tau)$")
pl.show()
# pl.savefig('ball_rnaas2021_fig1.pdf')
