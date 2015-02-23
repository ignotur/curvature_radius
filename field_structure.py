from scipy.special import *
from scipy.integrate import *
import matplotlib.pyplot as plt
import numpy as np
import math
import cmath

## Program computes the curvature of
## magnetic field lines near magnetic pole
## of NS. Based on Asseo & Khechinashvili (2002)
## Written by A.P. Igoshev

## These functions correct for the Schwarzschild metric
def f_l (l, r, r_g):
	return hyp2f1(l, l+2, 2*(l+1), r_g/r)

def g_l (l, r, r_g):
	return hyp2f1(l+1, l+2, 2*(l+1), r_g/r)

def Y_lm       (l, m, theta, phi):
	return math.sqrt( (2.*l+1.)/(math.pi * 4.0) * math.factorial(l - m) / math.factorial(l + m) ) * lpmv (m, l, math.cos(theta)) * cmath.exp(1j*m*phi)

def Y_lm_diff  (l, m, theta, phi):
	return -math.sqrt( (2.*l+1.)/(math.pi * 4.0) * math.factorial(l - m) / math.factorial(l + m) ) *  lpmn(m, l, math.cos(theta))[1][m][l] * cmath.exp(1j*m*phi) * math.sin(theta)

## Classical non-relativistic magnetic field components. Eqs. (4), (5), (6)

def B_r_lm     (l, m, theta, phi, r, q_lm, r_g, r_ns):
	return -4.*math.pi * (l+1.)/(2.0*l + 1.0) * q_lm * (r_ns / r) ** (l+2.0) * Y_lm (l, m, theta, phi)

def B_theta_lm (l, m, theta, phi, r, q_lm, r_g, r_ns):
	return 4.0 * math.pi / (2.0*l + 1.0) * q_lm * (r_ns / r) ** (l + 2.0) * Y_lm_diff (l, m, theta, phi)
 
def B_phi_lm   (l, m, theta, phi, r, q_lm, r_g, r_ns):
	return 4.0 * math.pi / (2.0*l + 1.0) * q_lm * (r_ns / r) ** (l + 2.0) * (1j * m) * math.sin (theta) * Y_lm (l, m, theta, phi)

## Magntic field components strength because of relativistic effects

def B_wave_r_lm (l, m, theta, phi, r, q_lm, r_g):
	return f_l (l, r, r_g) * B_r_lm (l, m, theta, phi, r, q_lm, r_g, r_ns)

def B_wave_theta_r_lm (l, m, theta, phi, r, q_lm, r_g):
	return sqrt(1.0 - r_g/r) * g_l (l, r, r_g) * B_theta_lm (l, m, theta, phi, r, q_lm, r_g, r_ns)

def B_wave_phi_r_lm (l, m, theta, phi, r, q_lm, r_g):
	return sqrt(1.0 - r_g/r) * g_l (l, r, r_g) * B_phi_lm (l, m, theta, phi, r, q_lm, r_g, r_ns)

## Total magnetic field (classical) eq. 

def B (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns):
	B_r = 0
	B_theta = 0
	B_phi = 0
	for l in range (0, l_lim):
		for m in range (0, m_lim):
			B_r     = B_r     + B_r_lm     (l, m, theta, phi, r, Q[l, m], r_g, r_ns)
			B_theta = B_theta + B_theta_lm (l, m, theta, phi, r, Q[l, m], r_g, r_ns)
			B_phi   = B_phi   + B_phi_lm   (l, m, theta, phi, r, Q[l, m], r_g, r_ns)

	return [B_r, B_theta, B_phi]

## Total magnetic field (general relativity effects)

def B_wave (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns):
	B_r = 0
	B_theta = 0
	B_phi = 0
	for l in range (0, l_lim):
		for m in range (0, m_lim):
			B_r     = B_r     + B_wave_r_lm     (l, m, theta, phi, r, Q[l, m], r_g, r_ns)
			B_theta = B_theta + B_wave_theta_lm (l, m, theta, phi, r, Q[l, m], r_g, r_ns)
			B_phi   = B_phi   + B_wave_phi_lm   (l, m, theta, phi, r, Q[l, m], r_g, r_ns)

	return [B_r, B_theta, B_phi]

## Differential equation for geometry of magnetic field lines. eq. (33-34)
## Possibly it is not proper to divide on r_ns (??)

def diff_eq (angle, r, Q, l_lim, m_lim, r_g, r_ns):
	res = np.zeros(2)
	theta = angle[0]
	phi   = angle[1]
	field = B (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns)
	res[0] = (1.0/r) * field[1].real / field[0].real
	res[1] = (1.0/r*math.sin(theta)) * field[2].real / field[0].real
	return res


r_ns = 1e6      ## 10 km in cm/s
r_g  = 4.1e5    ## for M = 1.4 M_solar

r_d  = np.arange(50*r_ns, r_ns, -(49.0/1000.0)*r_ns) 

#######################################
## Multipole expansion for test case ##
#######################################
l_lim=4
m_lim=1
Q = np.zeros([l_lim, m_lim])
print Q
Q[1][0] = 100000.0
#Q[2][0] = -100000.0
#Q[3][0] = 100000.0

#######################################


#delta_r = 49.0/1.e3 * r_ns
#for i in range (0, len(r_d)):
#	print Y_lm_diff  (0, 0, init_cond[0], 0)
#	print B_theta_lm (0, 0, init_cond[0], 0, r_d[i], Q[0][0], r_g, r_ns)
#	print B (l_lim, m_lim, Q, r_d[i], init_cond[0], init_cond[1], r_g, r_ns)
#	print diff_eq (init_cond, r_d[i], Q, l_lim, m_lim, r_g, r_ns)
#	tmp = delta_r * diff_eq (init_cond, r_d[i], Q, l_lim, m_lim, r_g, r_ns)
#	init_cond[0] = init_cond[0] + tmp[0]
#	print 'Angle change: ', tmp, init_cond[0] 


## ---------- Final test ----------------
## Initial condition - dipole field at large (50 R_ns) distance
init_cond = np.zeros(2)
init_cond[0] = math.asin(1.45e-2 * math.sqrt(50.0))
init_cond[1] = 0.0



res_analit = np.arcsin(1.45e-2 * np.sqrt(r_d/r_ns))
res = odeint (diff_eq, init_cond, r_d, args=(Q, l_lim, m_lim, r_g, r_ns)).T
res_analit = np.arcsin(1.45e-2 * np.sqrt(r_d/r_ns)/10.)
res = odeint (diff_eq, init_cond, r_d, args=(Q, l_lim, m_lim, r_g, r_ns)).T



#print res[0]
print res[0][100:103], res_analit[100:103]
plt.plot (r_d, res[0], r_d, res_analit)
plt.xscale('log')
plt.xlabel('Distance from NS surface (cm)')
plt.ylabel('Angle (radians)')
plt.show()
##---------------------------------------


#print Y_lm_diff (10, 0, math.radians(3.*1.75), 0), sph_harm(0, 10, 0, math.radians(3.*1.75))
#print lpmn(2, 4, .0)[0]
#print lpmv(2, 4, 0.0)
#print lpmv(0, 4, 0.0)
#for i in range (0, 101):
#	print eval_legendre(9, -1. + 0.02*i)
#	print Y_lm(10, 0, math.radians(i*1.75), 0)
