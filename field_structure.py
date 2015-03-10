from scipy.special import *
from scipy.integrate import *
from string import *
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

## A unit vector in the direction of local magnetic field. eq. (9)

def b_unit (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns):
	vect = B (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns)
	size_vect = math.sqrt(vect[0].real**2 + vect[1].real**2 + vect[2].real**2)
	return [ vect[0].real/size_vect, vect[1].real/size_vect, vect[2].real/size_vect ]

## Numerical divergency of local magnetic field

def div_b_uni_num  (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns):
	h = 1e2
	b_r_l = b_unit (l_lim, m_lim, Q, r-h, theta, phi, r_g, r_ns)
	b_r_r = b_unit (l_lim, m_lim, Q, r+h, theta, phi, r_g, r_ns)
	db_r=[]
	db_r.append(0.5*(b_r_r[0] - b_r_l[0]) / h)
	db_r.append(0.5*(b_r_r[1] - b_r_l[1]) / h)
	db_r.append(0.5*(b_r_r[2] - b_r_l[2]) / h)

	h = 1e-4
	b_theta_l = b_unit (l_lim, m_lim, Q, r, theta-h, phi, r_g, r_ns)
	b_theta_r = b_unit (l_lim, m_lim, Q, r, theta+h, phi, r_g, r_ns)
	db_theta=[]
	db_theta.append(0.5*(b_theta_r[0] - b_theta_l[0]) / (r*h))
	db_theta.append(0.5*(b_theta_r[1] - b_theta_l[1]) / (r*h))
	db_theta.append(0.5*(b_theta_r[2] - b_theta_l[2]) / (r*h))

	h = 1e-4
	b_phi_l = b_unit (l_lim, m_lim, Q, r, theta, phi-h, r_g, r_ns)
	b_phi_r = b_unit (l_lim, m_lim, Q, r, theta, phi+h, r_g, r_ns)
	db_phi=[]
	db_phi.append(0.5*(b_phi_r[0] - b_phi_l[0]) / (r*math.sin(theta)*h))
	db_phi.append(0.5*(b_phi_r[1] - b_phi_l[1]) / (r*math.sin(theta)*h))
	db_phi.append(0.5*(b_phi_r[2] - b_phi_l[2]) / (r*math.sin(theta)*h))

	return [db_r, db_theta, db_phi]	

## Function to compute curvature radius

def curvature_radius  (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns):
	b_ =  b_unit (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns)
	div_b_ = div_b_uni_num  (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns)
	f1 = b_[0] * (div_b_[0])[0] + b_[1] * (div_b_[1])[0] + b_[2] * (div_b_[2])[0]
	f2 = b_[0] * (div_b_[0])[1] + b_[1] * (div_b_[1])[1] + b_[2] * (div_b_[2])[1]
	f3 = b_[0] * (div_b_[0])[2] + b_[1] * (div_b_[1])[2] + b_[2] * (div_b_[2])[2]
	res = 1.0 / math.sqrt(f1**2 + f2**2 + f3**2)
	res = res / r_ns

#	res = 1.0 / (np.dot(first_factor, second_factor))
	return res

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

def diff_eq_flat_angle (r, theta, Q, l_lim, m_lim, r_g, r_ns):
#	res = np.zeros(2)
#	theta = angle[0]
#	phi   = angle[1]
	phi=0
	field = B (l_lim, m_lim, Q, r, theta, phi, r_g, r_ns)
#	res[0] = (1.0/r) * field[1].real / field[0].real
	res =  r * field[0].real / field[1].real
#	res[1] = (1.0/r*math.sin(theta)) * field[2].real / field[0].real
	return res


## Initial condition - dipole field at large (50 R_ns) distance
## Test over analytical solution when it is possible (pure dipole field)

def plot_angle (l_lim, m_lim, Q, P, flag_dipole, r_ns, r_g, r_d):
	for i in range (-10, 10):
		init_cond = np.zeros(2)
		init_cond[0] = math.asin(1.45e-2 * math.sqrt(1.0/P) * i * math.sqrt(50.0)/10.)
		init_cond[1] = 0.0
		res_analit = np.arcsin(1.45e-2 * math.sqrt(1.0/P) * i * np.sqrt(r_d/r_ns)/10.)
		res = odeint (diff_eq, init_cond, r_d, args=(Q, l_lim, m_lim, r_g, r_ns)).T
		print i, len(r_d), len(res), len(res_analit)
		plt.plot(r_d, res[0], 'g')
		if (flag_dipole):
			plt.plot(r_d, res_analit, 'b')
	plt.show()

def plot_xy (l_lim, m_lim, Q, P, flag_dipole, r_ns, r_g, r_d):
	for i in range (-10, 10):
		init_cond = np.zeros(2)
		init_cond[0] = math.asin(1.45e-2 * math.sqrt(1.0/P) * i * math.sqrt(50.0)/10.)
		init_cond[1] = 0.0
		res_analit = np.arcsin(1.45e-2 * math.sqrt(1.0/P) * i * np.sqrt(r_d/r_ns)/10.)
		res = odeint (diff_eq, init_cond, r_d, args=(Q, l_lim, m_lim, r_g, r_ns)).T
#		print i, len(r_d), len(res), len(res_analit)
		x=[]
		y=[]
		for k in range (0, len(res[0])):
			x.append(r_d[k]*math.cos(res[0][k]))		
			y.append(r_d[k]*math.sin(res[0][k]))
#		print x

#		plt.xscale('log')
#		plt.yscale('log')
		plt.plot(x, y, 'g')
		if (flag_dipole):
			x=[]
			y=[]
			for k in range (0, len(res[0])):
				x.append(r_d[k]*math.cos(res_analit[k]))		
				y.append(r_d[k]*math.sin(res_analit[k]))
			plt.plot(x, y, 'b')
		#################################################
		## Field lines from the 'classical' magnetic pole
		##################################################


#	for i in range (-40, 40):
#		init_cond = np.zeros(2)
#		init_cond[0] = math.asin(1.45e-2 * math.sqrt(1.0/P) * i * math.sqrt(1.0)/5.)
#		init_cond[1] = 0.0
#		init_cond_r = r_ns
#		upper_value=init_cond[0]/math.fabs(init_cond[0])
#		print 'look here----', upper_value
#		theta_d = np.arange(init_cond[0], 0.5*upper_value, 0.5/100.0*upper_value)
#		if (i != 0):
#			res = odeint (diff_eq_flat_angle, init_cond_r, theta_d, args=(Q, l_lim, m_lim, r_g, r_ns)).T	
#			flag_open = False
#			print '***********************'
#			print res
#			print '***********************'
#			#print theta_d
#			res_int = res[0]
#			print res_int
#			print '***********************'
#			x=[]
#			y=[]
#			for k in range (0, len(res_int)):
#				x.append(res_int[k]*math.cos(theta_d[k]))	
#				y.append(res_int[k]*math.sin(theta_d[k]))
#			print x
#			print y
#			plt.plot(x, y, 'r')

	plt.show()




r_ns = 1e6      ## 10 km in cm/s
r_g  = 4.1e5    ## for M = 1.4 M_solar

r_d  = np.arange(50*r_ns, r_ns, -(49.0/10000.0)*r_ns) 

#######################################
## Multipole expansion for test case ##
#######################################
l_lim=70
m_lim=1
Q = np.zeros([l_lim, m_lim])
#print Q
#Q[1][0] = 100000.0
#Q[3][0] = -90000.0
#Q[20][0] = 10000.0
#Q[10][0] = 300000.0

#######################################

########################################
## Reading multipole structure from file
########################################

g=open ('multipoles_noAccretion.d', 'r')
lines = g.readlines()

for i in range (5890, 5950):
	line = split(lines[i])
	value = float(line[1])
	Q[i-5889][0] = value
	print i-5889, line, value
	


P = 0.1
flag_dipole = True


res_analit = math.asin(1.45e-2 * math.sqrt(1.0/P) * np.sqrt(2.0*r_ns/r_ns))
print 'Here it is going to print curvature radius', res_analit
print 'R = 1.5 R_ns'
for i in range (1, 20):
	value = B (l_lim, m_lim, Q, 1.5*r_ns, i*res_analit/20., 0.0,r_g,r_ns)
	det = math.sqrt(value[0].real**2 + value[1].real**2) 
	print i*res_analit/20., curvature_radius(l_lim, m_lim, Q, 1.5*r_ns, i*res_analit/20., 0, r_g, r_ns), '\t', value[0].real, '\t', value[1].real,'\t',det 
print 'R = 5 R_ns'
for i in range (1, 20):
	value = B (l_lim, m_lim, Q, 5.0*r_ns, i*res_analit/20., 0.0,r_g,r_ns)
	det = math.sqrt(value[0].real**2 + value[1].real**2) 
	print i*res_analit/20., curvature_radius(l_lim, m_lim, Q, 5.0*r_ns, i*res_analit/20., 0, r_g, r_ns), '\t', value[0].real, '\t', value[1].real,'\t', det

print 'R = 10 R_ns'
for i in range (1, 20):
	value = B (l_lim, m_lim, Q, 10.0*r_ns, i*res_analit/20., 0.0,r_g,r_ns)
	det = math.sqrt(value[0].real**2 + value[1].real**2) 
	print i*res_analit/20., curvature_radius(l_lim, m_lim, Q, 10.*r_ns, i*res_analit/20., 0, r_g, r_ns), '\t', value[0].real, '\t', value[1].real,'\t', det






plot_xy (l_lim, m_lim, Q, P, flag_dipole, r_ns, r_g, r_d)

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

#print Y_lm_diff (10, 0, math.radians(3.*1.75), 0), sph_harm(0, 10, 0, math.radians(3.*1.75))
#print lpmn(2, 4, .0)[0]
#print lpmv(2, 4, 0.0)
#print lpmv(0, 4, 0.0)
#for i in range (0, 101):
#	print eval_legendre(9, -1. + 0.02*i)
#	print Y_lm(10, 0, math.radians(i*1.75), 0)
