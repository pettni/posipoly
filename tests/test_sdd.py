import sympy
from sympy.abc import x, y
import numpy as np
import scipy.sparse as sp

import mosek
from posipoly.polynomial import Polynomial
from posipoly.polylintrans import PolyLinTrans
from posipoly.sdd import *
from posipoly.utils import *

def test_is_dd():
	np.testing.assert_equal(is_dd(np.array([[1, 0],[0, 1]] )), True)
	np.testing.assert_equal(is_dd(np.array([[1, 1],[1, 1]] )), True)
	np.testing.assert_equal(is_dd(np.array([[1, 1.01],[1.01, 1]] )), False)
	np.testing.assert_equal(is_dd(np.array([[1, 1.01,0],[1.01, 1, 0], [0,0,1]] )), False)
	np.testing.assert_equal(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1]] )), False)
	np.testing.assert_equal(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )), True)

	np.testing.assert_equal(is_sdd(np.array([[1, 0],[0, 1]] )), True)
	np.testing.assert_equal(is_sdd(np.array([[1, 1],[1, 1]] )), True)
	np.testing.assert_equal(is_sdd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1]] )), False)
	np.testing.assert_equal(is_sdd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )), True)
	np.testing.assert_equal(is_sdd(np.array([[-1, 0], [0, -1]] )), False)

def test_sdd_index1():
	x11,x22,x12,y11,y22,y12,z11,z22,z12 = sympy.symbols('x11,x22,x12,y11,y22,y12,z11,z22,z12')

	M = [[x11,x22,x12], [y11,y22,y12], [z11,z22,z12]]

	tt = [[0 for i in range(3)] for j in range(3)]
	for i in range(3):
		for j in range(3):
			for idx in sdd_index(i,j,3):
				tt[i][j] = tt[i][j] + M[idx[0]][idx[1]]

	np.testing.assert_equal(tt[0][0]-x11-y11, sympy.numbers.Zero)
	np.testing.assert_equal(tt[0][1]-x12, sympy.numbers.Zero)
	np.testing.assert_equal(tt[0][2]-y12, sympy.numbers.Zero)
	np.testing.assert_equal(tt[1][1]-x22-z11, sympy.numbers.Zero)
	np.testing.assert_equal(tt[1][2]-z12, sympy.numbers.Zero)
	np.testing.assert_equal(tt[2][2]-z22-y22, sympy.numbers.Zero)

def test_sdd_index2():
	x11,x22,x12 = sympy.symbols('x11,x22,x12')

	M = [[x11,x22,x12]]

	tt = [[0 for i in range(2)] for j in range(2)]
	for i in range(2):
		for j in range(2):
			for idx in sdd_index(i,j,2):
				tt[i][j] = tt[i][j] + M[idx[0]][idx[1]]

	np.testing.assert_equal(tt[0][0]-x11, sympy.numbers.Zero)
	np.testing.assert_equal(tt[0][1]-x12, sympy.numbers.Zero)
	np.testing.assert_equal(tt[1][1]-x22, sympy.numbers.Zero)

def test_sdd1():

	tot_deg = 8               # overall degree of problem
	sigma_deg = tot_deg - 2   # degree of sigma

	p = Polynomial.from_sympy(-x**2 - y**2 + x)
	g = Polynomial.from_sympy(1 - x**2 - y**2)

	A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
	A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcm()  # matrix to coefs
	A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcm()           # matrix to coefs

	n_gamma = A_gamma.shape[1]
	n_sigma = A_sigma.shape[1]
	n_S1 = A_S1.shape[1]

	Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
	beq = p.mon_coefs(tot_deg)

	c = np.zeros(Aeq.shape[1])
	c[0] = 1

	numcon, numvar = Aeq.shape

	env = mosek.Env() 
	task = env.Task(0,0)

	# add variables and constraints
	task.appendvars(numvar)
	task.appendcons(numcon)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# objective
	task.putcslice(0, numvar, c)
	task.putobjsense(mosek.objsense.maximize)

	# add eq constraints (fx: fixed)
	task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# add sdd contraints
	add_sdd_mosek(task, n_gamma, n_sigma)      # make sigma(x) sdsos
	add_sdd_mosek(task, n_gamma+n_sigma, n_S1) # make pos(x) sdsos

	task.optimize()

	np.testing.assert_equal(task.getsolsta(mosek.soltype.itr), mosek.solsta.optimal)
	opt_gamma = [0.]
	task.getxxslice(mosek.soltype.itr, 0, 1, opt_gamma)

	np.testing.assert_almost_equal(opt_gamma, [-2])


def test_sdd2():

	tot_deg = 8               # overall degree of problem
	sigma_deg = tot_deg - 2   # degree of sigma

	p = Polynomial.from_sympy(x**2 + y**2)
	g = Polynomial.from_sympy(1 - x**2 - y**2)

	A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
	A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcm()  # matrix to coefs
	A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcm()           # matrix to coefs

	n_gamma = A_gamma.shape[1]
	n_sigma = A_sigma.shape[1]
	n_S1 = A_S1.shape[1]

	Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
	beq = p.mon_coefs(tot_deg)

	c = np.zeros(Aeq.shape[1])
	c[0] = 1

	numcon, numvar = Aeq.shape

	env = mosek.Env() 
	task = env.Task(0,0)

	# add variables and constraints
	task.appendvars(numvar)
	task.appendcons(numcon)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# objective
	task.putcslice(0, numvar, c)
	task.putobjsense(mosek.objsense.maximize)

	# add eq constraints (fx: fixed)
	task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# add sdd contraints
	add_sdd_mosek(task, n_gamma, n_sigma)      # make sigma(x) sdsos
	add_sdd_mosek(task, n_gamma+n_sigma, n_S1) # make pos(x) sdsos

	task.optimize()

	np.testing.assert_equal(task.getsolsta(mosek.soltype.itr), mosek.solsta.optimal)
	opt_gamma = [0.]
	task.getxxslice(mosek.soltype.itr, 0, 1, opt_gamma)

	np.testing.assert_almost_equal(opt_gamma, [0])

def test_sdd3():

	tot_deg = 8               # overall degree of problem
	sigma_deg = tot_deg - 2   # degree of sigma

	p = Polynomial.from_sympy(2+(x-0.5)**2 + y**2)
	g = Polynomial.from_sympy(1 - x**2 - y**2)

	A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
	A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcm()  # matrix to coefs
	A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcm()           # matrix to coefs

	n_gamma = A_gamma.shape[1]
	n_sigma = A_sigma.shape[1]
	n_S1 = A_S1.shape[1]

	Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
	beq = p.mon_coefs(tot_deg)

	c = np.zeros(Aeq.shape[1])
	c[0] = 1

	numcon, numvar = Aeq.shape

	env = mosek.Env() 
	task = env.Task(0,0)

	# add variables and constraints
	task.appendvars(numvar)
	task.appendcons(numcon)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# objective
	task.putcslice(0, numvar, c)
	task.putobjsense(mosek.objsense.maximize)

	# add eq constraints (fx: fixed)
	task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# add sdd contraints
	add_sdd_mosek(task, n_gamma, n_sigma)      # make sigma(x) sdsos
	add_sdd_mosek(task, n_gamma+n_sigma, n_S1) # make pos(x) sdsos

	task.optimize()

	np.testing.assert_equal(task.getsolsta(mosek.soltype.itr), mosek.solsta.optimal)
	opt_gamma = [0.]
	task.getxxslice(mosek.soltype.itr, 0, 1, opt_gamma)

	np.testing.assert_almost_equal(opt_gamma, [2])



def test_spd1():

	tot_deg = 8               # overall degree of problem
	sigma_deg = tot_deg - 2   # degree of sigma

	p = Polynomial.from_sympy(-x**2 - y**2 + x)
	g = Polynomial.from_sympy(1 - x**2 - y**2)

	A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
	A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcm()  # matrix to coefs
	A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcm()           # matrix to coefs

	n_gamma = A_gamma.shape[1]
	n_sigma = A_sigma.shape[1]
	n_S1 = A_S1.shape[1]

	Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
	beq = p.mon_coefs(tot_deg)

	c = np.zeros(Aeq.shape[1])
	c[0] = 1

	numcon, numvar = Aeq.shape

	env = mosek.Env() 
	task = env.Task(0,0)

	# add variables and constraints
	task.appendvars(numvar)
	task.appendcons(numcon)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# objective
	task.putcslice(0, numvar, c)
	task.putobjsense(mosek.objsense.maximize)

	# add eq constraints (fx: fixed)
	task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# add sdd contraints
	add_spd_mosek(task, n_gamma, n_sigma)      # make sigma(x) sdsos
	add_spd_mosek(task, n_gamma+n_sigma, n_S1) # make pos(x) sdsos

	task.optimize()

	np.testing.assert_equal(task.getsolsta(mosek.soltype.itr), mosek.solsta.optimal)
	opt_gamma = [0.]
	task.getxxslice(mosek.soltype.itr, 0, 1, opt_gamma)

	np.testing.assert_almost_equal(opt_gamma, [-2])


def test_spd2():

	tot_deg = 8               # overall degree of problem
	sigma_deg = tot_deg - 2   # degree of sigma

	p = Polynomial.from_sympy(x**2 + y**2)
	g = Polynomial.from_sympy(1 - x**2 - y**2)

	A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
	A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcm()  # matrix to coefs
	A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcm()           # matrix to coefs

	n_gamma = A_gamma.shape[1]
	n_sigma = A_sigma.shape[1]
	n_S1 = A_S1.shape[1]

	Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
	beq = p.mon_coefs(tot_deg)

	c = np.zeros(Aeq.shape[1])
	c[0] = 1

	numcon, numvar = Aeq.shape

	env = mosek.Env() 
	task = env.Task(0,0)

	# add variables and constraints
	task.appendvars(numvar)
	task.appendcons(numcon)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# objective
	task.putcslice(0, numvar, c)
	task.putobjsense(mosek.objsense.maximize)

	# add eq constraints (fx: fixed)
	task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# add sdd contraints
	add_spd_mosek(task, n_gamma, n_sigma)      # make sigma(x) sdsos
	add_spd_mosek(task, n_gamma+n_sigma, n_S1) # make pos(x) sdsos

	task.optimize()

	np.testing.assert_equal(task.getsolsta(mosek.soltype.itr), mosek.solsta.optimal)
	opt_gamma = [0.]
	task.getxxslice(mosek.soltype.itr, 0, 1, opt_gamma)

	np.testing.assert_almost_equal(opt_gamma, [0])

def test_spd3():

	tot_deg = 8               # overall degree of problem
	sigma_deg = tot_deg - 2   # degree of sigma

	p = Polynomial.from_sympy(2+(x-0.5)**2 + y**2)
	g = Polynomial.from_sympy(1 - x**2 - y**2)

	A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
	A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcm()  # matrix to coefs
	A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcm()           # matrix to coefs

	n_gamma = A_gamma.shape[1]
	n_sigma = A_sigma.shape[1]
	n_S1 = A_S1.shape[1]

	Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
	beq = p.mon_coefs(tot_deg)

	c = np.zeros(Aeq.shape[1])
	c[0] = 1

	numcon, numvar = Aeq.shape

	env = mosek.Env() 
	task = env.Task(0,0)

	# add variables and constraints
	task.appendvars(numvar)
	task.appendcons(numcon)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# objective
	task.putcslice(0, numvar, c)
	task.putobjsense(mosek.objsense.maximize)

	# add eq constraints (fx: fixed)
	task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# add sdd contraints
	add_spd_mosek(task, n_gamma, n_sigma)      # make sigma(x) sdsos
	add_spd_mosek(task, n_gamma+n_sigma, n_S1) # make pos(x) sdsos

	task.optimize()

	np.testing.assert_equal(task.getsolsta(mosek.soltype.itr), mosek.solsta.optimal)
	opt_gamma = [0.]
	task.getxxslice(mosek.soltype.itr, 0, 1, opt_gamma)

	np.testing.assert_almost_equal(opt_gamma, [2])



def test_psd():

	env = mosek.Env() 
	task = env.Task(0,0)

	numvar = 3

	# add variables and constraints
	task.appendvars(numvar)

	# make vars unbounded (fr: free)
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# set a00 = 1, a01 = a10 = 1
	task.appendcons(2)

	task.putarow(0, [0], [1.])
	task.putarow(1, [1], [1.])
	task.putconboundslice(0, 2, [mosek.boundkey.fx] * 2, [1.] * 2, [1.] * 2)

	task.putcj(2, 1.)
	task.putobjsense(mosek.objsense.minimize)

	# add sdd contraints
	add_spd_mosek(task, 0, numvar)

	task.optimize()

	opt = [0.] * numvar

	task.getxxslice(mosek.soltype.itr, 0, numvar, opt)

	mat = vec_to_mat(opt)
	v, _ = np.linalg.eig(mat)

	np.testing.assert_almost_equal(min(v), 0)