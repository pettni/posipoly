import sympy
from sympy.abc import x, y
import numpy as np

from posipoly import *
from posipoly.ppp import *

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

def test_ppp_0():

  # find minimal value x such that
  # [1 1; 1 x] is positive semi-definite

  c = np.array([0,0,1])

  Aeq = sp.coo_matrix(np.array([[1,0,0], [0,1,0]]))
  beq = np.array([1, 1])

  env = mosek.Env() 
  task = env.Task(0,0)

  # Add free variables and objective
  task.appendvars(3)
  task.putvarboundslice(0, 3, [mosek.boundkey.fr] * 3, [0.]*3, [0.]*3 )
  task.putcslice(0, 3, c)
  task.putobjsense(mosek.objsense.minimize)

  task.appendcons(2)
  task.putaijlist(Aeq.row, Aeq.col, Aeq.data)
  task.putconboundslice(0, 2, [mosek.boundkey.fx] * 2, beq, beq)

  add_psd_mosek( task, sp.coo_matrix(np.eye(3)), np.zeros(3) )

  task.optimize()

  solution = [0.] * len(c)
  task.getxxslice(mosek.soltype.itr, 0, len(c), solution)

  mat = vec_to_mat(solution)
  v, _ = np.linalg.eig(mat)

  np.testing.assert_almost_equal(min(v), 0)

def test_ppp_1():
  # p(x) = a0 + a1 x + a2 x^2 sos
  # p(2) = 1
  # max a1

  # sos constraint added via variable
  prob = PPP()
  prob.add_var('s', 1, 2, 'pp')

  prob.add_constraint({'s': PTrans.eval(1, 2, [0], [2])}, Polynomial.one(1), 'eq')
  prob.set_objective({'s': -PTrans.eval0(1, 1)*PTrans.diff(1,2,0)})

  sol, _ = prob.solve('psd')
  pol = prob.get_poly('s')

  np.testing.assert_almost_equal(pol(2), 1)
  np.testing.assert_almost_equal((PTrans.eval0(1, 1)*PTrans.diff(1,2,0)*pol)(1), 0.25)
  np.testing.assert_almost_equal([pol[(0,)], pol[(1,)], pol[(2,)]], [0.25, 0.25, 0.0625], decimal=3)

  # same problem, sos constraint added via add_constraint
  prob = PPP()
  prob.add_var('s', 1, 2, 'coef')

  prob.add_constraint({'s': PTrans.eval(1, 2, [0], [2])}, Polynomial.one(1), 'eq')
  prob.set_objective({'s': -PTrans.eval0(1, 1)*PTrans.diff(1,2,0)})
  prob.add_constraint({'s': PTrans.eye(1,2)}, Polynomial.zero(1), 'pp')

  sol, _ = prob.solve('psd')
  pol = prob.get_poly('s')

  np.testing.assert_almost_equal(pol(2), 1)
  np.testing.assert_almost_equal((PTrans.eval0(1, 1)*PTrans.diff(1,2,0)*pol)(1), 0.25)
  np.testing.assert_almost_equal([pol[(0,)], pol[(1,)], pol[(2,)]], [0.25, 0.25, 0.0625], decimal=3)

def test_ppp1():

  tot_deg = 6               # overall degree of problem
  sigma_deg = tot_deg - 2   # degree of sigma
  n = 2

  p = Polynomial.from_sympy(-x**2 - y**2 + x, [x,y])
  g = Polynomial.from_sympy(1 - x**2 - y**2, [x,y])

  prob = PPP()
  prob.add_var('gamma', n, 0, 'coef')
  prob.add_var('sigma', n, sigma_deg, 'pp')
  prob.add_constraint({'gamma': -PTrans.eye(n, 0, n, tot_deg),
                       'sigma': -PTrans.mul_pol(n, sigma_deg, g)}, 
                       -p, 'pp')
  prob.set_objective({'gamma': [-1]})

  prob.solve('psd') 
  opt_gamma = prob.get_poly('gamma')(0,0)
  np.testing.assert_almost_equal(opt_gamma, -2.)

  prob.solve('sdd') 
  opt_gamma = prob.get_poly('gamma')(0,0)
  np.testing.assert_almost_equal(opt_gamma, -2.)

def test_ppp2():

  tot_deg = 6               # overall degree of problem
  sigma_deg = tot_deg - 2   # degree of sigma
  n = 2

  p = Polynomial.from_sympy(x**2 + y**2, [x,y])
  g = Polynomial.from_sympy(1 - x**2 - y**2, [x,y])

  prob = PPP()
  prob.add_var('gamma', n, 0, 'coef')
  prob.add_var('sigma', n, sigma_deg, 'pp')
  prob.add_constraint({'gamma': -PTrans.eye(n, 0, n, tot_deg),
                       'sigma': -PTrans.mul_pol(n, sigma_deg, g)}, 
                       -p, 'pp')
  prob.set_objective({'gamma': [-1]})

  prob.solve('psd') 
  opt_gamma = prob.get_poly('gamma')(0,0)
  np.testing.assert_almost_equal(opt_gamma, 0.)

  prob.solve('sdd') 
  opt_gamma = prob.get_poly('gamma')(0,0)
  np.testing.assert_almost_equal(opt_gamma, 0.)

def test_ppp3():

  tot_deg = 6               # overall degree of problem
  sigma_deg = tot_deg - 2   # degree of sigma
  n = 2

  p = Polynomial.from_sympy(2+(x-0.5)**2 + y**2, [x,y])
  g = Polynomial.from_sympy(1 - x**2 - y**2, [x,y])

  prob = PPP()
  prob.add_var('gamma', n, 0, 'coef')
  prob.add_var('sigma', n, sigma_deg, 'pp')
  prob.add_constraint({'gamma': -PTrans.eye(n, 0, n, tot_deg),
                       'sigma': -PTrans.mul_pol(n, sigma_deg, g)}, 
                       -p, 'pp')
  prob.set_objective({'gamma': [-1]})

  prob.solve('psd') 
  opt_gamma = prob.get_poly('gamma')(0,0)
  np.testing.assert_almost_equal(opt_gamma, 2.)

  prob.solve('sdd') 
  opt_gamma = prob.get_poly('gamma')(0,0)
  np.testing.assert_almost_equal(opt_gamma, 2.)
