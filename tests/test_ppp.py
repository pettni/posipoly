import sympy
from sympy.abc import x, y
import numpy as np
import scipy.sparse as sp

import mosek
from posipoly.polynomial import Polynomial
from posipoly.polylintrans import PolyLinTrans
from posipoly.ppp import *
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

def test_psd():

  # find minimal value x such that
  # [1 1; 1 x] is positive semi-definite

  c = np.array([0,0,1])

  Aeq = np.array([[1,0,0], [0,1,0]])
  beq = np.array([1, 1])

  ppp_list = [ [0, 3] ]

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'psd')

  mat = vec_to_mat(sol)
  v, _ = np.linalg.eig(mat)

  np.testing.assert_almost_equal(min(v), 0)

def test_sdd1():

  tot_deg = 6               # overall degree of problem
  sigma_deg = tot_deg - 2   # degree of sigma

  p = Polynomial.from_sympy(-x**2 - y**2 + x, [x,y])
  g = Polynomial.from_sympy(1 - x**2 - y**2, [x,y])

  A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
  A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcg()  # gram to coefs
  A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcg()           # gram to coefs

  n_gamma = A_gamma.shape[1]
  n_sigma = A_sigma.shape[1]
  n_S1 = A_S1.shape[1]

  Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
  beq = p.mon_coefs(tot_deg)

  c = np.zeros(Aeq.shape[1])
  c[0] = -1

  ppp_list = [ [n_gamma, n_sigma], [n_gamma+n_sigma, n_S1] ]

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'sdd')
  np.testing.assert_almost_equal(sol[0], -2)

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'psd')
  np.testing.assert_almost_equal(sol[0], -2)


def test_sdd2():

  tot_deg = 6               # overall degree of problem
  sigma_deg = tot_deg - 2   # degree of sigma

  p = Polynomial.from_sympy(x**2 + y**2, [x,y])
  g = Polynomial.from_sympy(1 - x**2 - y**2, [x,y])

  A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
  A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcg()  # gram to coefs
  A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcg()           # gram to coefs

  n_gamma = A_gamma.shape[1]
  n_sigma = A_sigma.shape[1]
  n_S1 = A_S1.shape[1]

  Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
  beq = p.mon_coefs(tot_deg)

  c = np.zeros(Aeq.shape[1])
  c[0] = -1

  ppp_list = [ [n_gamma, n_sigma], [n_gamma+n_sigma, n_S1] ]

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'sdd')
  np.testing.assert_almost_equal(sol[0], 0.)

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'psd')
  np.testing.assert_almost_equal(sol[0], 0.)

def test_sdd3():

  tot_deg = 6               # overall degree of problem
  sigma_deg = tot_deg - 2   # degree of sigma

  p = Polynomial.from_sympy(2+(x-0.5)**2 + y**2, [x,y])
  g = Polynomial.from_sympy(1 - x**2 - y**2, [x,y])

  A_gamma = PolyLinTrans.eye(1, 2, 0, tot_deg).as_Tcc()     # coefs to coefs
  A_sigma = PolyLinTrans.mul_pol(2, sigma_deg, g).as_Tcg()  # gram to coefs
  A_S1 = PolyLinTrans.eye(2, 2, tot_deg).as_Tcg()           # gram to coefs

  n_gamma = A_gamma.shape[1]
  n_sigma = A_sigma.shape[1]
  n_S1 = A_S1.shape[1]

  Aeq = sp.bmat([[A_gamma, A_sigma, A_S1]])
  beq = p.mon_coefs(tot_deg)

  c = np.zeros(Aeq.shape[1])
  c[0] = -1

  ppp_list = [ [n_gamma, n_sigma], [n_gamma+n_sigma, n_S1] ]

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'sdd')
  np.testing.assert_almost_equal(sol[0], 2.)

  sol, _ = solve_ppp(c, Aeq, beq, None, None, ppp_list, 'psd')
  np.testing.assert_almost_equal(sol[0], 2.)


def test_ppp():

  Aiq = -np.eye(2)
  biq = np.zeros(2)

  c = [1, 1]

  sol, sta = solve_ppp(c, None, None, Aiq, biq, [])
  
  np.testing.assert_equal(sta, mosek.solsta.optimal)
  np.testing.assert_almost_equal(sol, [0,0])

  Aeq = np.array([[1,0]])
  beq = [1]

  sol, sta = solve_ppp(c, Aeq, beq, Aiq, biq, [])
  np.testing.assert_equal(sta, mosek.solsta.optimal)
  np.testing.assert_almost_equal(sol, [1,0])

  Aeq = np.array([[1,2]])
  beq = [1]

  sol, sta = solve_ppp(c, Aeq, beq, Aiq, biq, [])
  np.testing.assert_equal(sta, mosek.solsta.optimal)
  np.testing.assert_almost_equal(sol, [0,0.5])
