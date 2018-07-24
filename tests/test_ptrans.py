import numpy as np

from posipoly import *

def test_const_term():
  T = PTrans.const_term(2, 2, keep_dims=False)
  np.testing.assert_equal(T.n0, 2)
  np.testing.assert_equal(T.n1, 1)
  np.testing.assert_equal(T.d1, 0)

  T = PTrans.const_term(2, 2, keep_dims=True)
  np.testing.assert_equal(T.n0, 2)
  np.testing.assert_equal(T.n1, 2)
  np.testing.assert_equal(T.d1, 0)


def test_lvar():
  L = PTrans.elvar(2,2,0,0)
  np.testing.assert_equal(L[0,0][(0,)], 1)
  np.testing.assert_equal(L[0,1][(1,)], 1)
  np.testing.assert_equal(L[0,2][(2,)], 1)
  np.testing.assert_equal(L[1,0][(0,)], 0)

def test_multi_lvar():
  L = PTrans.elvar(3,3,[0, 1],[0,3])
  np.testing.assert_equal(L[0,0,0][(0,)], 1)
  np.testing.assert_equal(L[0,0,1][(1,)], 1)
  np.testing.assert_equal(L[0,0,2][(2,)], 1)
  np.testing.assert_equal(L[1,1,0][(0,)], 0)
  np.testing.assert_equal(L[0,1,0][(0,)], 3)

def test_multi_lvar2():
  L = PTrans.elvar(3,3,[2, 0],[3,1])
  np.testing.assert_equal(L[0,0,0][(0,)], 1)
  np.testing.assert_equal(L[0,0,1][(0,)], 3)
  np.testing.assert_equal(L[0,0,2][(0,)], 9)
  np.testing.assert_equal(L[2,0,0][(0,)], 1)

def test_lvar2():
  L = PTrans.elvar(2,2,0,1)
  np.testing.assert_equal(L[0,0][(0,)], 1)
  np.testing.assert_equal(L[0,1][(1,)], 1)
  np.testing.assert_equal(L[0,2][(2,)], 1)
  np.testing.assert_equal(L[1,0][(0,)], 1)

def test_mulpol():
  poly = Polynomial({ (2,0): 1, (0,2): -1, (1,0): 3 })
  L = PTrans.mul_pol(2, 3, poly)
  np.testing.assert_equal(L.d0, 3)
  np.testing.assert_equal(L.d1, 5)
  np.testing.assert_equal(L[0,0][1,0], 3)

def test_sparse_matrix():
  L = PTrans(2,2)
  L[1,0][0,1] = 3
  L.updated()
  spmat = L.Acg
  np.testing.assert_equal(spmat.shape[1], 6)
  np.testing.assert_equal(spmat.shape[0], 3)
  np.testing.assert_equal(spmat.row[0], 1)
  np.testing.assert_equal(spmat.col[0], 2)
  np.testing.assert_equal(spmat.data[0], 6)

def test_sparse_matrix2():
  L = PTrans.eye(2,2)
  spmat = L.Acg
  for (i, idx) in enumerate(spmat.row):
    np.testing.assert_equal(idx, spmat.col[i]) # diagonal
    if i in [0,3,5]:
      np.testing.assert_equal(spmat.data[i], 1)
    else:
      np.testing.assert_equal(spmat.data[i], 2)

def test_diff():
  L = PTrans.diff(2,2,0)
  np.testing.assert_equal(L[0,0][0,0], 0)
  np.testing.assert_equal(L[1,0][0,0], 1)
  np.testing.assert_equal(L[2,0][1,0], 2)
  np.testing.assert_equal(L[4,0][3,0], 0) # above degree

def test_mul():
  L = PTrans.eye(2,3)
  L2 = L * 3
  spmat = L2.Acc
  np.testing.assert_equal(spmat.row, spmat.col)
  np.testing.assert_equal(spmat.data, [3] * 10)

def test_int():
  L = PTrans.int(2,2,0)
  np.testing.assert_equal(L[1,1][2,1], 1./2)
  np.testing.assert_equal(L[1,0][2,0], 1./2)
  np.testing.assert_equal(L[2,0][1,0], 0)
  np.testing.assert_equal(L[0,1][1,1], 1)
  np.testing.assert_equal(L[2,0][3,0], 1./3) # above degree

def test_integrate():
  L = PTrans.integrate(3,3,[0,1],[[0,1],[0,1]])
  np.testing.assert_equal(L[0,0,2][(2,)], 1)
  np.testing.assert_equal(L[1,2,0][(0,)], 1./6)
  np.testing.assert_equal(L[1,1,1][(1,)], 1./4)

def test_integrate2():
  L = PTrans.integrate(3,3,[0],[[-5,1]])
  np.testing.assert_equal(L[0,2,1][2,1], 6)
  np.testing.assert_equal(L[1,2,0][2,0], -12)

def test_evallintrans():
  L = PTrans.eye(2,3,2)
  P = Polynomial({(1,0): 2, (0,1): 2})  # 2 x + 2 y

  Pt = L * P

  np.testing.assert_equal(Pt.evaluate(1,0), 2)
  np.testing.assert_equal(Pt.evaluate(0,1), 2)
  np.testing.assert_equal(Pt.evaluate(1,1), 4)

  L = PTrans.diff(2,3,1)  # differentiation w.r.t. y
  P = Polynomial({(2,0): 1, (0,2): 1})  # x^2 + y^2

  Pt = L * P                   # should be 2y

  np.testing.assert_equal(Pt.evaluate(0,1), 2)
  np.testing.assert_equal(Pt.evaluate(0,2), 4)
  np.testing.assert_equal(Pt.evaluate(1,0), 0)

def test_composition():

  g = Polynomial({(0,2): 1, (2,0): 1})

  L = PTrans.composition(1, 2, [g])
  P = Polynomial({(0,): 1, (1,): 2, (2,): 3})

  Pt = L * P

  np.testing.assert_equal(Pt.d, 4)
  np.testing.assert_equal(Pt.data[0,0], 1)
  np.testing.assert_equal(Pt.data[2,0], 2)
  np.testing.assert_equal(Pt.data[0,2], 2)
  np.testing.assert_equal(Pt.data[4,0], 3)
  np.testing.assert_equal(Pt.data[2,2], 6)
  np.testing.assert_equal(Pt.data[0,4], 3)


def test_double_composition():

  g1 = Polynomial({(2,0): 1})  # x**2
  g2 = Polynomial({(0,2): 1})  # y**2

  L = PTrans.composition(2, 2, [g1, g2])

  P = Polynomial({(0,1): 1, (1,1): 2, (1,0): 3})   # w + 2 * z*w + 3 z

  Pt = L * P   # should be y**2 + 2 * x**2 * y**2 + 3 * x**2

  np.testing.assert_equal(Pt.d, 4)
  np.testing.assert_equal(Pt.data[2,0], 3)
  np.testing.assert_equal(Pt.data[0,2], 1)
  np.testing.assert_equal(Pt.data[2,2], 2)


def test_gaussian_expectation():

  g = Polynomial({(0,2): 1, (2,0): 1, (1,1): 2, (1,2): 1})

  L = PTrans.gaussian_expectation(n0=2, d0=3, i=1, sigma=4)

  g_t = L * g

  np.testing.assert_equal(g_t.n, 1)

  np.testing.assert_equal(g_t.evaluate(0), 16)
  np.testing.assert_equal(g_t.evaluate(1), 33)

def test_gaussian_expectation2():

  g = Polynomial({(0,2): 1, (0,4):2})

  L = PTrans.gaussian_expectation(n0=2, d0=4, i=1, sigma=0.2)

  g_t = L * g
  np.testing.assert_equal(g_t.n, 1)

  np.testing.assert_almost_equal(g_t.evaluate(0), 0.2**2 + 2*3*0.2**4)
  np.testing.assert_almost_equal(g_t.evaluate(1), 0.2**2 + 2*3*0.2**4)
