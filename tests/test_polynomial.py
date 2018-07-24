import numpy as np

from posipoly import *

def test_evaluate():

  p = Polynomial(2, {(2,0): 1, (0,2): 1})

  np.testing.assert_equal(p(1,1), 2)
  np.testing.assert_equal(p(2,2), 8)
  np.testing.assert_equal(p(1,3), 10)
  np.testing.assert_equal(p(3,1), 10)

def test_from_grlex():

  p = Polynomial.from_mon_coefs(2, [1,1,1])

  np.testing.assert_equal(p.d, 1)
  np.testing.assert_equal(p(1,1), 3)
  np.testing.assert_equal(p(1,-1), 1)
  np.testing.assert_equal(p(1,-2), 0)

def test_raise():
  p = Polynomial(2, {(1,0): 1, (0,1): 1})

  pr1 = p**1

  np.testing.assert_equal(pr1.d, 1)
  np.testing.assert_equal(pr1[1,0], 1)
  np.testing.assert_equal(pr1[0,0], 0)
  np.testing.assert_equal(pr1[0,1], 1)

  pr2 = p**2

  np.testing.assert_equal(pr2.d, 2)
  np.testing.assert_equal(pr2[2,0], 1)
  np.testing.assert_equal(pr2[1,1], 2)
  np.testing.assert_equal(pr2[0,2], 1)

  pr3 = p**3

  np.testing.assert_equal(pr3.d, 3)
  np.testing.assert_equal(pr3[3,0], 1)
  np.testing.assert_equal(pr3[2,1], 3)
  np.testing.assert_equal(pr3[1,2], 3)
  np.testing.assert_equal(pr3[0,3], 1)

def test_mul():

  p = Polynomial(2, {(1,0): 1, (0,1): 1})

  pr2 = p*p

  np.testing.assert_equal(pr2.d, 2)
  np.testing.assert_equal(pr2[2,0], 1)
  np.testing.assert_equal(pr2[1,1], 2)
  np.testing.assert_equal(pr2[0,2], 1)

  pr3 = p*p*p

  np.testing.assert_equal(pr3.d, 3)
  np.testing.assert_equal(pr3[3,0], 1)
  np.testing.assert_equal(pr3[2,1], 3)
  np.testing.assert_equal(pr3[1,2], 3)
  np.testing.assert_equal(pr3[0,3], 1)

  pconst = p * 2
  np.testing.assert_equal(pconst.d, 1)
  np.testing.assert_equal(pconst[1,0], 2)
  np.testing.assert_equal(pconst[0,1], 2)
