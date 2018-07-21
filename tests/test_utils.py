import numpy as np

from posipoly.utils import *

def test_k_to_ij1():
  i, j = k_to_ij(0, 10)
  np.testing.assert_equal(i, 0)
  np.testing.assert_equal(j, 0)
  i, j = k_to_ij(9, 10)
  np.testing.assert_equal(i, 3)
  np.testing.assert_equal(j, 3)
  i, j = k_to_ij(5, 10)
  np.testing.assert_equal(i, 1)
  np.testing.assert_equal(j, 2)

def test_k_to_ij2():
  i, j = k_to_ij(0, 1)
  np.testing.assert_equal(i, 0)
  np.testing.assert_equal(j, 0)

def test_k_to_ij3():
  i, j = k_to_ij(2, 6)
  np.testing.assert_equal(i, 0)
  np.testing.assert_equal(j, 2)
  i, j = k_to_ij(3, 6)
  np.testing.assert_equal(i, 1)
  np.testing.assert_equal(j, 1)

def test_ij_to_k():
  k = ij_to_k(0,2,6) 
  np.testing.assert_equal(k, 2)
  k = ij_to_k(2,0,6) 
  np.testing.assert_equal(k, 2)
  k = ij_to_k(3,3,10) 
  np.testing.assert_equal(k, 9)
  k = ij_to_k(0,0,10) 
  np.testing.assert_equal(k, 0)
  k = ij_to_k(1,1,6) 
  np.testing.assert_equal(k, 3)
