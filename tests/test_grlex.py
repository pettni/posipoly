import numpy as np
import itertools

from posipoly.grlex import *

def test_index_to_grlex():
  np.testing.assert_equal(index_to_grlex(29, 3), (2,0,2))
  np.testing.assert_equal(index_to_grlex(0, 3), (0,0,0))
  np.testing.assert_equal(index_to_grlex(9, 3), (2,0,0))

def test_index_to_grlex_to_index():
  for i in range(100):
    np.testing.assert_equal(grlex_to_index(index_to_grlex(i, 3)), i)

def test_grlex_iter():
  iterator = grlex_iter( (0,0,0,0) )

  idx1 = next(iterator)

  for k in range(600):
    idx2 = next(iterator)
    np.testing.assert_equal( (grlex_key( idx1) < grlex_key(idx2 )), True )
    idx1 = idx2

def test_grlex_iter2():
  iterator = grlex_iter( (0,0,0,0), 4 )

  for k in range(count_monomials_leq(4,4)):
    next(iterator)
  
  try:
    next(iterator)
  except StopIteration:
    pass
  except:
    self.fail('Unexpected exception thrown:')
  else:
    self.fail('ExpectedException not thrown')

def test_grlex_iter_multi():
  m1 = 5
  m2 = 3
  m3 = 10
  iterator = multi_grlex_iter( (0,0,0), [[0], [1], [2]], [m1,m2,m3] )

  comb_iter = itertools.product( range(m1+1), range(m2+1), range(m3+1) )

  for k in range((m1+1) * (m2+1) * (m3+1)):
    asd =  next(iterator)
    if k == m1+1:
      np.testing.assert_equal(asd, (0,1,0))

  np.testing.assert_equal(asd , (m1,m2,m3))

  try:
    next(iterator)
  except StopIteration:
    pass
  except:
    self.fail('Unexpected exception thrown:')
  else:
    self.fail('ExpectedException not thrown')

def test_vec_to_grlex1():
  coef, exp = vec_to_grlex(10,3)
  np.testing.assert_equal(exp, [(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,2), (0,1,1), (1,0,1), (0,2,0), (1,1,0), (2,0,0) ])
  np.testing.assert_equal(coef, [1,2,2,2,1,2,2,1,2,1])

def test_vec_to_grlex2():
  coef, exp = vec_to_grlex(10,2)
  np.testing.assert_equal(exp, [(0,0), (0,1), (1,0), (0,2), (0,2), (1,1), (0,3), (2,0), (1,2), (0,4) ])
  np.testing.assert_equal(coef, [1,2,2,2,1,2,2,1,2,1])

def test_vec_to_grlex3():
  coef, exp = vec_to_grlex(28,1)
  np.testing.assert_equal(exp, [tuple([i+j]) for j in range(7) for i in range(j,7)])
  np.testing.assert_equal(coef, [1 if i==j else 2 for j in range(7) for i in range(j,7)])

