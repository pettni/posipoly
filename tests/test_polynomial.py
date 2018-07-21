from posipoly.polynomial import *

def test_evaluate():

  p = Polynomial({(2,0): 1, (0,2): 1})

  np.testing.assert_equal(p.evaluate(1,1), 2)
  np.testing.assert_equal(p.evaluate(2,2), 8)
  np.testing.assert_equal(p.evaluate(1,3), 10)
  np.testing.assert_equal(p.evaluate(3,1), 10)

def test_from_grlex():

  p = Polynomial.from_mon_coefs(2, [1,1,1])

  np.testing.assert_equal(p.d, 1)
  np.testing.assert_equal(p.evaluate(1,1), 3)
  np.testing.assert_equal(p.evaluate(1,-1), 1)
  np.testing.assert_equal(p.evaluate(1,-2), 0)

