from posipoly.polynomial import *

def test_evaluate():

  p = Polynomial({(2,0): 1, (0,2): 1})

  np.testing.assert_equal(p.evaluate(1,1), 2)
  np.testing.assert_equal(p.evaluate(2,2), 8)
  np.testing.assert_equal(p.evaluate(1,3), 10)
  np.testing.assert_equal(p.evaluate(3,1), 10)

