import numpy as np

from posipoly.polylintrans import *
from posipoly.polynomial import *

def test_lvar():
	L = PolyLinTrans.elvar(2,2,0,0)
	np.testing.assert_equal(L[0,0][(0,)], 1)
	np.testing.assert_equal(L[0,1][(1,)], 1)
	np.testing.assert_equal(L[0,2][(2,)], 1)
	np.testing.assert_equal(L[1,0][(0,)], 0)

def test_multi_lvar():
	L = PolyLinTrans.elvar(3,3,[0, 1],[0,3])
	np.testing.assert_equal(L[0,0,0][(0,)], 1)
	np.testing.assert_equal(L[0,0,1][(1,)], 1)
	np.testing.assert_equal(L[0,0,2][(2,)], 1)
	np.testing.assert_equal(L[1,1,0][(0,)], 0)
	np.testing.assert_equal(L[0,1,0][(0,)], 3)

def test_multi_lvar2():
	L = PolyLinTrans.elvar(3,3,[2, 0],[3,1])
	np.testing.assert_equal(L[0,0,0][(0,)], 1)
	np.testing.assert_equal(L[0,0,1][(0,)], 3)
	np.testing.assert_equal(L[0,0,2][(0,)], 9)
	np.testing.assert_equal(L[2,0,0][(0,)], 1)

def test_lvar2():
	L = PolyLinTrans.elvar(2,2,0,1)
	np.testing.assert_equal(L[0,0][(0,)], 1)
	np.testing.assert_equal(L[0,1][(1,)], 1)
	np.testing.assert_equal(L[0,2][(2,)], 1)
	np.testing.assert_equal(L[1,0][(0,)], 1)

def test_mulpol():
	poly = Polynomial({ (2,0): 1, (0,2): -1, (1,0): 3 })
	L = PolyLinTrans.mul_pol(2, 3, poly)
	np.testing.assert_equal(L.d0, 3)
	np.testing.assert_equal(L.d1, 5)
	np.testing.assert_equal(L[0,0][1,0], 3)

def test_sparse_matrix():
	L = PolyLinTrans(2,2)
	L[1,0][0,1] = 3
	L.updated()
	spmat = L.as_Tcm()
	np.testing.assert_equal(spmat.shape[1], 6)
	np.testing.assert_equal(spmat.shape[0], 3)
	np.testing.assert_equal(spmat.row[0], 1)
	np.testing.assert_equal(spmat.col[0], 2)
	np.testing.assert_equal(spmat.data[0], 6)

def test_sparse_matrix2():
	L = PolyLinTrans.eye(2,2,2)
	spmat = L.as_Tcm()
	for (i, idx) in enumerate(spmat.row):
		np.testing.assert_equal(idx, spmat.col[i]) # diagonal
		if i in [0,3,5]:
			np.testing.assert_equal(spmat.data[i], 1)
		else:
			np.testing.assert_equal(spmat.data[i], 2)

def test_diff():
	L = PolyLinTrans.diff(2,2,0)
	np.testing.assert_equal(L[0,0][0,0], 0)
	np.testing.assert_equal(L[1,0][0,0], 1)
	np.testing.assert_equal(L[2,0][1,0], 2)
	np.testing.assert_equal(L[4,0][3,0], 0) # above degree

def test_mul():
	L = PolyLinTrans.eye(2,2,3)
	L2 = L * 3
	spmat = L2.as_Tcc()
	np.testing.assert_equal(spmat.row, spmat.col)
	np.testing.assert_equal(spmat.data, [3] * 10)

def test_int():
	L = PolyLinTrans.int(2,2,0)
	np.testing.assert_equal(L[1,1][2,1], 1./2)
	np.testing.assert_equal(L[1,0][2,0], 1./2)
	np.testing.assert_equal(L[2,0][1,0], 0)
	np.testing.assert_equal(L[0,1][1,1], 1)
	np.testing.assert_equal(L[2,0][3,0], 1./3) # above degree

def test_integrate():
	L = PolyLinTrans.integrate(3,3,[0,1],[[0,1],[0,1]])
	np.testing.assert_equal(L[0,0,2][(2,)], 1)
	np.testing.assert_equal(L[1,2,0][(0,)], 1./6)
	np.testing.assert_equal(L[1,1,1][(1,)], 1./4)

def test_integrate2():
	L = PolyLinTrans.integrate(3,3,[0],[[-5,1]])
	np.testing.assert_equal(L[0,2,1][2,1], 6)
	np.testing.assert_equal(L[1,2,0][2,0], -12)

def test_evallintrans():
	L = PolyLinTrans.eye(2,2,3)
	p2 = eval_Lintrans(2, L, [1,2,3])
	np.testing.assert_equal(p2[0:3], [1,2,3])
	L = PolyLinTrans.diff(2,3,1)
	p2 = eval_Lintrans(2, L, [1,2,3,4,5])
	np.testing.assert_equal(p2[0:3], [2,8,5])
