import sympy as sp
import numpy as np

from posipoly.sdd import _sdd_index, is_dd, is_sdd

def test_is_dd():
	np.testing.assert_equal(is_dd(np.array([[1, 0],[0, 1]] )), True)
	np.testing.assert_equal(is_dd(np.array([[1, 1],[1, 1]] )), True)
	np.testing.assert_equal(is_dd(np.array([[1, 1.01],[1.01, 1]] )), False)
	np.testing.assert_equal(is_dd(np.array([[1, 1.01,0],[1.01, 1, 0], [0,0,1]] )), False)
	np.testing.assert_equal(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1]] )), False)
	np.testing.assert_equal(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )), True)
	np.testing.assert_equal(is_sdd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )), True)

def test_sdd_index1():
	x11,x22,x12,y11,y22,y12,z11,z22,z12 = sp.symbols('x11,x22,x12,y11,y22,y12,z11,z22,z12')

	M = [[x11,x22,x12], [y11,y22,y12], [z11,z22,z12]]

	tt = [[0 for i in range(3)] for j in range(3)]
	for i in range(3):
		for j in range(3):
			for idx in _sdd_index(i,j,3):
				tt[i][j] = tt[i][j] + M[idx[0]][idx[1]]

	np.testing.assert_equal(tt[0][0]-x11-y11, sp.numbers.Zero)
	np.testing.assert_equal(tt[0][1]-x12, sp.numbers.Zero)
	np.testing.assert_equal(tt[0][2]-y12, sp.numbers.Zero)
	np.testing.assert_equal(tt[1][1]-x22-z11, sp.numbers.Zero)
	np.testing.assert_equal(tt[1][2]-z12, sp.numbers.Zero)
	np.testing.assert_equal(tt[2][2]-z22-y22, sp.numbers.Zero)

def test_sdd_index2():
	x11,x22,x12 = sp.symbols('x11,x22,x12')

	M = [[x11,x22,x12]]

	tt = [[0 for i in range(2)] for j in range(2)]
	for i in range(2):
		for j in range(2):
			for idx in _sdd_index(i,j,2):
				tt[i][j] = tt[i][j] + M[idx[0]][idx[1]]

	np.testing.assert_equal(tt[0][0]-x11, sp.numbers.Zero)
	np.testing.assert_equal(tt[0][1]-x12, sp.numbers.Zero)
	np.testing.assert_equal(tt[1][1]-x22, sp.numbers.Zero)
