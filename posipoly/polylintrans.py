"""
	Class PolyLinTrans for handling linear transformations between coefficient vectors,
	that can be used to formulate optimization problems over polynomials without resorting
	to (slow) symbolic computations.
"""

import copy
from math import ceil, sqrt

from itertools import chain
import scipy.sparse as sp

from posipoly.polynomial import Polynomial
from posipoly.grlex import *

class PolyLinTransRow(object):
	"""docstring for PolyLinTransRow"""

	def __init__(self, n):
		self.n = n
		self.coeffs = {}

	def __getitem__(self, midx):
		if len(midx) != self.n:
			raise TypeError('Multiindex does not match polynomial dimension')
		try:
			return self.coeffs[midx]
		except KeyError:
			return 0.

	def __setitem__(self, midx, val):
		if len(midx) != self.n:
			raise TypeError('Multiindex does not match polynomial dimension')
		self.coeffs[midx] = float(val)

class PolyLinTrans(object):
	"""Class representing a linear transformation of polynomial coefficients"""

	def __init__(self, n0, n1):
		self.n0 = n0 		# number of variables
		self.n1 = n1 		# final number of variables
		self.d0 = 0   	# initial degree
		self.d1 = 0		  # final degree
		self.cols = {}

	@staticmethod
	def eye(n0, n1, d0, d1=None):
		'''
		Identity transformation
		'''
		if n1 < n0:
			raise InputError('eye requires n1 >= n0')
		p = PolyLinTrans(n0, n1)
		p.d0 = d0
		for idx in grlex_iter((0,) *  n0, d0):
			idx_mod = idx + (0,) * (n1-n0)
			p[idx][idx_mod] = 1.
		if d1 is not None:
			p.d1 = d1
		else:
			p.d1 = d0
		return p

	@staticmethod
	def diff(n, d, xi):
		'''
		Differentiation transformation w.r.t variable xi
		'''
		p = PolyLinTrans(n,n)
		p.d0 = d
		for idx in grlex_iter((0,)*n, d):
			k = idx[xi]
			new_idx = tuple([(idx[i] if i != xi else idx[i]-1) for i in range(n)])
			if min(new_idx) >= 0:
				p[idx][new_idx] = float(k)
		p.d1 = d-1
		return p

	@staticmethod
	def int(n, d, xi):
		'''
		Integration transformation w.r.t variable xi
		'''
		p = PolyLinTrans(n,n)
		p.d0 = d
		p.d1 = d+1
		for idx in grlex_iter((0,)*n, d):
			k = idx[xi]
			new_idx = tuple([(idx[i] if i != xi else idx[i]+1) for i in range(n)])
			p[idx][new_idx] = 1./(float(k)+1)
		return p

	@staticmethod
	def elvar(n, d, xi, val):
		'''
		Transformation resulting from setting xi[i] = val[i] (new polynomial has less variables)
		'''
		if isinstance(xi, list):
			xi, val = [list(x) for x in zip(*sorted(zip(xi, val), key=lambda pair: pair[0]))]
			if len(xi) > 1:
				return PolyLinTrans.elvar(n-1,d,xi[:-1], val[:-1]) * PolyLinTrans.elvar(n,d,xi[-1], val[-1])
			else:
				return PolyLinTrans.elvar(n,d,xi[0], val[0])
		p = PolyLinTrans(n,n-1)
		for idx in grlex_iter((0,)*n, d):
			new_idx = [idx[i] for i in range(len(idx)) if i != xi]
			p[idx][tuple(new_idx)] += val**idx[xi]
		p.updated()
		return p

	@staticmethod
	def mul_pol(n, d, poly):
		'''
		Transformation representing multiplication of a degree d polynomial with a
		polynomial poly represented as ( ( midx1, cf1 ), (midx2, cf2), ... )
		'''
		p = PolyLinTrans(n,n)
		maxdeg = 0
		for midx, cf in poly.data.items():
			for idx in grlex_iter((0,)*n, d):
				new_idx = tuple([idx[i] + midx[i] for i in range(n)])
				p[idx][new_idx] = float(cf)
			maxdeg = max([maxdeg, sum(midx)])
		p.d0 = d
		p.d1 = d + maxdeg
		return p

	@staticmethod
	def integrate(n, d, dims, boxes):
		'''
		Transformation representing integration over variables in 'dims'
		over a hyperbox 'box'

		Ex: The mapping p(x,y,z) |--> q(x,z) = \int_0^1 p(x,y,z) dy 
			is obtained by mon(q) = A_int * mon(p) for 
			>> A_int = integrate(3,2,[1],[[0,1]])
		'''
		dim_box = sorted(zip(dims, boxes), key = lambda obj : -obj[0])

		p = PolyLinTrans.eye(n,n,d)  # start with right-hand identity

		for (dim, box) in dim_box:
			int_trans = PolyLinTrans.int(p.n1,d,dim)  # p.n1 -> p.n1
			upper_trans = PolyLinTrans.elvar(p.n1,d+1,dim,box[1]) # p.n1 -> p.n1-1
			lower_trans = PolyLinTrans.elvar(p.n1,d+1,dim,box[0]) # p.n1 -> p.n1-1
			p = (upper_trans - lower_trans) * int_trans * p

		# for (i, xi) in enumerate(dims):
		# 	int_trans = PolyLinTrans.int(n,d,xi)
		# 	upper_trans = PolyLinTrans.elvar(n,d+1,xi,box[i][1])
		# 	lower_trans = PolyLinTrans.elvar(n,d+1,xi,box[i][0])
		# 	p = (upper_trans - lower_trans) * int_trans * p
		p.updated() # degrees become misleading here..
		return p

	def transform(self, poly):
		'''transform a Polynomial'''
		if not poly.d <= self.d0:
			raise Exception('polynomial has too high degree')

		new_mon_coefs =	self.as_Tcc().dot(poly.mon_coefs(self.d0))
		return Polynomial.from_mon_coefs(self.n1, new_mon_coefs)

	def __getitem__(self, midx):
		if len(midx) != self.n0:
			raise TypeError('Multiindex does not match polynomial dimension')
		try:
			return self.cols[midx]
		except KeyError:
			self.cols[midx] = PolyLinTransRow(self.n1)
			return self.cols[midx]

	def __str__(self):
		ret = 'Transformation from n=%d, d=%d to n=%d, d=%d : \n' % (self.n0, self.d0, self.n1, self.d1)
		for key1, col in self.cols.items():
			for key2, val in col.coeffs.items():
				ret += str(key1) + ' --> ' + str(key2) + ' : ' + str(val) + '\n'
		return ret

	def __add__(self, other):
		""" Sum of two linear transformations """
		if not self.n0 == other.n0 and self.n1 == other.n1:
			raise TypeError('Dimension mismatch')
		ret = PolyLinTrans(self.n0, self.n1)
		for midx1, col in chain(self.cols.items(), other.cols.items()):
			for midx2, val in col.coeffs.items():
				try:
					ret[midx1][midx2] += val
				except KeyError:
					# didn't exist yet
					ret[midx1][midx2] = val
		ret.d0 = max(self.d0, other.d0)
		ret.d1 = max(self.d1, other.d1)
		return ret

	def __iadd__(self, other):
		if not self.n0 == other.n0 and self.n1 == other.n1:
			raise TypeError('Dimension mismatch')
		for midx1, col in other.cols.items():
			for midx2, val in col.coeffs.items():
				try:
					self[midx1][midx2] += val
				except KeyError:
					# didn't exist yet
					self[midx1][midx2] = val
		self.d0 = max(self.d0, other.d0)
		self.d1 = max(self.d1, other.d1)
		return self

	def __sub__(self, other):
		""" Difference of two linear transformations """
		if not self.n0 == other.n0 and self.n1 == other.n1:
			raise TypeError('Dimension mismatch')
		ret = copy.deepcopy(self)
		for midx1, col in other.cols.items():
			for midx2, val in col.coeffs.items():
				try:
					ret[midx1][midx2] -= val
				except KeyError:
					# didn't exist yet
					ret[midx1][midx2] = -val
		ret.d0 = max(self.d0, other.d0)
		ret.d1 = max(self.d1, other.d1)
		return ret

	def __mul__(self, other):
		""" Product of two linear transformations (other is the right one) """
		if not isinstance(other, PolyLinTrans):
			# assume scalar multiplication
			ret = self
			for midx1, col in ret.cols.items():
				for midx2, val in col.coeffs.items():
					ret[midx1][midx2] *= other
			return ret
		if not self.n0 == other.n1:
			raise TypeError('Dimension mismatch')
		ret = PolyLinTrans(other.n0, self.n1)
		for midx1, col in other.cols.items():
			for midxk, val1 in other[midx1].coeffs.items():
				for midx2, val2 in self[midxk].coeffs.items():
					ret[midx1][midx2] += val1 * val2

		ret.d0 = other.d0
		ret.d1 = self.d1
		return ret

	def __neg__(self):
		ret = copy.deepcopy(self)
		for midx1, col in ret.cols.items():
			for midx2, val in col.coeffs.items():
				col.coeffs[midx2] = -val
		return ret

	def purge(self):
		""" Remove zeros in representation """
		for midx1, col in self.cols.items():
			remove = [k for k,v in col.coeffs.items() if v == 0.]
			for k in remove: del col.coeffs[k]
		remove = [k for k,v in self.cols.items() if len(v.coeffs) == 0]
		for k in remove: del self.cols[k]

	def updated(self):
		self.d0 = 0
		self.d1 = 0
		for key1, col in self.cols.items():
			for key2, val in col.coeffs.items():
				self.d1 = max(self.d1, sum(key2))
			self.d0 = max(self.d0, sum(key1))

	def as_Tcc(self):
		""" 
			Return a grlex-ordered representation A of the transformation.

			That is, if p is a grlex coefficient vector, 
				A p 
			is a grlex coefficient vector of the transformed polynomial.
		"""
		i = []
		j = []
		v = []
		nrow = count_monomials_leq(self.n1, self.d1)
		ncol = count_monomials_leq(self.n0, self.d0)
		for midx1, col in self.cols.items():
			idx1 = grlex_to_index(midx1)
			for midx2, val in col.coeffs.items():
				i.append(grlex_to_index(midx2))
				j.append(idx1)
				v.append(val)
		return sp.coo_matrix( (v, (i, j)), shape = (nrow, ncol) )

	def as_Tcg(self):
		""" Return a representation A of the transformation from a vector
			v representing a gram matrix S = mat(v).

			That is, if p contains the coefficients of a gram matrix, 
				A p 
			is a grlex coefficient vector of the transformed polynomial Trans( x^T S x ).
		"""
		half_deg = int(ceil(float(self.d0)/2))
		num_mon = count_monomials_leq(self.n0, half_deg)
		len_vec = num_mon*(num_mon+1)//2  # py3 integer division
		coefs, exps = vec_to_grlex(len_vec, self.n0)
		
		i = []
		j = []
		v = []

		for k, (coef,midx_mid) in enumerate(zip(coefs, exps)):
			for midx_f, val in self[midx_mid].coeffs.items():
				i.append( grlex_to_index(midx_f) )
				j.append( k )
				v.append( val * coef )

		nrow = count_monomials_leq(self.n1, self.d1)
		ncol = len_vec
		return scipy.sparse.coo_matrix( (v, (i, j)), shape = (nrow, ncol) )


