# posipoly

Collection of methods to efficiently create optimization problems over positive polynomials using Mosek as the underlying solver.

## Installation

```
pip install -r requirements.txt
python setup.py install   # s/install/develop to install `locally`
```

Run tests

```
nosetests
```

Look at an example in `examples/` to see how to do optimization.

## Features

 - Efficient representation of linear polynomial transformations via the PolyLinTrans class.
 - Unified interface for setting up and solving positive polynomial programming (ppp) problems. Currently supports optimization in the PSD (for SOS) and SDD (for SDSOS) cones.

## Math background and code overview

There are two ways to represent polynomial variables in an optimization problem: gram matrix representation and coefficient vector representation. Let `Z(x)` be a vector of monomials of to some given maximal degree, then a polynomial of the form `Z(x)' * C * Z(x)`, where `C` is a symmetric matrix, is in **gram matrix representation**. A polynomial of the form `c' * Z(x)` where `c` is a vector of coefficients, is in **coefficient vector representation**, and we call `c` a **c-format** representation. Positivity constraints are imposed on the gram matrix, so positive variables should be defined in gram matrix representation. In particular, a gram polynomial is SOS if C is positive semi-definite (PSD), it is SDSOS if C is scaled diagonally dominant (SDD), and it is DSOS if C is diagonally dominant (DD).

For polynomial variables that are not positive, coefficient vector representation is preferable since it is more parsimonious than gram representation for a given degree. To avoid redundancy also gram matrices are represented in vector form. For a symmetric matrix
```
C = [C11 C12 ... C1n
     C12 C22 ... C2n
      :   :       :
     C1n C2n ... Cnn]
```
the vector representation is
```
vec(C) = [C11 C12 ... C1n C22 ... C2n ... Cnn] 
```
of length n(n+1)/2. We call this a **g-format** representation
.

### Retrieve constraints in coefficient form

A linear polynomial transformation is a mapping between polynomial rings such that the transformed coefficients are linear in the original coefficients. Examples include the identity transform (possibly between different dimensions and degrees), differentiation, multiplication with a given polynomial, etc. Many such transformations are implemented in `PolyLinTrans` as static member methods. Furthermore, methods to provide sparse (`scipy.sparse.coo_matrix`) transformation matrices of two types are available: gram to coefficient, and coefficient to coefficient.
```
trans = PolyLinTrans.eye(2,2,2,2)

# get T such that for a gram polynomial Z(x)' * S * Z(x), transformed polynomial is (T * vec(S))' * Z(x)
T = trans.as_Tcg()  # g-format to c-format
# get T such that for a coefficient polynomial c' * Z(x), transformed polynomial is (T * c)' * Z(x)
T = trans.as_Tcc()  # c-format to c-format
```

### Define and solve a PPP problem via the `PPP` class

The class `PPP` provides a convenient way to set up and solve a PPP problem.

```
prob = PPP()
```

 1. Add variables to the `PPP` object
```
# add a positive polynomial variable (stored in gram format) in n0 variables and of degree d0
prob.add_var(var0, n0, d0, 'pp')   
# add a polynomial variable (stored in coefficient format) in n1 variables and of degree d1  
prob.add_var(var1, n1, d1, 'coef')   

```
 2. Add constraints
```
# Add constraint trans00.var0 + trans01.var1 = b0
prob.add_row({'var0': trans00, 'var1': trans01}, b0, 'eq')
# Add constraint trans10.var0 + trans11.var1 <= b1
prob.add_row({'var0': trans10, 'var1': trans11}, b1, 'iq')
```
 3. Set objective
```
# Set objective to min c' var0
prob.set_objective({'var0': c})
```
 4. Solve the problem
```
sol, sta = prob.solve()
```

### Define and solve a PPP problem manually

The `solve_ppp` function imposes positivity constraints on gram matrices that are represented in vector form. Let `mat` be the inverse transformation, i.e. `mat(vec(C)) = C`. 

PP constraints are added by specifying that segments of the variable vector represent such matrices. 

```
# General format:
#   min  c' x   
#   s.t. Aeq x = beq
#        Aiq x <= biq
#        mat(x[s,s+l]) in pp_cone for s,l in pp_list
solve_ppp(c, Aeq, beq, Aiq, biq, pp_list, 'psd')   # optimize in the PSD cone
solve_ppp(c, Aeq, beq, Aiq, biq, pp_list, 'sdd')   # same problem in the SDD cone
```
The matrices `Aeq` and `beq` can be obtained from `PolyLinTrans` objects by calling the `as_Tcg()` and `as_Tcc()` member methods to obtain matrix representations of the transformations from g-format to c-format, and from c-format to c-format, respectively.

## TODO list

 - Implement class for sparsity patterns and enable sparse optimization
 - Interface with other solvers than Mosek (or interface with a package like cvxpy)

## Research questions

 - Is it possible to restrict the sparsity pattern of multiplier polynomials without loss of generality?
