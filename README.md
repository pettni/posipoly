# posipoly

Collection of methods to efficiently create optimization problems over positive polynomials using Mosek as the underlying solver.

**This toolbox is in an early development phase. Use at own risk; syntax will be unstable and bugs are likely plenty.**

## Getting started

The commercial solver Mosek (free licenses for academic use) is currently the only supported solver. Install Mosek and set up the Python interface according to [instructions](https://docs.mosek.com/8.1/pythonapi/install-interface.html#pip-and-wheels). Make sure that ```python -c 'import mosek'``` runs without errors. Then proceed to install the package.

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

 - Efficient representation of linear polynomial transformations via the PolyLinTrans class. As opposed to e.g. [SOSTOOLS](http://www.cds.caltech.edu/sostools/), a symbolic engine is not used when setting up a problem. Instead constraints are defined directly with linear operators represented in parsimonious sparse form, which hopefully avoids computational overhead (magnitude of savings is TBD).
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
mat_to_vec(C) = [C11 C12 ... C1n C22 ... C2n ... Cnn] 
```
of length n(n+1)/2. We call this a **g-format** representation
.

### Retrieve constraints in coefficient form

A linear polynomial transformation is a mapping between polynomial rings such that the transformed coefficients are linear in the original coefficients. Examples include the identity transform (possibly between different dimensions and degrees), differentiation, multiplication with a given polynomial, etc. Many such transformations are implemented in `PTrans` as static member methods. Furthermore, methods to provide sparse (`scipy.sparse.coo_matrix`) transformation matrices of two types are available: gram to coefficient, and coefficient to coefficient.
```
trans = PTrans.eye(2,2)

# get A such that for a gram polynomial Z(x)' * S * Z(x), transformed polynomial is (A * vec(S))' * Z(x)
A = trans.Acg  # g-format to c-format
# get A such that for a coefficient polynomial c' * Z(x), transformed polynomial is (A * c)' * Z(x)
A = trans.Acc  # c-format to c-format
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
 2. Add constraints represented by linear transformations of the coefficient representations via `PTrans` objects. Degrees and dimensions must add up here, i.e. if `var0` is in n variables and of degree d, then `T00.d0 = d, T00.n0 = n`. Furthermore, all transformations in a row must have the same target dimension and degree.
```
# Add constraint T00.var0 + T01.var1 = b0
prob.add_row({'var0': T00, 'var1': T01}, b0, 'eq')
# Add constraint T10.var0 <= b1
prob.add_row({'var0': T10}, b1, 'iq')
```
 3. Set objective
```
# Set objective to min c' var0
prob.set_objective({'var0': c})
```
 4. Optimize while imposing cone constraints on all `pp`-type variables 
```
sol, sta = prob.solve('psd')  # or 'sdd'
```

### Define and solve a PPP problem manually

The `solve_ppp` function imposes positivity constraints on gram matrices that are represented in vector form. Let `vec_to_mat` be the inverse transformation, i.e. `vec_to_mat(mat_to_vec(C)) = C`. These two functions are available in `posipoly.utils`

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
The matrices `Aeq` and `beq` can be obtained from `PTrans` objects via the `Acg` and `Acc` properties to obtain matrix representations of the transformations from g-format to c-format, and from c-format to c-format, respectively.

## Future list

 - Profile the grlex code. Potentially switch to grevlex and/or use Cython to make this part faster
 - Implement class for sparsity patterns and enable sparse optimization
 - Interface with other solvers than Mosek (or interface with a package like cvxpy)

## Research questions

 - Is it possible to restrict the sparsity pattern of multiplier polynomials without loss of generality?
