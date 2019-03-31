from sympy.matrices.expressions import MatrixExpr
from sympy import MatrixBase, Dummy, Lambda, Function, DiagonalMatrix, DiagonalOf, Basic, FunctionClass
from sympy.matrices.expressions.diagonal import DiagonalizeVector, diagonalize_vector


class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy.matrices.expressions import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    exp(X...)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    exp(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])...)
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """

    def __new__(cls, function, expr):
        obj = MatrixExpr.__new__(cls, expr)
        if not isinstance(function, FunctionClass):
            d = Dummy("d")
            function = Lambda(d, function(d))
        obj._function = function
        obj._expr = expr
        return obj

    def _hashable_content(self):
        return (self.function, self.expr)

    @property
    def function(self):
        return self._function

    @property
    def expr(self):
        return self._expr

    @property
    def shape(self):
        return self.expr.shape

    def func(self, expr):
        return ElementwiseApplyFunction(self.function, expr)

    def doit(self, **kwargs):
        deep = kwargs.get("deep", True)
        expr = self.expr
        if deep:
            expr = expr.doit(**kwargs)
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        else:
            return self

    def as_explicit(self):
        raise NotImplementedError

    def _entry(self, i, j, **kwargs):
        return self.function(self.expr[i, j])

    def _eval_derivative_matrix_lines(self, x):
        from sympy import HadamardProduct, hadamard_product, Mul, MatMul, Identity

        d = Dummy("d")
        function = self.function(d)
        fdiff = function.fdiff()
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        lr = self.expr._eval_derivative_matrix_lines(x)
        ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
        if 1 in x.shape:
            # Vector:
            iscolumn = self.shape[1] == 1
            ewdiff = diagonalize_vector(ewdiff)
            # TODO: check which axis is not 1
            for i in lr:
                if iscolumn:
                    ptr1 = [i.first_pointer]
                    ptr2 = [Identity(ewdiff.shape[0])]
                else:
                    ptr1 = [Identity(ewdiff.shape[1])]
                    ptr2 = [i.second_pointer]
                #i.first_pointer = ptr1  # [DiagonalizeVector, ptr1]
                #i.second_pointer = ptr2
                #i._first_pointer_parent
                # TODO: check if pointers point to two different lines:

                # Unify lines:
                #if iscolumn:
                #l = [j[0] for j in i._lines]
                lines = i._lines

                def mul(*args):
                    return Mul.fromiter(args)

                def hadamard_or_mul(arg1, arg2):
                    if arg1.shape == arg2.shape:
                        return hadamard_product(arg1, arg2)
                    elif arg1.shape[1] == arg2.shape[0]:
                        return MatMul(arg1, arg2).doit()
                    elif arg1.shape[0] == arg2.shape[0]:
                        return MatMul(arg2.T, arg1).doit()
                    raise NotImplementedError

                #i._lines = [[hadamard_product, [[mul, [ewdiff, l[0]]], ptr2[0]]]]
                i._lines = [[hadamard_or_mul, [[mul, [ewdiff, ptr1[0]]], ptr2[0]]]]
                i._first_pointer_parent = i._lines[0][1][0][1]
                i._first_pointer_index = 1
                i._second_pointer_parent = i._lines[0][1]
                i._second_pointer_index = 1

        else:
            # Matrix case:
            for i in lr:
                ptr1 = [i.first_pointer]
                ptr2 = [i.second_pointer]
                subexpr1 = [DiagonalizeVector, ptr1]
                subexpr2 = [DiagonalizeVector, ptr2]
                i.first_pointer = subexpr1
                i.second_pointer = subexpr2
                i._first_pointer_parent = subexpr1[1]
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr2[1]
                i._second_pointer_index = 0
                # TODO: check if pointers point to two different lines:

                # Unify lines:
                l = i._lines
                i._lines = [[MatMul, [l[0], ewdiff, l[1]]]]
        return lr
