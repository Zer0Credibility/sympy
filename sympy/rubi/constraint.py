from matchpy import Constraint, substitute
from sympy import sympify
from sympy.logic.boolalg import BooleanTrue
from .matchpy2sympy import matchpy2sympy

class cons(Constraint):
    def __init__(self, expr, vars):
        self.expr = expr
        self.vars = frozenset(v.name for v in vars)

    def __call__(self, substitution):
        if isinstance(self.expr, bool): #handle rules without constraints
            return self.expr

        sub = substitute(self.expr, substitution)
        res = matchpy2sympy(sub)

        if isinstance(res, BooleanTrue) or res == True:
            return True
        else:
            return False

    @property
    def variables(self):
        return self.vars

    def with_renamed_vars(self, renaming):
        if isinstance(self.expr, bool):
            copy = cons(self.expr, [])
        else:
            copy = cons(self.expr.with_renamed_vars(renaming), [])
        copy.vars = frozenset(renaming.get(v, v) for v in self.vars)
        return copy

    def __eq__(self, other):
        return isinstance(other, cons) and other.vars == self.vars and other.expr == self.expr

    def __hash__(self):
        return hash((self.vars, self.expr))


class FreeQ(Constraint):
    def __init__(self, expr, var):
        self.expr = expr if isinstance(expr, str) else expr.name
        self.var = var if isinstance(var, str) else var.name

    def __call__(self, substitution):
        return substitution[self.var] not in substitution[self.expr]

    @property
    def variables(self):
        return frozenset([self.expr, self.var])

    def with_renamed_vars(self, renaming):
        return FreeQ(
            renaming.get(self.expr, self.expr),
            renaming.get(self.var, self.var)
        )

    def __eq__(self, other):
        return isinstance(other, FreeQ) and other.expr == self.expr and other.var == self.var

    def __hash__(self):
        return hash((self.expr, self.var))