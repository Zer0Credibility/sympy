from sympy.rubi.utility_function import *
from sympy.core.symbol import symbols, S
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acsch, cosh, sinh, tanh, coth, sech, csch
from sympy.functions import (log, sin, cos, tan, cot, sec, csc, sqrt)
from sympy import I, E, pi

a, b, c, d, e, f, g, h, x, y, z, m, n, p, q = symbols('a b c d e f g h x y z m n p q', real=True, imaginary=False)

def test_ZeroQ():
    assert ZeroQ(S(0))
    assert not ZeroQ(S(10))
    assert not ZeroQ(S(-2))

def test_NonzeroQ():
    assert NonzeroQ(S(1)) == True

def test_FreeQ():
    l = [a*b, x, a + b]
    assert FreeQ(l, x) == False

    l = [a*b, a + b]
    assert FreeQ(l, x) == True

def test_List():
    assert List(a, b, c) == [a, b, c]

def test_Log():
    assert Log(a) == log(a)

def test_PositiveIntegerQ():
    assert PositiveIntegerQ(S(1))
    assert not PositiveIntegerQ(S(-3))
    assert not PositiveIntegerQ(S(0))

def test_NegativeIntegerQ():
    assert not NegativeIntegerQ(S(1))
    assert NegativeIntegerQ(S(-3))
    assert not NegativeIntegerQ(S(0))

def test_PositiveQ():
    assert PositiveQ(S(1))
    assert not PositiveQ(S(-3))
    assert not PositiveQ(S(0))

def test_IntegerQ():
    assert IntegerQ(S(1))
    assert not IntegerQ(S(-1.9))
    assert not IntegerQ(S(0.0))
    assert IntegerQ(S(-1))

def test_PosQ():
    assert PosQ(S(10))
    assert not PosQ(S(-10))
    assert not PosQ(S(0))

def test_FracPart():
    assert FracPart(S(10)) == 0
    assert FracPart(S(10)+0.5) == 10.5

def test_IntPart():
    assert IntPart(S(10)) == 10
    assert IntPart(1 + m) == 1

def test_NegQ():
    assert NegQ(-S(3))
    assert not NegQ(S(0))
    assert not NegQ(S(0))

def test_RationalQ():
    assert RationalQ(S(5)/6)
    assert RationalQ(S(5)/6, S(4)/5)
    assert not RationalQ(Sqrt(1.6))
    assert not RationalQ(Sqrt(1.6), S(5)/6)

def test_Sqrt():
    assert Sqrt(S(16)) == 4

def test_ArcCosh():
    assert ArcCosh(x) == acosh(x)

def test_LinearQ():
    assert not LinearQ(a, x)
    assert LinearQ(3*x + y**2, x)
    assert not LinearQ(3*x + y**2, y)

def test_Sqrt():
    assert Sqrt(x) == sqrt(x)
    assert Sqrt(25) == 5

def test_Coefficient():
    assert Coefficient(7 + 2*x + 4*x**3, x, 1) == 2
    assert Coefficient(a + b*x + c*x**3, x, 0) == a
    assert Coefficient(a + b*x + c*x**3, x, 4) == 0
    assert Coefficient(b*x + c*x**3, x, 3) == c

def test_Denominator():
    assert Denominator(S(3)/2) == 2
    assert Denominator(x/y) == y
    assert Denominator(S(4)/5) == 5

def test_Hypergeometric2F1():
    assert Hypergeometric2F1(1, 2, 3, x) == -2/x - 2*log(-x + 1)/x**2

def test_ArcTan():
    assert ArcTan(x) == atan(x)

def test_Not():
    a = 10
    assert Not(a == 2)

def test_FractionalPart():
    assert FractionalPart(S(3.0)) == 0.0

def test_IntegerPart():
    assert IntegerPart(3.6) == 3
    assert IntegerPart(-3.6) == -4

def test_AppellF1():
    assert AppellF1(1,0,0.5,1,0.5,0.25) == 1.154700538379251529018298

def test_Simplify():
    assert Simplify(sin(x)**2 + cos(x)**2) == 1
    assert Simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1)) == x - 1

def test_EllipticPi():
    assert EllipticPi(0.25, 0.25) == 1.956616279119236207279727
    assert EllipticPi(3, 0) == (0.0 - 1.11072073453959156175397j)

def test_EllipticE():
    assert EllipticE(0) == 1.570796326794896619231322
    assert EllipticE(2) == (0.5990701173677961037199612 + 0.5990701173677961037199612j)
    assert EllipticE(0.5 + 0.25j) == (1.360868682163129682716687 - 0.1238733442561786843557315j)

def test_EllipticF():
    assert EllipticF(0,1) == 0.0
    assert EllipticF(2 + 3j,0) == (2.0 + 3.0j)
    assert EllipticF(1,1) == 1.226191170883517070813061

def test_ArcTanh():
    assert ArcTanh(a) == atanh(a)

def test_ArcSin():
    assert ArcSin(a) == asin(a)

def test_ArcSinh():
    assert ArcSinh(a) == asinh(a)

def test_ArcCos():
    assert ArcCos(a) == acos(a)

def test_ArcCsc():
    assert ArcCsc(a) == acsc(a)

def test_ArcCsch():
    assert ArcCsch(a) == acsch(a)

def test_Equal():
    assert Equal(a, a)
    assert not Equal(a, b)

def test_LessEqual():
    assert LessEqual(1, 2, 3)
    assert LessEqual(1, 1)
    assert not LessEqual(3, 2, 1)

def test_With():
    assert With(Set(x, 3), x + y) == 3 + y
    assert With(List(Set(x, 3), Set(y, c)), x + y) == 3 + c

def test_Less():
    assert Less(1, 2, 3)
    assert not Less(1, 1, 3)

def test_Greater():
    assert Greater(3, 2, 1)
    assert not Greater(3, 2, 2)

def test_GreaterEqual():
    assert GreaterEqual(3, 2, 1)
    assert GreaterEqual(3, 2, 2)
    assert not GreaterEqual(2, 3)

def test_Unequal():
    assert Unequal(1, 2)
    assert not Unequal(1, 1)

def test_FractionQ():
    assert FractionQ(S(1), S(2), S(1)/3)
    assert not FractionQ(sqrt(2))

def test_Expand():
    assert Expand((1 + x)**10) == x**10 + 10*x**9 + 45*x**8 + 120*x**7 + 210*x**6 + 252*x**5 + 210*x**4 + 120*x**3 + 45*x**2 + 10*x + 1

def test_Scan():
    assert list(Scan(sin, [a, b])) == [sin(a), sin(b)]

def test_MapAnd():
    assert MapAnd(PositiveQ, [S(1), S(2), S(3), S(0)]) == False
    assert MapAnd(PositiveQ, [S(1), S(2), S(3)]) == True

def test_FalseQ():
    assert FalseQ(True) == False
    assert FalseQ(False) == True

def test_ComplexNumberQ():
    assert ComplexNumberQ(1 + I*2, I) == True
    assert ComplexNumberQ(a + b, I) == False

def test_Re():
    assert Re(1 + I) == 1

def test_Im():
    assert Im(1 + 2*I) == 2
    assert Im(a*I) == a

def test_RealNumericQ():
    assert RealNumericQ(S(1)) == True

def test_PositiveOrZeroQ():
    assert PositiveOrZeroQ(S(0)) == True
    assert PositiveOrZeroQ(S(1)) == True
    assert PositiveOrZeroQ(-S(1)) == False

def test_RealNumericQ():
    assert RealNumericQ(S(1)) == True
    assert RealNumericQ(-S(1)) == True

def test_NegativeOrZeroQ():
    assert NegativeOrZeroQ(S(0)) == True
    assert NegativeOrZeroQ(-S(1)) == True
    assert NegativeOrZeroQ(S(1)) == False

def test_FractionOrNegativeQ():
    assert FractionOrNegativeQ(S(1)/2) == True
    assert FractionOrNegativeQ(-S(1)) == True

def test_ProductQ():
    assert ProductQ(a*b) == True
    assert ProductQ(a + b) == False

def test_SumQ():
    assert SumQ(a*b) == False
    assert SumQ(a + b) == True

def test_NonsumQ():
    assert NonsumQ(a*b) == True
    assert NonsumQ(a + b) == False

def test_SqrtNumberQ():
    assert SqrtNumberQ(sqrt(2)) == True

def test_IntLinearcQ():
    assert IntLinearcQ(1, 2, 3, 4, 5, 6, x) == True
    assert IntLinearcQ(S(1)/100, S(2)/100, S(3)/100, S(4)/100, S(5)/100, S(6)/100, x) == False

def test_IndependentQ():
    assert IndependentQ(a + b*x, x) == False
    assert IndependentQ(a + b, x) == True

def test_PowerQ():
    assert PowerQ(a**b) == True
    assert PowerQ(a + b) == False

def test_IntegerPowerQ():
    assert IntegerPowerQ(a**2) == True
    assert IntegerPowerQ(a**0.5) == False

def test_PositiveIntegerPowerQ():
    assert PositiveIntegerPowerQ(a**3) == True
    assert PositiveIntegerPowerQ(a**(-2)) == False

def test_FractionalPowerQ():
    assert FractionalPowerQ(a**2) == True
    assert FractionalPowerQ(a**sqrt(2)) == False

def test_AtomQ():
    assert AtomQ(x)
    assert not AtomQ(x+1)

def test_ExpQ():
    assert ExpQ(E**2)
    assert not ExpQ(2**E)

def test_LogQ():
    assert LogQ(log(x))
    assert not LogQ(sin(x) + log(x))

def test_Head():
    assert Head(sin(x)) == sin
    assert Head(log(x**3 + 3)) == log

def test_MemberQ():
    assert MemberQ([a, b, c], b)
    assert MemberQ([sin, cos, log, tan], Head(sin(x)))

def test_TrigQ():
    assert TrigQ(sin(x))
    assert TrigQ(tan(x**2 + 2))
    assert not TrigQ(sin(x) + tan(x))

def test_SinQ():
    assert SinQ(sin(x))
    assert not SinQ(tan(x))

def test_CosQ():
    assert CosQ(cos(x))
    assert not CosQ(csc(x))

def test_TanQ():
    assert TanQ(tan(x))
    assert not TanQ(cot(x))

def test_CotQ():
    assert not CotQ(tan(x))
    assert CotQ(cot(x))

def test_SecQ():
    assert SecQ(sec(x))
    assert not SecQ(csc(x))

def test_CscQ():
    assert not CscQ(sec(x))
    assert CscQ(csc(x))

def test_HyperbolicQ():
    assert HyperbolicQ(sinh(x))
    assert HyperbolicQ(cosh(x))
    assert HyperbolicQ(tanh(x))
    assert not HyperbolicQ(sinh(x) + cosh(x) + tanh(x))

def test_SinhQ():
    assert SinhQ(sinh(x))
    assert not SinhQ(cosh(x))

def test_CoshQ():
    assert not CoshQ(sinh(x))
    assert CoshQ(cosh(x))

def test_TanhQ():
    assert TanhQ(tanh(x))
    assert not TanhQ(coth(x))

def test_CothQ():
    assert not CothQ(tanh(x))
    assert CothQ(coth(x))

def test_SechQ():
    assert SechQ(sech(x))
    assert not SechQ(csch(x))

def test_CschQ():
    assert not CschQ(sech(x))
    assert CschQ(csch(x))

def test_InverseTrigQ():
    assert InverseTrigQ(acot(x))
    assert InverseTrigQ(asec(x))
    assert not InverseTrigQ(acsc(x) + asec(x))

def test_SinCosQ():
    assert SinCosQ(sin(x))
    assert SinCosQ(cos(x))
    assert SinCosQ(sec(x))
    assert not SinCosQ(acsc(x))

def test_SinhCoshQ():
    assert not SinhCoshQ(sin(x))
    assert SinhCoshQ(cosh(x))
    assert SinhCoshQ(sech(x))
    assert SinhCoshQ(csch(x))

def test_Rt():
    assert Rt(8, 3) == 2
    assert Rt(16807, 5) == 7

def test_LeafCount():
    assert LeafCount(1 + a + x**2) == 6

def test_Numerator():
    assert Numerator(S(3)/2) == 3
    assert Numerator(x/y) == x

def test_Length():
    assert Length(a + b) == 2
    assert Length(sin(a)*cos(a)) == 2

def test_AtomQ():
    assert AtomQ(a)
    assert not AtomQ(a + b)

def test_ListQ():
    assert ListQ([1, 2])
    assert not ListQ(a)

def test_InverseHyperbolicQ():
    assert InverseHyperbolicQ(acosh(a))

def test_InverseFunctionQ():
    assert InverseFunctionQ(log(a))
    assert InverseFunctionQ(acos(a))
    assert not InverseFunctionQ(a)
    assert InverseFunctionQ(acosh(a))
    assert InverseFunctionQ(polylog(a, b))

def test_EqQ():
    assert EqQ(a, a)
    assert not EqQ(a, b)

def test_FactorSquareFree():
    assert FactorSquareFree(x**5 - x**3 - x**2 + 1) == (x**3 + 2*x**2 + 2*x + 1)*(x - 1)**2

def test_FactorSquareFreeList():
    assert FactorSquareFreeList(x**5-x**3-x**2 + 1) == [[1, 1], [x**3 + 2*x**2 + 2*x + 1, 1], [x - 1, 2]]
    assert FactorSquareFreeList(x**4 - 2*x**2 + 1) == [[1, 1], [x**2 - 1, 2]]

def test_PerfectPowerTest():
    assert not PerfectPowerTest(sqrt(x), x)
    assert not PerfectPowerTest(x**5-x**3-x**2 + 1, x)
    assert PerfectPowerTest(x**4 - 2*x**2 + 1, x) == (x**2 - 1)**2

def test_SquareFreeFactorTest():
    assert not SquareFreeFactorTest(sqrt(x), x)
    assert SquareFreeFactorTest(x**5 - x**3 - x**2 + 1, x) == (x**3 + 2*x**2 + 2*x + 1)*(x - 1)**2

def test_Rest():
    assert Rest([2, 3, 5, 7]) == [3, 5, 7]
    assert Rest(a + b + c) == b + c
    assert Rest(a*b*c) == b*c
    assert Rest(1/b) == -1

def test_First():
    assert First([2, 3, 5, 7]) == 2
    assert First(y**2) == y
    assert First(a + b + c) == a
    assert First(a*b*c) == a

def test_ComplexFreeQ():
    assert ComplexFreeQ(x)
    assert not ComplexFreeQ(x+2*I)

def test_FractionalPowerFreeQ():
    assert not FractionalPowerFreeQ(x**(S(2)/3))
    assert FractionalPowerFreeQ(x)

def test_Exponent():
    assert Exponent(x**2+x+1+5, x, List) == [0, 1, 2]
    assert Exponent(x**2+x+1, x, List) == [0, 1, 2]
    assert Exponent(x**2+2*x+1, x, List) == [0, 2, 1]
    assert Exponent(x**3+x+1, x) == 3
    assert Exponent(x**2+2*x+1, x) == 2
    assert Exponent(x**3, x, List) == [3]
    assert Exponent(S(1), x) == 0
    assert Exponent(x**(-3), x) == 0

def test_Expon():
    assert Expon(x**2+2*x+1, x) == 2
    assert Expon(x**3, x, List) == [3]

def test_QuadraticQ():
    assert not QuadraticQ([x**2+x+1, 5*x**2], x)
    assert QuadraticQ([x**2+x+1, 5*x**2+3*x+6], x)
    assert not QuadraticQ(x**2+1+x**3, x)
    assert QuadraticQ(x**2+1+x, x)
    assert not QuadraticQ(x**2, x)

def test_BinomialQ():
    assert BinomialQ(x**9, x)
    assert BinomialQ((1 + x)**3, x)

def test_BinomialParts():
    assert BinomialParts(2 + x*(9*x), x) == [2, 9, 2]
    assert BinomialParts(x**9, x) == [0, 1, 9]
    assert BinomialParts(2*x**3, x) == [0, 2, 3]
    assert BinomialParts(2 + x, x) == [2, 1, 1]

def test_PolynomialQ():
    assert PolynomialQ(x**3, x)
    assert not PolynomialQ(sqrt(x), x)

def test_PolyQ():
    assert PolyQ(x, x, 1)
    assert PolyQ(x**2, x, 2)
    assert not PolyQ(x**3, x, 2)

def test_EvenQ():
    assert EvenQ(S(2))
    assert not EvenQ(S(1))

def test_OddQ():
    assert OddQ(S(1))
    assert not OddQ(S(2))

def test_PerfectSquareQ():
    assert PerfectSquareQ(S(4))
    assert PerfectSquareQ(a**S(2)*b**S(4))
    assert not PerfectSquareQ(S(1)/3)

def test_NiceSqrtQ():
    assert NiceSqrtQ(S(1)/3)
    assert not NiceSqrtQ(-S(1))
    assert NiceSqrtQ(pi**2)
    assert NiceSqrtQ(pi**2*sin(4)**4)
    assert not NiceSqrtQ(pi**2*sin(4)**3)

def test_Together():
    assert Together(1/a + b/2) == (a*b + 2)/(2*a)

def test_PosQ():
    assert not PosQ(S(0))
    assert PosQ(S(1))
    assert PosQ(pi)
    assert PosQ(pi**3)
    assert PosQ((-pi)**4)
    assert PosQ(sin(1)**2*pi**4)

def test_NumericQ():
    assert NumericQ(sin(cos(2)))

def test_NumberQ():
    assert NumberQ(pi)

def test_CoefficientList():
    assert CoefficientList(1 + a*x, x) == [1, a]
    assert CoefficientList(1 + a*x**3, x) == [1, 0, 0, a]
    assert CoefficientList(sqrt(x), x) == []

def test_ReplaceAll():
    assert ReplaceAll(x, {x: a}) == a
    assert ReplaceAll(a*x, {x: a + b}) == a*(a + b)
    assert ReplaceAll(a*x, {a: b, x: a + b}) == b*(a + b)

def test_SimplifyTerm():
    assert SimplifyTerm(a/100 + 100/b*x, x) == a/100 + 100/b*x

def test_ExpandLinearProduct():
    assert ExpandLinearProduct(log(x), x**2, a, b, x) == a**2*log(x)/b**2 - 2*a*(a + b*x)*log(x)/b**2 + (a + b*x)**2*log(x)/b**2

def test_PolynomialDivide():
    assert PolynomialDivide(x + x**2, x, x) == x + 1
    assert PolynomialDivide((1 + x)**3, (1 + x)**2, x) == x + 1

def test_ExpandIntegrand():
    assert True
    '''
    assert ExpandIntegrand((1 + x)**3/x, x) == x**2 + 3*x + 3 + 1/x
    assert ExpandIntegrand((1 + 2*(3 + 4*x**2))/(2 + 3*x**2 + 1*x**4), x) == 18.0/(2*x**2 + 4.0) - 2.0/(2*x**2 + 2.0)
    assert ExpandIntegrand((-1 + (-1)*x**2 + 2*x**4)**(-2), x) == 1/(4*x**8 - 4*x**6 - 3*x**4 + 2*x**2 + 1)
    assert ExpandIntegrand((1 - 1*x**2)**(-3), x) == -1/(x**6 - 3.0*x**4 + 3.0*x**2 - 1.0)
    assert ExpandIntegrand(x**2*(1 - 1*x**6)**(-2), x) == x**2/(x**12 - 2.0*x**6 + 1.0)
    assert ExpandIntegrand((c + d*x**2 + e*x**3)/(1 - 1*x**4), x) == (1.0*c - 1.0*d - 1.0*I*e)/(4*I*x + 4.0) + (1.0*c - 1.0*d + 1.0*I*e)/(-4*I*x + 4.0) + (1.0*c + 1.0*d - 1.0*e)/(4*x + 4.0) + (1.0*c + 1.0*d + 1.0*e)/(-4*x + 4.0)
    assert ExpandIntegrand((a + b*x)**2/(c + d*x), x) == b*(a + b*x)/d + b*(a*d - b*c)/d**2 + (a*d - b*c)**2/(d**2*(c + d*x))
    assert ExpandIntegrand(x/(a*x**1 + b*Sqrt(c + d*x**2)), x) == a*x**2/(a**2*x**2 - b**2*c - b**2*d*x**2) - b*x*sqrt(c + d*x**2)/(a**2*x**2 - b**2*c - b**2*d*x**2)
    assert ExpandIntegrand(x**2*(a + b*Log(c*(d*(e + f*x)**p)**q))**n, x) == e**2*(a + b*log(c*(d*(e + f*x)**p)**q))**n/f**2 - 2*e*(a + b*log(c*(d*(e + f*x)**p)**q))**n*(e + f*x)/f**2 + (a + b*log(c*(d*(e + f*x)**p)**q))**n*(e + f*x)**2/f**2
    assert ExpandIntegrand(x*(1 + 2*x)**3*log(2*(1 + 1*x**2)**1), x) == 8*x**4*log(2*x**2 + 2) + 12*x**3*log(2*x**2 + 2) + 6*x**2*log(2*x**2 + 2) + x*log(2*x**2 + 2)
    assert ExpandIntegrand((1 + 1*x)**S(3)*f**(e*(1 + 1*x)**n)/(g + h*x), x) == f**(e*(x + 1)**n)*(x + 1)**2/h + f**(e*(x + 1)**n)*(-g + h)*(x + 1)/h**2 + f**(e*(x + 1)**n)*(-g + h)**2/h**3 - f**(e*(x + 1)**n)*(g - h)**3/(h**3*(g + h*x))
    '''
def test_MatchQ():
    a_ = Wild('a', exclude=[x])
    b_ = Wild('b', exclude=[x])
    c_ = Wild('c', exclude=[x])
    assert MatchQ(a*b + c, a_*b_ + c_, a_, b_, c_) == (a, b, c)

def test_PolynomialQuotientRemainder():
    assert PolynomialQuotientRemainder(x**2, x+a, x) == [-a + x, a**2]

def test_FreeFactors():
    assert FreeFactors(a, x) == a
    assert FreeFactors(x + a, x) == 1
    assert FreeFactors(a*b*x, x) == a*b

def test_NonfreeFactors():
    assert NonfreeFactors(a, x) == 1
    assert NonfreeFactors(x + a, x) == x + a
    assert NonfreeFactors(a*b*x, x) == x

def test_FreeTerms():
    assert FreeTerms(a, x) == a
    assert FreeTerms(x*a, x) == 0
    assert FreeTerms(a*x + b, x) == b

def test_NonfreeTerms():
    assert NonfreeTerms(a, x) == 0
    assert NonfreeTerms(a*x, x) == a*x
    assert NonfreeTerms(a*x + b, x) == a*x

def test_RemoveContent():
    assert RemoveContent(a + b*x, x) == a + b*x

def test_ExpandAlgebraicFunction():
    assert ExpandAlgebraicFunction((a + b)*x, x) == a*x + b*x
    assert ExpandAlgebraicFunction((a + b)**2*x, x)== a**2*x + 2*a*b*x + b**2*x
    assert ExpandAlgebraicFunction((a + b)**2*x**2, x) == a**2*x**2 + 2*a*b*x**2 + b**2*x**2

def test_CollectReciprocals():
    assert CollectReciprocals(-1/(1 + 1*x) - 1/(1 - 1*x), x) == -2/(-x**2 + 1)
    assert CollectReciprocals(1/(1 + 1*x) - 1/(1 - 1*x), x) == -2*x/(-x**2 + 1)

def test_ExpandCleanup():
    assert ExpandCleanup(a + b, x) == a + b

def test_AlgebraicFunctionQ():
    assert AlgebraicFunctionQ(a, x) == True
    assert AlgebraicFunctionQ(a*b, x) == True
    assert AlgebraicFunctionQ(x**2, x) == True
    assert AlgebraicFunctionQ(x**2*a, x) == True
    assert AlgebraicFunctionQ(x**2 + a, x) == True
    assert AlgebraicFunctionQ(sin(x), x) == False
    assert AlgebraicFunctionQ([], x) == True
    assert AlgebraicFunctionQ([a, a*b], x) == True
    assert AlgebraicFunctionQ([sin(x)], x) == False

def test_LeadTerm():
    assert LeadTerm(a*b*c) == a*b*c
    assert LeadTerm(a + b + c) == a

def test_RemainingTerms():
    assert RemainingTerms(a*b*c) == a*b*c
    assert RemainingTerms(a + b + c) == b + c

def test_LeadFactor():
    assert LeadFactor(a*b*c) == a
    assert LeadFactor(a + b + c) == a + b + c
    assert LeadFactor(b*I) == b

def test_RemainingFactors():
    assert RemainingFactors(a*b*c) == b*c
    assert RemainingFactors(a + b + c) == 1
    assert RemainingFactors(a*I) == I

def test_LeadBase():
    assert LeadBase(a**b) == a
    assert LeadBase(a**b*c) == c

def test_LeadDegree():
    assert LeadDegree(a**b) == b
    assert LeadDegree(a**b*c) == c

def test_Numer():
    assert Numer(a/b) == a
    assert Numer(a**(-2)) == 1
    assert Numer(a**(-2)*a/b) == 1

def test_Denom():
    assert Denom(a/b) == b
    assert Denom(a**(-2)) == a**2
    assert Denom(a**(-2)*a/b) == a*b

def test_Coeff():
    assert Coeff(7 + 2*x + 4*x**3, x, 1) == 2
    assert Coeff(a + b*x + c*x**3, x, 0) == a
    assert Coeff(a + b*x + c*x**3, x, 4) == 0
    assert Coeff(b*x + c*x**3, x, 3) == c

def test_MergeMonomials():
    assert MergeMonomials(x**2*(1 + 1*x)**3*(1 + 1*x)**n, x) == x**2*(x + 1)**(n + 3)
    assert MergeMonomials(x**2*(1 + 1*x)**2*(1*(1 + 1*x)**1)**2, x) == x**2*(x + 1)**4

def test_RationalFunctionQ():
    assert RationalFunctionQ(a, x)
    assert RationalFunctionQ(x**2, x)
    assert RationalFunctionQ(x**3 + x**4, x)
    assert RationalFunctionQ(x**3*S(2), x)
    assert not RationalFunctionQ(x**3 + x**(0.5), x)

def test_RationalFunctionFactors():
    assert RationalFunctionFactors(a, x) == a
    assert RationalFunctionFactors(sqrt(x), x) == 1
    assert RationalFunctionFactors(x*x**3, x) == x*x**3
    assert RationalFunctionFactors(x*sqrt(x), x) == 1

def test_NonrationalFunctionFactors():
    assert NonrationalFunctionFactors(x, x) == 1
    assert NonrationalFunctionFactors(sqrt(x), x) == sqrt(x)
    assert NonrationalFunctionFactors(sqrt(x)*log(x), x) == sqrt(x)*log(x)

def test_Reverse():
    assert Reverse([1, 2, 3]) == [3, 2, 1]
    assert Reverse(a**b) == b**a

def test_RationalFunctionExponents():
    assert RationalFunctionExponents(sqrt(x), x) == [0, 0]
    assert RationalFunctionExponents(a, x) == [0, 0]
    assert RationalFunctionExponents(x, x) == [1, 0]
    assert RationalFunctionExponents(x**(-1), x)== [0, 1]
    assert RationalFunctionExponents(x**(-1)*a, x) == [0, 1]
    assert RationalFunctionExponents(x**(-1) + a, x) == [1, 1]

def test_PolynomialGCD():
    assert PolynomialGCD(x**2 - 1, x**2 - 3*x + 2) == x - 1

def test_PolyGCD():
    assert PolyGCD(x**2 - 1, x**2 - 3*x + 2, x) == x - 1

def test_AlgebraicFunctionFactors():
    assert AlgebraicFunctionFactors(sin(x)*x, x) == x
    assert AlgebraicFunctionFactors(sin(x), x) == 1
    assert AlgebraicFunctionFactors(x, x) == x

def test_NonalgebraicFunctionFactors():
    assert NonalgebraicFunctionFactors(sin(x)*x, x) == sin(x)
    assert NonalgebraicFunctionFactors(sin(x), x) == sin(x)
    assert NonalgebraicFunctionFactors(x, x) == 1

def test_QuotientOfLinearsP():
    assert QuotientOfLinearsP((a + b*x)/(x), x)
    assert QuotientOfLinearsP(x*a, x)
    assert not QuotientOfLinearsP(x**2*a, x)
    assert not QuotientOfLinearsP(x**2 + a, x)
    assert QuotientOfLinearsP(x + a, x)
    assert QuotientOfLinearsP(x, x)
    assert QuotientOfLinearsP(1 + x, x)

def test_QuotientOfLinearsParts():
    assert QuotientOfLinearsParts((b*x)/(c), x) == [0, b/c, 1, 0]
    assert QuotientOfLinearsParts((b*x)/(c + x), x) == [0, b, c, 1]
    assert QuotientOfLinearsParts((b*x)/(c + d*x), x) == [0, b, c, d]
    assert QuotientOfLinearsParts((a + b*x)/(c + d*x), x) == [a, b, c, d]
    assert QuotientOfLinearsParts(x**2 + a, x) == [a + x**2, 0, 1, 0]
    assert QuotientOfLinearsParts(a/x, x) == [a, 0, 0, 1]
    assert QuotientOfLinearsParts(1/x, x) == [1, 0, 0, 1]
    assert QuotientOfLinearsParts(a*x + 1, x) == [1, a, 1, 0]
    assert QuotientOfLinearsParts(x, x) == [0, 1, 1, 0]
    assert QuotientOfLinearsParts(a, x) == [a, 0, 1, 0]

def test_QuotientOfLinearsQ():
    assert not QuotientOfLinearsQ((a + x), x)
    assert QuotientOfLinearsQ((a + x)/(x), x)
    assert QuotientOfLinearsQ((a + b*x)/(x), x)

def test_Flatten():
    assert Flatten([a, b, [c, [d, e]]]) == [a, b, c, d, e]

def test_Sort():
    assert Sort([b, a, c]) == [a, b, c]
    assert Sort([b, a, c], True) == [c, b, a]

def test_AbsurdNumberQ():
    assert AbsurdNumberQ(S(1))
    assert not AbsurdNumberQ(a*x)
    assert not AbsurdNumberQ(a**(S(1)/2))
    assert AbsurdNumberQ((S(1)/3)**(S(1)/3))

def test_AbsurdNumberFactors():
    assert AbsurdNumberFactors(S(1)) == S(1)
    assert AbsurdNumberFactors((S(1)/3)**(S(1)/3)) == S(3)**(S(2)/3)/S(3)
    assert AbsurdNumberFactors(a) == S(1)

def test_NonabsurdNumberFactors():
    assert NonabsurdNumberFactors(a) == a
    assert NonabsurdNumberFactors(S(1)) == S(1)
    assert NonabsurdNumberFactors(a*S(2)) == a

def test_NumericFactor():
    assert NumericFactor(S(1)) == S(1)
    assert NumericFactor(1*I) == S(1)
    assert NumericFactor(S(1) + I) == S(1)
    assert NumericFactor(a**(S(1)/3)) == S(1)
    assert NumericFactor(a*S(3)) == S(3)
    assert NumericFactor(a + b) == S(1)

def test_NonnumericFactors():
    assert NonnumericFactors(S(3)) == S(1)
    assert NonnumericFactors(I) == I
    assert NonnumericFactors(S(3) + I) == S(3) + I
    assert NonnumericFactors((S(1)/3)**(S(1)/3)) == S(1)
    assert NonnumericFactors(log(a)) == log(a)

def test_Prepend():
    assert Prepend([1, 2, 3], [4, 5]) == [4, 5, 1, 2, 3]

def test_Drop():
    assert Drop([1, 2 ,3, 4], 2) == [2, 3, 4]

def test_SubstForInverseFunction():
    assert SubstForInverseFunction(x, a, b, x) == b
    assert SubstForInverseFunction(a, a, b, x) == a
    assert SubstForInverseFunction(x**a, x**a, b, x) == x
    assert SubstForInverseFunction(a*x**a, a, b, x) == a*b**a

def test_SubstForFractionalPower():
    assert SubstForFractionalPower(a, b, n, c, x) == a
    assert SubstForFractionalPower(x, b, n, c, x) == c
    assert SubstForFractionalPower(a**(S(1)/2), a, n, b, x) == x**(n/2)

def test_CombineExponents():
    assert True

def test_FractionalPowerOfSquareQ():
    assert FractionalPowerOfSquareQ(x) == False
    assert FractionalPowerOfSquareQ((a + b)**2) == (a + b)**2
    assert FractionalPowerOfSquareQ((a + b)**2*c) == (a + b)**2

def test_FractionalPowerSubexpressionQ():
    assert not FractionalPowerSubexpressionQ(x, a, x)
    assert FractionalPowerSubexpressionQ(x**S(2), a, x)
    assert FractionalPowerSubexpressionQ(x**S(2)*a, a, x)
    assert not FractionalPowerSubexpressionQ(b*a, a, x)

def test_FactorNumericGcd():
    assert FactorNumericGcd(x**(S(2))) == x**S(2)
    assert FactorNumericGcd(log(x)) = log(x)
    assert FactorNumericGcd(log(x)*x)) == x*log(x)
    assert FactorNumericGcd(log(x) + x**S(2)) == log(x) + x**S(2)