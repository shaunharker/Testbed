

def Extend(f):
    def F(x):
        try:
            return f(x)
        except:
            return None
    return F

class Type:
    """
    A Type T is a pair of pure functions (c, d) where:

    c: U -> U, c(None) = None
    d: U -> U, d(None) = None

    For all x, d(c(d(x))) == d(x).
    For all x, c(d(c(x))) == c(x)
    """
    def __init__(self, constructor, deconstructor):
        self.c = Extend(constructor)
        self.d = Extend(deconstructor)

    def __call__(self, x):
        y = self.c(x)
        assert y in self
        return y

    def __contains__(self, y):
        return self.c(self.d(y)) == y


Identity = Type(constructor=lambda x: x,
                deconstructor=lambda x: x) # Viewed as a function

Universe = Identity # Viewed as a type

class Function:
    def __init__(self, function, domain=Universe, codomain=Universe):
        self.domain = domain
        self.codomain = codomain
        def F(x):
            assert x in domain
            y = function(x)
            assert y in codomain
            return y
        self.function = Extend(F)

    def __call__(self, x):
        return self.function(x)

Recognized = (
    lambda f:
        Type(constructor=lambda x: x,
             deconstructor=lambda x: x if f(x) else None))

Builtin = (
    lambda T:
        Type(constructor=lambda x: T(x),
             deconstructor=lambda x: x))

def Assert(x):
    assert x

String = Builtin(str)

Integer = Builtin(int)

List = Builtin(list)

Dict = Builtin(dict)

TypedList = (
    lambda T:
        Type(constructor=List.constructor,
             deconstructor=Function(
                function=lambda x: x,
                domain=Recognized(
                    lambda L:
                        L in List and
                        all(x in T
                            for x in L)))

MinDict = (
    lambda minimum:
        Type(constructor=Dict.constructor,
             deconstructor=Function(
                function=lambda x: x,
                domain=Recognized(
                    lambda D:
                        D in Dict and
                        all(k in D and
                            D[k] in v
                            for (k,v) in minimum.items())))

MaxDict = (
    lambda maximum:
        Type(constructor=Dict.constructor,
             deconstructor=Function(
                function=lambda x: x,
                domain=Recognized(
                    lambda D:
                        D in Dict and
                        all(k not in D or
                            D[k] in v
                            for (k,v) in maximum.items())))


TypedDict = (
    lambda required, optional={}:
        Type(constructor=Dict.constructor,
             deconstructor=Function(
                function=lambda x: x,
                domain=Recognized(
                    lambda D:
                        D in MinDict(required) and
                        D in MaxDict({**required,
                                      **optional})))))

StringLiteral = (
    lambda *choices:
        Type(constructor=String.constructor,
             deconstructor=Function(
                function=lambda x: x,
                domain=Recognized(
                    lambda s:
                        s in String and
                        s in choices))

Boolean = StringLiteral("true", "false")

# A proof system.

DependentPair = (
    lambda A, B:

)
