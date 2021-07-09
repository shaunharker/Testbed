

def Extend(f):
    def F(x):
        try:
            return f(x)
        except:
            return None
    return F

class Type:
    """
    Let U denote the universe of python objects that
    our type system will be allowed to encounter.

    We admit:

    None, int, str, list, dict

    A Type T is a pair of pure functions (c, d) where:

    c: U -> U
    d: U -> U

    c(None) = None
    d(None) = NoneArgs

    For all x in PythonArgumentSpace, c(d(c(x))) = c(x).
    For all x in PythonObjectSpace, d(c(d(x))) = d(x).

    The function c is known as the constructor and is
    available through the `__call__` method.

    The function d is known as the deconstructor and is
    available through the `deconstruct` method.

    The __contains__ method (which handles expressions of
    the form `x in T`) checks if deconstructing and
    reconstructing gives back an equal object, i.e.

        lambda x: c(d(x)) == x  (*)

    The set of x such that x in T is what we understand to be the set
    of python objects of this type.

    Type creators are expected to have sorted out this guarantee somehow.
    It will be checked whenever d is called.

    One might also be interested in
        lambda x: d(c(x)) == x

    This latter is like the type of the "canonical" constructor arguments.

    lambda x: c(d(c(x))) == c(x) follows from (*)

    We might have a trivial constructor for some use cases.
    We might have a trivial deconstructor for some use cases.
    """
    def __init__(self, docstring, constructor, deconstructor):
        self.c = Extend(constructor)
        self.d = Extend(deconstructor)

    def __call__(self, x):
        y = self.c(x)
        assert y in self
        return y

    def __contains__(self, y):
        return self.c(self.d(y)) == y


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

class Function:
    def __init__(self, f, domain=None, codomain=None):
        self.f = Extend(f)
        self.domain = domain
        self.codomain = codomain
        def F(x):
            if x in
    def __call__(self, x):
        return self.f(x)
        
String = Builtin(str)

Integer = Builtin(int)

List = Builtin(list)

Dict = Builtin(dict)

TypedList = (
    lambda T:
        Type(constructor=List.constructor,
             recognizer=lambda L: L in List and
                                  all(x in T for x in L)))

MinDict = (
    lambda minimum:
        Type(constructor=Dict.constructor,
             recognizer=lambda D: D in Dict and
                                  all(k in D and D[k] in v
                                      for (k,v) in minimum.items())))

MaxDict = (
    lambda minimum:
        Type(constructor=Dict.constructor,
             recognizer=lambda D: D in Dict and
                                  all(k not in D or D[k] in v
                                  for (k,v) in maximum.items())))

TypedDict = (
    lambda required, optional={}:
        Type(constructor=Dict.constructor,
             recognizer=lambda D: D in MinDict(required) and
                                  D in MaxDict({**required,**optional})))

StringLiteral = (
    lambda *choices:
        Type(constructor=String.constructor,
             recognizer=lambda s: s in String and s in choices))

Boolean = StringLiteral("true", "false")

# A proof system.

class Proof:
    def __init__(self, T):
        self.T = T
    def __contains__(self, proof):
