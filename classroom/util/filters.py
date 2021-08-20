from sortedcontainers import SortedList
import asyncio

def liveapp(f, kwargs=None, output=None):
    if output is None:
        output = []
    async def loop(f, kwargs, output):
        # idea: memory handling technique. at max length, randomly forget an item each time
        # for some use-cases, this is great behavior. as we stream new memories in, the old stuff
        # fades away with a half-life of ~N iterations. (1-1/N)^N ~ 1/e chance of surviving an
        # epoch, to be more precise. So the distribution of memories ends up being what?
        # a sensible algorithm is to go to a double-length, then eliminate entries in approximation
        # to the distribution the "perfect" method would have been.
        # this is easy, as we just do a loop over randrange. but then we have to transform that list
        # due to problems with repeats, if we want to do it perfectly right.
        kwargs_at_step = lambda n: {k:v[n] for (k,v) in kwargs.items()}
        closure = lambda n: f(**kwargs_at_step(n))
        position = len(output)
        while True:
            try:
                output.append(f(**kwargs_at_step(position)))
                position += 1
            except:
                await asyncio.sleep(.01)
    return (output, asyncio.create_task(loop(f, kwargs, output)))

def package(list_or_dict):
    """
    Given a list or dictionary of livelist, produce a livelist of their items.
      List[LiveList[T]] -> LiveList[List[T]]
    or
      Dict[K,LiveList[V]] -> LiveList[Dict[K,V]]
    """
    pass

def unpackage(list_or_dict):
    """
    Given a list or dictionary of livelist, produce a livelist of their items.
      LiveList[List[T]] -> List[LiveList[T]]
    or
      LiveList[Dict[K,V]] -> Dict[K,LiveList[V]]
    """
    pass

class TwoWindowFilter:
    def __init__(self, lag=1024):
        self.lag = lag
        self.reg1 = [0.0, 0]
        self.reg2 = [0.0, 0]
        self.count = 0
        assert self.lag % 2 == 0

    def __call__(self, x):
        self.reg1[0] += x
        self.reg1[1] += 1
        self.reg2[0] += x
        self.reg2[1] += 1
        i = self.count % self.lag
        j = self.lag // 2
        self.count = self.count + 1
        if i == 0:
            self.reg2 = [0.0, 0]
            return self.reg1[0]/self.reg1[1]
        if i == j:
            self.reg1 = [0.0, 0]
            return self.reg2[0]/self.reg2[1]
        mu1 = self.reg1[0]/self.reg1[1]
        mu2 = self.reg2[0]/self.reg2[1]
        if i < j:
            t = i/j
            return (1-t)*mu1 + t*mu2
        else:
            t = (i-j)/j
            return t*mu1 + (1-t)*mu2

class CountFilter:
    def __init__(self):
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.count

class DiffFilter:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        y = self.x
        self.x = x
        if y is None:
            return 0
        else:
            self.x = x
            return x - y


class SumFilter:
    def __init__(self):
        self.x = 0

    def __call__(self, x):
        self.x += x
        return self.x


class LogSumFilter:
    def __init__(self):
        self.x = 0

    def __call__(self, x):
        self.x += x
        return log(self.x)/log(2.0)


class KalmanFilter1D:
    def __init__(self, Q=1e-4, R=1e-2, mean=0.0, variance=1.0):
        self.Q = Q
        self.R = R
        self.mean = mean
        self.variance = variance

    def __call__(self, x):
        v = self.variance + self.Q
        self.variance = (self.R*v)/(self.R + v)
        self.mean += self.variance*(x-self.mean)/self.R
        return self.mean

class MedianFilter:
    def __init__(self, memory_limit=1024):
        self.memory_limit = memory_limit
        self.bins = SortedList()
        self.step = 0
    def __call__(self, x):
        self.bins.add((x, self.step))
        self.step += 1
        if len(self.bins) == self.memory_limit:
            # Ran out of spots. Kick out the eldest of the two outliers.
            (a, n) = self.bins.pop(0)
            (b, m) = self.bins.pop(-1)
            if n < m:
                self.bins.add((b, m))
            else:
                self.bins.add((a, n))
        return self.median()

    def median(self):
        L = [x for (x, n) in self.bins]
        n = len(L)
        if len(L) % 2 == 0:
            return (L[n//2-1] + L[n//2])/2
        else:
            return L[n//2]
