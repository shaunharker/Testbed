from sortedcontainers import SortedList


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
