import datetime, time

class Reporter:
    def __init__(self,
                 logfile = None,
                 time_between_report=1.0,
                 time_between_autocomplete=60.0,
                 time_between_saves=3600.0) :
        self.n = 0
        if logfile is None:
            logfile = str(time.time()) + '.log'
        self.logfile = logfile
        self.log = open(logfile, 'a')
        self.log.write('date,step,time,loss\n')

    def step(self, loss):
        self.n = self.n + 1
        self.log.write(','.join([
            datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S"),
            str(self.n),
            str(time.time()),
            str(loss)])+'\n')
