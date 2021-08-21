import asyncio


class Fun:
    def __init__(self, f, *args, output=None, **kwargs):
        self.args = args
        if output is None:
            self.output = []
        else:
            self.output = output
        self.kwargs = kwargs
        self.task = asyncio.create_task(Fun.loop(f, args, kwargs, output))

    def __del__(self):
        print("Fun: Task cancelled.")
        self.task.cancel()

    @staticmethod
    async def loop(f, args, kwargs, output):
        args_at_step = lambda n: [x[n] for x in args]
        kwargs_at_step = lambda n: {k:v[n] for (k,v) in kwargs.items()}
        closure = lambda n: f(*args_at_step(n), **kwargs_at_step(n))
        n = len(output)
        while True:
            try:
                output.append(closure(n))
                n += 1
            except asyncio.CancelledError:
                return
            except:
                await asyncio.sleep(.01)
