import asyncio

def app(f, kwargs=None, output=None):
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
