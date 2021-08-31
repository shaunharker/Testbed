import torch


@torch.no_grad()
def beam(prompt=None,
         n_generate=128,
         n_ctx=None,
         temp=1.0,
         encode=None,
         decode=None,
         output=None):
    """
    Autocomplete using the model

    ## Args
    * `prompt: str` an optional prompt to begin with
    * `n_generate: int` the number of bytes/tokens to generate
    * `n_ctx: int` the number of bytes/tokens in the context window
    * `encode` the function that can turn an str into a sequence of bytes/tokens suitable for the model.
    defaults to utf8encode
    * `decode` the function that can turn the sequences of bytes/tokens used by the model to a str
    defaults to utf8decode
    * `output: Optional[List[int]]` a list to stream the output bytes/tokens to (as `int`s; they will not be decoded to `str`).

    ## TODO
    * make streaming autocomplete with streamed characters (i.e. length 1 strings) using asyncio
    """
    Categorical = torch.distributions.Categorical
    if n_ctx is None:
        n_ctx = self.model.n_ctx
    if encode is None:
        encode = utf8encode
    if decode is None:
        decode = utf8decode
    if prompt is None:
        prompt = decode(self.dataset.batch(1, 2*n_ctx, offset=None).tolist()[0])  # kludge
    print(f"=== Prompt ===\n{prompt}\n=== Autocompletion ===\n")
    x = encode(prompt)
    x = x[-n_ctx:]

    # Get comfortable kids, because this one is a doozy.

    def node(x, n_classes):
        """
        Given a prompt `x`, break the cylinder space x + [256]^* into `n_classes` disjoint cylindrical pieces which
        together cover the space. Aim to make the classes roughly equal in probability.
        A cylinder is an infinite set of the form S_1 x S_2 x ... x S_n x [256]^* where the S_n are subsets of [256].

        Yes, I realize this is an unusual usage of the word cylinder.
        What can I say? I'm a mathematician. Don't sue me.

        There is a really cool relationship to binary (or generalized) decision diagrams. This basically is the generalization.
        There is also a relationship to cylindrical algebraic decomposition (same kind of cylinder, except reals instead of [256]), which is related to computational algebraic geometry and general including Grobner basis which is the term-rewriting algorithm in disguise. Knuth-Bendix completion algorithm -- rewriting a system so that it is confluent.

        Now, back to the problem at hand.

        There are two ways to go about this. One is to keep it "character-level" -- that is, build a tree such that the arcs are labelled by the single letter differences between parent and child. But more generally we can build a DAG, and have multi-byte labels. I don't know how profitable such changes are.

        I do know that we have to progress one byte at a time at the level of calling the model and getting the probs.

        Okay, sure. At each stage, we make a choice to ignore a bunch of low probability bytes.
        One way to do this might have some funny behavior, but here it is:
        take the highest probabilities that sum to more than p of the probability, or the k highest, whichever is smaller.
        Hmm. If there aren't clear favorites, we shouldn't proceed at all? Not unless we want to do something very exhausting, like some second-order analysis, right?
        So, just take everything above a certain probability threshold that implicitly limits how many there will be.
        Say, 25 percent. At most four items, then, but in practice there are never going to be four. because they'd have to all be precisely equal at 25 percent which is quite improbable so we can just break ties arbitrarily if it happens. Now, this is nice because there are four ways to continue: the top three, or "something else, stop here"
        We can thus reduce the 256 probabilities down to four in a simple formula.

        Say, a function `reduce_probs` that took a vector of 256 probabilities and worked like this:

        ```python
        probs = # some vector summing to 1
        choices, probs = reduce_probs(probs)
        assert choices == [3, 4, 7]  # for example
        assert probs == [26.2, 27.1, 30.0]  # for example
        ```

        So, that's the best-first analysis for each prompt.
        So this gives us a tree structure we can unfold. This could go quite some distance.
        And eventually, we'd be far enough down the tree that we knew it didn't matter where we came from.
        Of course, at that point the tree will have grown to 3**n_ctx width, which is quite astronomical, so
        this is seemingly not a practical observation. It is interesting to think about the dag structure that
        occurs when we merge nodes that do not differ in their "prompt" (i.e. the n_ctx arc byte labels leading down the tree to them from above). If we merge in time as well, so it's just a system of all n_ctx length prompts with
        the 3-best graph with probabilities we get some kind of Markov chain structure, but we need to specify how to
        handle the missing probability for the non-top choices (either condition on always being a top continuation, which probably gets weird fast, or just think about the structure that has all the labelled edges properly done).

        Okay, we've flown around in the math space around this idea for a few minutes, so what's the verdict, boss?
        The idea is we unfold nodes in a best first manner. There is probably a name like "A*" or something.
        Our goal is to split the largest probability. So just start building the tree, starting with a root.

        So draw the root and write 1.0 next to it.

        Now pick a node. (I bet you picked the root, didn't you? How'd I know?)
        Draw the top three children as new nodes, draw arcs, and label them with transition probabilities, and also label the node with the probability that it does something else, sort of a leaf probability for non-leaf nodes, if you like that interpretation. It's the probability of the 253 worst choices out of 256 all added up.

        Also label each node with the "incoming probability" which is the product of the weights of the arcs leading to it
        from root.

        Call the last three steps the "grow" step applied to a node, in this case applied to the root node, as the choice was forced.

        Now, what we do is apply grow in a best-first manner to grow our tree, and we stop when we reach a node budget, since each node is a possible continuation.

        If we do this repeatedly to grow text, we might want to be strict and enforce that if we sampled from a node that didn't continue in some way, that we were not going to allow those probabilities. This would then introduce a need to prevent certain nodes from being constructed based on the structure of the previous tree, and also some math research to characterize all this.

        Make life a little easier and store the negative log prob, maybe, then we can do subtree sums on node values.
        there might even by a python structure for this kind of crap.



        """
        probs = self.model.inference(torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]
    def sampler(x):
        x = list(x)
        for _ in range(n_generate):
            probs = self.model.inference(torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]
            if temp > 0:
                y = Categorical(probs=probs**(1.0/temp)).sample().item()
            else:
                y = torch.argmax(probs).item()
            x = (x + [y])[-n_ctx:]
            if output is not None:
                output.append(y)
            yield y
    return decode(list(sampler(x)))
