# Shaun Harker
# Disclaimer: Look, I just wanted to see if I could.

id = lambda x: x
app = lambda f, x: f(x)
comp = lambda f: lambda g: lambda x: app(g,app(f,x))
fmap = lambda f: lambda *xs: tuple(f(x) for x in xs)
comap = lambda *fs: lambda x: tuple(f(x) for f in fs)
zap = lambda *fs: lambda *xs: tuple(f(x) for (f,x) in zip(fs,xs))
seq = lambda *fs: lambda x: id if len(fs)==0 else (fs[0] if len(fs) == 1 else seq(fs[1:])(fs[0](x)))

ones = lambda *shape: torch.ones(shape,device='cuda')
zeros = lambda *shape: torch.zeros(shape,device='cuda')
empty = lambda *shape: torch.empty(shape,device='cuda')
randn = lambda *shape: torch.randn(shape,device='cuda')

#parameters = lambda *shape: torch.empty(shape,device='cuda',requires_grad=True)
parameters = lambda kind: lambda *shape, scale: scale*kind(shape,device='cuda',requires_grad=True)

embedding = lambda pd: lambda x: torch.index_select(pd["W"],0,x.view(-1)).view(x.shape+(W.shape[-1],))
diagonal = lambda pd: lambda x: x*pd["D"]
linear = lambda pd: lambda x: x@pd["W"]
bias = lambda pd: lambda x: x+pd["b"]
affine = lambda pd: seq(linear(pd["W"]),bias(pd["b"]))
sigmoid = lambda x: 1/(1+torch.exp(x))
gelu = lambda x: 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
softmax = lambda x: (lambda x, ex: ex/torch.sum(ex,dim=-1,keepdim=True))(x,torch.exp(x))
crossentropy = lambda y: lambda x: (torch.log(torch.sum(torch.exp(x),dim=-1)).view(-1)-torch.index_select(x.view(-1,x.shape[-1]),-1,y.view(-1)))/math.log(x.shape[-1]).view(y.shape)
E = lambda x: torch.sum(x,dim=-1,keepdim=True)/x.shape[-1]
layernorm = lambda pd: lambda x: (lambda x,Ex: pd["b"]+pd["g"]*(x-Ex)/(pd["eps"]+torch.sqrt(E(x**2)-Ex**2)))(E(x))
dropout = lambda hd: lambda x: torch.nn.Dropout(hd["p"])(x)
attn = lambda hd: lambda **xd: (lambda Q,K,V: softmax(hd("A")(Q.shape[-2])+(Q/math.sqrt(hd["d_k"]))@K.transpose(-1,-2))@V)(xd["Q"],xd["K"],xd["V"])
split_heads = lambda hd: lambda x: x.view(x.shape[-1:]+(hd["n_heads"],-1)).transpose(-2,-3)
merge_heads = lambda x: x.transpose(-2,-3).view(x.shape[-2:],-1)

mhattn = lambda A, n_heads: lambda Q,K,V: merge_heads(attn(hd["mask"])(*map(split_heads(n_heads),[Q,K,V])))
mhattn = lambda A, n_heads: lambda Q,K,V: seq(

seq(comap(seq(affine(pd[_]),split_heads(n_heads)) for _ in ["Q","K","V"]),attn(hd["attn"])

lambda a, b:
lambda x: fmap(split_heads(n_heads))


...

[g]
(affine(pd["Q"]),affine(pd["K"]),affine(pd["V"]))
comap(split_heads(n_heads))
attn(hd["mask"])
merge_heads

fmap(split_heads(n_heads))(comap
bidirectional_mask = lambda n: ones(n,n)
causal_mask = lambda n: 1-1/torch.tril(ones(n,n))
half_causal_mask = lambda n: 1-1/torch.cat(torch.cat([ones(n//2,n//2),zeros(n//2,n//2)],dim=1),torch.tril(ones(n,n))[n//2:,:])
attn_weights = lambda n_model, n_heads, d_head, n_layers: [parameters((n_model,n_heads*d_head)) for _ in range(n_layers)]
residual = lambda f: lambda x: x+f(x)

mlp = lambda pd: seq(affine(pd["affine0"]),gelu,affine(pd["affine1"]))
rfdln = lambda pd: lambda f: seq(residual(seq(f,dropout(pd["dropout"]))),layernorm(pd["layernorm"])

lambda hd: lambda pd: seq(rfdln(pd["rfdln0"])(seq(mhattn(hd["mhattn"]),affine(pd["affine"]))),rfdln(pd["rfdln1"])(mlp(pd["mlp"])))



self.criterion(X[...,n_ctx//2:,:].view(-1,self.n_vocab),Y[...,n_ctx//2:].view(-1)).view(X.shape[:-2]+(-1,))/math.log(self.n_vocab)

# lambdacontext

class contextmanager()
