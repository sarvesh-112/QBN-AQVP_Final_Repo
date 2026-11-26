import sys, time, numpy as np
sys.path.insert(0, '/home/claude/code/qbn')
from core import *
X, A = load_dataset('metr-la'); print(X.shape, A.shape, (A>0).sum()//2)
subs = greedy_subgraphs(A, seed=0)
for S in subs:
    W = A[np.ix_(S,S)]; print(S, int((np.triu(W,1)>0).sum()))
W = A[np.ix_(subs[0],subs[0])]
bn = BNCircuit(W); sv = StateVector(W)
rng = np.random.default_rng(1)
al = rng.uniform(-1,4,(50,10))
print('BN vs SV max err', np.abs((2*bn.Q(al)-1) - sv.expZ(al)).max())
# jacobian vs finite differences
a = al[0]; Qs = bn.Q(shifted_alphas(a)); J = 0.5*(Qs[1:11]-Qs[11:]).T
h=1e-6; Jfd = np.array([(bn.Q((a+h*np.eye(10)[m])[None])[0]-bn.Q((a-h*np.eye(10)[m])[None])[0])/(2*h) for m in range(10)]).T
print('PSR vs FD', np.abs(J-Jfd).max(), 'upper', np.abs(np.triu(J,1)).max())
t=time.time()
for _ in range(300): bn.Q(shifted_alphas(a))
print('bn step ms', (time.time()-t)/300*1000)
ph = rng.uniform(-1,1,(41,10)); t=time.time()
for _ in range(50): sv.expZ(np.repeat(a[None],41,0), ph)
print('sv step ms', (time.time()-t)/50*1000)
Q2,J2 = bn.QJ(a); print('QJ vs PSR', np.abs(J2-J).max(), np.abs(Q2-Qs[0]).max())
t=time.time()
for _ in range(300): bn.QJ(a)
print('QJ ms', (time.time()-t)/300*1000)
