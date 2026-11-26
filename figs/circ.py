import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42; matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from core import load_dataset
from qiskit import QuantumCircuit
X, A = load_dataset('metr-la')
P = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'main_metr.pkl'),'rb'))
S = P['subs'][0]
W = A[np.ix_(S, S)]
def circ(W, labels=True, xin=None):
    n = W.shape[0]
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(np.pi*(xin[i] if xin is not None else 0.5), i) if False else qc.ry(0.0, i)
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > 0:
                qc.cry(np.pi*W[i, j]/2, i, j)
    qc.measure_all()
    return qc
# small example: the five nodes of subgraph 1 with most edges among them
best = None
import itertools
for c in itertools.combinations(range(10), 5):
    Wc = W[np.ix_(c, c)]
    e = int((np.triu(Wc, 1) > 0).sum())
    conn = True
    if best is None or e > best[0]:
        best = (e, c)
c = best[1]; Ws = W[np.ix_(c, c)]
print('small example nodes', c, 'edges', best[0])
from qiskit.circuit import Parameter
def circ_sym(W):
    n = W.shape[0]
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(Parameter('α'+str(i+1).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉'))), i)
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > 0:
                qc.cry(round(np.pi*W[i, j]/2, 2), i, j)
    qc.measure_all()
    return qc
style = {'fontsize': 11, 'subfontsize': 9}
f = circ_sym(Ws).draw('mpl', style=style, fold=-1, plot_barriers=False)
f.savefig(sys.argv[1] + '/circuit_small.pdf', bbox_inches='tight'); f.savefig(sys.argv[1] + '/circuit_small.png', dpi=80, bbox_inches='tight')
f = circ_sym(W).draw('mpl', style=style, fold=-1, plot_barriers=False)
f.savefig(sys.argv[1] + '/circuit_full.pdf', bbox_inches='tight'); f.savefig(sys.argv[1] + '/circuit_full.png', dpi=60, bbox_inches='tight')
print('edges full', int((np.triu(W,1)>0).sum()))
