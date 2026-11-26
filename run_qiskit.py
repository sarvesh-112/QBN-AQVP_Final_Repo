"""Hardware realism: build the QBN-AQVP circuit in Qiskit, transpile to IBM heavy-hex fake backends,
report routed two-qubit counts/depth, and measure score distortion under device-calibrated noise."""
import pickle, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino


def qbn_circuit(W, measure=True):
    n = W.shape[0]
    a = ParameterVector("a", n)
    qc = QuantumCircuit(n, n if measure else 0)
    for i in range(n):
        qc.ry(a[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                qc.cry(np.pi * W[i, j] / 2, i, j)
    if measure:
        qc.measure(range(n), range(n))
    return qc, a


def twoq(qc):
    ops = qc.count_ops()
    return sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "swap"))


def main(out="results/qiskit.pkl", n_inputs="40", shots="4000"):
    n_inputs, shots = int(n_inputs), int(shots)
    X, A = load_dataset("metr-la")
    subs = greedy_subgraphs(A, seed=0)
    res = dict(transpile={}, noise={})
    backends = dict(sherbrooke=FakeSherbrooke(), torino=FakeTorino())
    for bname, be in backends.items():
        for si, S in enumerate(subs):
            W = A[np.ix_(S, S)]
            qc, a = qbn_circuit(W)
            logical = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
            rows = []
            for seed in range(5):
                tq = transpile(qc, backend=be, optimization_level=3, seed_transpiler=seed)
                rows.append((twoq(tq), tq.depth(), len({q for inst in tq.data for q in inst.qubits
                                                         if inst.operation.name not in ("barrier", "measure")})))
            rows = np.array(rows)
            res["transpile"][(bname, si)] = dict(nedges=int((np.triu(W, 1) > 0).sum()), logical_cx=twoq(logical),
                                                 logical_depth=logical.depth(), routed_2q=rows[:, 0].tolist(),
                                                 routed_depth=rows[:, 1].tolist(), phys_qubits=rows[:, 2].tolist())
            print(bname, si, res["transpile"][(bname, si)], flush=True)
    # device-calibrated noise: ideal vs noisy <Z> on random METR-LA inputs
    be = backends["sherbrooke"]
    sim = AerSimulator.from_backend(be)
    rng = np.random.default_rng(0)
    for si, S in enumerate(subs):
        W = A[np.ix_(S, S)]
        n = len(S)
        bn = BNCircuit(W)
        qc, a = qbn_circuit(W)
        tq = transpile(qc, backend=be, optimization_level=3, seed_transpiler=0)
        ts = rng.integers(0, len(X), n_inputs)
        alphas = np.pi * X[ts][:, S] + rng.uniform(-0.1, 0.1, (n_inputs, n))
        circs = [tq.assign_parameters({a[i]: alphas[r, i] for i in range(n)}) for r in range(n_inputs)]
        t0 = time.time()
        counts = sim.run(circs, shots=shots, seed_simulator=si).result().get_counts()
        dt = time.time() - t0
        Zn = np.zeros((n_inputs, n))
        for r, c in enumerate(counts):
            for bits, cnt in c.items():
                b = np.array([int(ch) for ch in bits[::-1]])
                Zn[r] += cnt * (1 - 2 * b)
        Zn /= shots
        Zi = 2 * bn.Q(alphas) - 1
        slope = float((Zn * Zi).sum() / (Zi * Zi).sum())
        G = n + bn.nedges
        res["noise"][si] = dict(Zi=Zi, Zn=Zn, mad=float(np.abs(Zn - Zi).mean()), maxdev=float(np.abs(Zn - Zi).max()),
                                slope=slope, G=G, routed_2q=twoq(tq), sec_per_circ=dt / n_inputs,
                                depol_pred={p: (1 - p) ** G for p in (0.001, 0.01)})
        print("noise", si, res["noise"][si]["mad"], slope, dt, flush=True)
    pickle.dump(res, open(out, "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:])
