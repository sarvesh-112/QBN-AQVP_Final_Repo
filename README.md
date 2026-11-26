# QBN-AQVP benchmark code

Reproduces every number, table and figure in the IEEE Access manuscript
"Entangled Graph Encoders Are Classical Bayesian Networks: A Ground-Truth Benchmark of
Quantum Scores for Bellman Value Estimation".

## Data
Put these files in a `data/` folder and set `QBN_DATA=/path/to/data`:
- `metr-la.h5`, `pems-bay.h5` (DCRNN releases; e.g. github.com/deepkashiwa20/DL-Traff-Graph)
- `adj_mx.pkl`, `adj_mx_bay.pkl` (github.com/liyaguang/DCRNN, data/sensor_graph)

## Requirements
Python 3.10+, numpy, scipy, h5py, matplotlib; for the hardware study: qiskit, qiskit-aer, qiskit-ibm-runtime, pylatexenc.

## Run (≈3 CPU-hours on 2 cores)
    python run_main.py --dataset metr-la --noise --out results/main_metr.pkl   # main 10-node study + noise
    python run_qiskit.py results/qiskit.pkl 40 4000                             # transpilation + device noise
    ./run_rest.sh            # positive control, full networks, PEMS-BAY, gamma / window sensitivity
    python run_extras.py     # theory checks, forced feedback, device-calibrated noise in the loop
    python run_c4fb.py       # forced online training for the learned classical score C4
    python figures.py ablation dynamics jacobian pc ; python fig_arch.py
    python make_numbers.py /path/to/paper/numbers.tex   # writes every number as a LaTeX macro

## Revision experiments (reviewer requests)
    python -I run_revision.py results/revision.pkl      # node orderings, M4-sym, offline-trained MLP (C5),
                                                        # lag/forecast-input scores, decision ties, scale stats
    python -I run_revision2.py results/revision2.pkl    # offline-trained circuit / product / graph-mixing scores
    python -I compute_extras_rev.py results/extras_rev.pkl   # decision-gap bins, full-network statistics
    python -I make_numbers_rev.py /path/to/paper/numbers_rev.tex   # macros + tabrow_rev.tex
    python figs/make_figs.py OUTDIR ; python figs/circ.py OUTDIR     # Fig. 1 and the circuit drawings

`results/` contains the pickles produced for the submitted manuscript.
Output paths for figures/numbers default to /home/claude/paper - edit OUT in figures.py / fig_arch.py if needed.
