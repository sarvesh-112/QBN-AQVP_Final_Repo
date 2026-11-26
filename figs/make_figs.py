import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = sys.argv[1]
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'stix'})
# ---------------- Fig 1 architecture ----------------
fig,ax=plt.subplots(figsize=(7.2,3.05)); ax.set_xlim(0,100); ax.set_ylim(0,43); ax.axis('off')
FS=7.6
def box(x,y,w,h,txt,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.0',fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=FS,linespacing=1.5)
def arr(x1,y1,x2,y2,c='#333333'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=c,lw=1.2,shrinkA=0,shrinkB=0))
box(0.5,24,12,13,'Snapshot\n$x_t\\in[0,1]^n$\nand graph $W$','#f2f2f2','#555555')
box(16,24,24,13,'(1) Graph circuit\n$R_Y(\\pi x_i+\\theta_i)$ on every qubit\n$CR_Y(\\pi w_{ij}/2)$ on every edge\nscore $s_i=\\langle Z_i\\rangle$','#e8f0fb','#1f5fae')
box(43.5,24,20,13,'(2) Reward\n$R_i=U_i(t)+\\lambda_t\\,s_i(t)$\n$\\lambda_t=0$: classical\nestimator recovered','#eaf6ea','#2e8b2e')
box(67,24,32.5,13,'(3) Graph Bellman step\n$\\hat V_i(t)=R_i+\\gamma\\,\\max_{j\\in\\mathcal{N}^{+}(i)}\\hat V_j(t-1)$\n$\\mathcal{N}^{+}(i)$: neighbours of $i$ and $i$ itself','#fff3e3','#d9822b')
box(67,2,32.5,13,'(4) Per-node TD error\n$\\delta_i(t)=U_i(t)+\\gamma\\max_{j}\\hat V_j(t+1)-\\hat V_i(t)$','#fdeaea','#c0392b')
box(0.5,2,63,13,'(5) Feedback: semi-gradient of $\\mathcal{L}_t=\\Vert\\delta(t)\\Vert^2/(2n)$\n$\\lambda\\leftarrow\\Pi_{[0,1]}(\\lambda+\\eta_\\lambda\\,\\delta^{\\top}s/n)$   (signed rule)\n$\\theta\\leftarrow\\theta+\\eta_\\theta\\,\\kappa\\lambda\\,J^{\\top}\\delta/n$,   $J_{km}=\\partial Q_k/\\partial\\theta_m$ (parameter shift)','#fdeaea','#c0392b')
arr(13.4,30.5,15.6,30.5); arr(40.6,30.5,43.1,30.5); arr(64.2,30.5,66.6,30.5); arr(83,23.6,83,15.4)
arr(66.6,8.5,64.0,8.5,c='#c0392b')
arr(28,15.4,28,23.6,c='#c0392b'); arr(53.5,15.4,53.5,23.6,c='#c0392b')
ax.text(29,19.2,'$\\theta$',color='#c0392b',fontsize=10); ax.text(54.5,19.2,'$\\lambda$',color='#c0392b',fontsize=10)
ax.text(28,39.6,'$Z$-basis statistics = a classical Bayesian network (Proposition 1)',ha='center',fontsize=7.2,style='italic',color='#1f5fae')
fig.savefig(f'{OUT}/architecture.pdf',bbox_inches='tight'); fig.savefig(f'{OUT}/architecture.png',dpi=110,bbox_inches='tight')
plt.close(fig)
