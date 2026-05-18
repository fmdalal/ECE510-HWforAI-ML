# Scope Assesment

1) Chiplet 9 originally used a 3rd-order Taylor polynomial to approximate exp(x) as an optimisation to avoid the hardware cost of a full exponential unit. This approximation is incorrect for the score range produced by the attention heads: the polynomial goes negative for inputs below −2, which means the computed probabilities are invalid for the majority of attention scores. Training on these corrupted probability distributions causes the model to learn wrong attention weights. The chiplet is being corrected to implement IEEE-compliant softmax using the 2^(x/ln2) decomposition with a 64-entry LUT, which is accurate across the full input domain and cannot produce negative values.

2) Full system PnR descoped — placement diverged at 100% overflow even at 7×7 mm die. M3 uses hierarchical PnR per chiplet instead.

3) TILE increased from 8 to 16 — at TILE=8, 98.2% of cells (2,118/2,156) were IO buffers because Yosys removed 18 of 21 modules as unused. 

4) Clock tightened from 10 ns to 5 ns — the M2 critical path was only 3.076 ns, leaving +6.924 ns slack unused. A tighter constraint will force real datapath optimisation.

