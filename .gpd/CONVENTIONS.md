# Project Conventions: Replication of Hidden Topological Transitions

## 1. Basic Constants & Units
- **Energy Unit**: $J = 1$
- **Lattice Constant**: $a = 1$
- **Boltzmann Constant**: $k_B = 1$
- **Lattice Type**: Simple cubic
- **Spin Type**: Classical unit vectors $\mathbf{S}_r \in \mathbb{R}^3, |\mathbf{S}_r| = 1$
- **Magnetic Period**: $d = 8$ (implies $Q = 2\pi/d$)

## 2. Hamiltonian Notation
- **Mixing Ratio**: $p \in [0, 1]$
- **Characteristic Wave Numbers**: $Q_\eta$ for $\eta \in \{1, \dots, 7\}$
  - $\eta \in \{1, 2, 3\}$ (Cubic 3Q): $Q_1=(+Q, 0, 0), Q_2=(0, +Q, 0), Q_3=(0, 0, +Q)$
  - $\eta \in \{4, 5, 6, 7\}$ (Tetrahedral 4Q): $Q_4=(+Q, -Q, -Q), Q_5=(-Q, +Q, -Q), Q_6=(-Q, -Q, +Q), Q_7=(+Q, +Q, +Q)$
- **Coupling Constants**:
  - For $\eta \in \{1, 2, 3\}$: $J_\eta = J(1-p), K_\eta = K(1-p), D_\eta = D(1-p)$
  - For $\eta \in \{4, 5, 6, 7\}$: $J_\eta = Jp, K_\eta = Kp, D_\eta = Dp$
- **Interaction Parameters**: $J=1, K=0.6, D=0.3$
- **DM Vectors**: $\mathbf{D}_\eta = D_\eta \frac{\mathbf{Q}_\eta}{|\mathbf{Q}_\eta|}$
- **Fourier Components**: $\mathbf{S}_Q = \frac{1}{\sqrt{N}} \sum_{\mathbf{r}} \mathbf{S}_{\mathbf{r}} e^{-i\mathbf{Q}\cdot\mathbf{r}}$

## 3. Methodology: Exact Steepest Descent
- **Magnetic Unit Cell (MUC)**: Cube of $N_0 = d^3$ spins.
- **Sublattice Averaged Spin**: $\mathbf{S}_{\mathbf{r}_0} = \frac{1}{N_{MUC}} \sum_{\mathbf{R}} \mathbf{S}_{\mathbf{R}+\mathbf{r}_0}$
- **Free Energy Functional**: $G(\{\mathbf{S}_{\mathbf{r}_0}\})$, maximized to find the thermodynamic limit partition function $Z \sim e^{N_{MUC} G}$.
- **Parameterization**: $\mathbf{S}_{\mathbf{r}_0}$ parameterized by $\{v_{0\mathbf{r}_0}, \theta_{\mathbf{r}_0}, \phi_{\mathbf{r}_0}\}$.

## 4. Observables
- **Order Parameter**: $m_\eta = \frac{1}{N_0} \mathbf{S}_{\mathbf{Q}_\eta} \cdot \mathbf{S}_{-\mathbf{Q}_\eta}$
- **Monopole Charge per MUC**: $N_m$ (integer)
- **Specific Heat**: $C = \frac{\partial \epsilon}{\partial T}$
- **Scalar Spin Chirality**: $\chi_{sc}$
- **Magnetization**: $m = \frac{1}{N_0 h} \sum_{\mathbf{r}_0} \mathbf{S}_{\mathbf{r}_0} \cdot \mathbf{h}$
