# DERV-01: Spin Lattice Hamiltonian Construction

## Objective
Construct the spin lattice Hamiltonian including RKKY-type, biquadratic, and Dzyaloshinskii-Moriya (DM) interactions as a function of the mixing ratio $p$, strictly replicating Phys. Rev. B 107, 094437 (2023).

## Hamiltonian Formulation
The system is modeled on a simple cubic lattice. The total Hamiltonian per spin is defined as:
$$H = H_{RKKY} + H_{B} + H_{DM} + H_{Zeeman}$$

### 1. RKKY and Biquadratic Terms
The interaction is parameterized by a mixing ratio $p \in [0, 1]$, which interpolates between the 3Q and 4Q regimes:
- For $\eta \in \{1, 2, 3\}$ (3Q vectors): $J_\eta = J(1-p)$, $K_\eta = K(1-p)$, $D_\eta = D(1-p)$
- For $\eta \in \{4, 5, 6, 7\}$ (4Q vectors): $J_\eta = Jp$, $K_\eta = Kp$, $D_\eta = Dp$

The energy contribution is:
$$H_{int} = \sum_{\eta} \left[ -J_\eta (S_{Q_\eta} \cdot S_{-Q_\eta}) + K_\eta (S_{Q_\eta} \cdot S_{-Q_\eta})^2 \right]$$

### 2. Dzyaloshinskii-Moriya (DM) Interaction
The DM term is implemented via the cross product of the Fourier components, projected onto the $Q$-vector direction:
$$H_{DM} = \sum_{\eta} -D_\eta \left( S_{Q_\eta} \times S_{-Q_\eta} \right) \cdot \hat{q}_\eta$$

### 3. Zeeman Term
For an external field $\mathbf{h}$ in direction $\hat{h}$:
$$H_{Zeeman} = -\mathbf{h} \cdot \sum_i \mathbf{S}_i$$

## Implementation Parameters
- $J = 1.0$ (Energy unit)
- $K = 0.6$
- $D = 0.3$
- Lattice period $D_{period} = 8$
