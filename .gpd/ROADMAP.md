# Research Roadmap: Replication of Hidden Topological Transitions

## Phase 1: Formalism & Hamiltonian
**Goal**: Establish the mathematical foundation and Hamiltonian specification for the 3D spin lattice.
- [ ] **DERV-01**: Construct the spin lattice Hamiltonian (RKKY, biquadratic, and DM interactions) as a function of mixing ratio $p$.
- [ ] **DERV-02**: Formulate the free energy functional $G(\{S_0\})$ for the exact steepest descent method.
**Success Criteria**: Completed derivations matching Ref-01, ready for numerical translation.

## Phase 2: Numerical Implementation
**Goal**: Implement the computational engine using JAX and Optax.
- [x] **NUMR-01**: Implement the $G$ functional maximization loop on V100 GPUs. (Implemented with Simulated Annealing, `jax.pmap` scaling, warm-starting, and guided seed injection)
- [x] **NUMR-02**: Implement the partition function calculation for the thermodynamic limit.
- [x] **NUMR-03**: Implement the extraction of order parameters $m_\eta$ and monopole charge $N_m$. (Implemented robust phase classification and exact thermal magnitude recovery)
**Success Criteria**: Convergent optimization loop that produces stable spin configurations.

## Phase 3: Phase Maps & Validation
**Goal**: Reproduce the phase diagrams and identify hidden topological transitions.
- [ ] **VALD-01**: Generate the $p$-$h$ phase diagram and identify the 3Q $\to$ 5Q $\to$ 4Q sequence.
- [ ] **VALD-02**: Reproduce critical mixing ratios $p \approx 0.517$ and $0.529$ (within 1% tolerance).
- [ ] **VALD-03**: Generate $h$-$T$ phase diagrams for $p=0.4, 0.5, 0.6$ across field directions $[100], [110], [111]$. (In Progress: p=0.4 completed for 3 directions)
- [ ] **VALD-04**: Calculate specific heat $C$ and scalar spin chirality $\chi_{sc}$ to verify "hidden" transitions.
**Success Criteria**: quantitative agreement with Ref-01 benchmarks.
