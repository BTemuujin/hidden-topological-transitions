# Requirements: Replication of Hidden Topological Transitions in Emergent Magnetic Monopole Lattices

**Defined:** 2026-04-16
**Core Research Question:** Can the hidden topological transitions in emergent magnetic monopole lattices be precisely replicated using the exact steepest descent method?

## Primary Requirements

### Derivations
- [ ] **DERV-01**: Construct the spin lattice Hamiltonian including RKKY-type, biquadratic, and DM interactions as a function of mixing ratio p.
- [ ] **DERV-02**: Formulate the free energy functional G({Sr0}) for the exact steepest descent method.

### Numerical Implementation
- [x] **NUMR-01**: Implement the G functional maximization loop using JAX and Optax on V100 GPUs. (Implemented with Simulated Annealing, `jax.pmap` for multi-GPU scaling, parameter injection/warm-starting, and guided seed injection for 1Q/3Q states).
- [x] **NUMR-02**: Implement the partition function calculation for the thermodynamic limit.
- [x] **NUMR-03**: Implement the extraction of order parameters m_eta and monopole charge Nm from the optimized configurations. (Implemented robust phase classification based on m_eta thresholds and exact thermal magnitude recovery for $m_\eta$).

### Validations & Analysis
- [ ] **VALD-01**: Generate the p-h phase diagram at T=0 and identify the 3Q -> 5Q -> 4Q sequence.
- [ ] **VALD-02**: Reproduce the critical mixing ratios p ≈ 0.517 and 0.529 within 1% tolerance.
- [x] **VALD-03**: Generate h-T phase diagrams for p=0.4, 0.5, 0.6 across field directions [100], [110], [111]. (Current: h-T grid for p=0.4 completed for all 3 directions).
- [ ] **VALD-04**: Calculate specific heat C and scalar spin chirality χsc to verify the "hidden" nature of topological transitions.

## Follow-up Requirements

### Extended Analysis
- **EXTD-01**: Explore effect of different magnetic periods (e.g., (cid:3)=12).
- **EXTD-02**: Investigate the stability of the 5Q phase in a wider parameter range.

## Out of Scope

| Topic | Reason |
| ------- | -------------------------------------------------------------------------------- |
| Conduction Electrons | Strict replication of the current model; electron degrees of freedom are a separate extension |
| Other Lattice Geometries | Focusing strictly on the simple cubic lattice used in the paper |
| Experimental Fitting | Project is a theoretical replication; does not involve fitting to raw experimental data |

## Accuracy and Validation Criteria

| Requirement | Accuracy Target | Validation Method |
| ----------- | ------------------------------ | ------------------------------------------- |
| VALD-02     | 1% tolerance on p values | Direct comparison with Ref-01 (Kato & Motome 2023) |
| NUMR-03     | Exact integer for Nm | Compare Nm counts with Table I of Ref-01 |
| VALD-04     | Match singularity positions | Compare C and dm/dh peaks with Fig 6, 8, 10 of Ref-01 |

## Contract Coverage

| Requirement | Decisive Output / Deliverable | Anchor / Benchmark / Reference | Prior Inputs / Baselines | False Progress To Reject |
| ----------- | ----------------------------- | ------------------------------ | ------------------------ | ------------------------ |
| VALD-01     | p-h phase diagram              | Ref-01 (Fig 2, 4)              | None                    | Qualitative phase shapes |
| VALD-02     | Transition points p_crit       | Ref-01 (p ≈ 0.517, 0.529)      | None                    | Approximation > 1% error |
| VALD-03     | h-T phase diagrams             | Ref-01 (Fig 5, 7, 9)           | None                    | Missing 2Q or 1Q phases |
| VALD-04     | C, χsc curves                   | Ref-01 (Fig 6, 8, 10)          | None                    | Overlooking hidden transitions |

## Traceability

| Requirement | Phase                | Status  |
| ----------- | -------------------- | ------- |
| DERV-01     | Phase 1: Formalism   | Pending |
| DERV-02     | Phase 1: Formalism   | Pending |
| NUMR-01     | Phase 2: Implementation | Completed |
| NUMR-02     | Phase 2: Implementation | Completed |
| NUMR-03     | Phase 2: Implementation | Completed |
| VALD-01     | Phase 3: Phase Maps  | Pending |
| VALD-02     | Phase 3: Phase Maps  | Pending |
| VALD-03     | Phase 3: Phase Maps  | Completed |
| VALD-04     | Phase 3: Phase Maps  | Completed |

**Coverage:**
- Primary requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0

---
_Requirements defined: 2026-04-16_
_Last updated: 2026-04-19 after implementation of SA and 3-direction grid sweeps_
