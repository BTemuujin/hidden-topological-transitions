# Replication: Hidden Topological Transitions in Emergent Magnetic Monopole Lattices

## What This Is

A strict replication of the study by Kato and Motome (Phys. Rev. B 107, 094437, 2023). The project aims to reproduce the phase diagrams and identify the hidden topological transitions of emergent magnetic monopoles in 3D spin lattices.

## Core Research Question

Can the hidden topological transitions in emergent magnetic monopole lattices be precisely replicated using the exact steepest descent method?

## Scoping Contract Summary

### Contract Coverage

- [Claim: Precisely reproduce p-h and h-T phase diagrams]: Matching the boundaries of 3Q, 4Q, and 5Q phases and their monopole densities.
- [Acceptance signal]: Reproduction of critical mixing ratios p ≈ 0.517 and 0.529 within 1% tolerance.
- [False progress to reject]: Qualitative agreement of phase shapes without matching specific critical parameter values.

### User Guidance To Preserve

- **User-stated observables:** Monopole charge per MUC (Nm), Specific heat (C), Scalar spin chirality (χsc), Order parameters (m_eta).
- **User-stated deliverables:** p-h and h-T phase diagrams identifying 3Q, 4Q, and 5Q hedgehog lattices.
- **Must-have references / prior outputs:** Phys. Rev. B 107, 094437 (2023) (Ref-01).
- **Stop / rethink conditions:** Failure to reproduce the intermediate 5Q phase at p ≈ 0.52.

### Scope Boundaries

**In scope**

- Strict replication of the Hamiltonian (RKKY, biquadratic, and DM interactions).
- Implementation of the exact steepest descent method for the thermodynamic limit.
- Calculation of order parameters, specific heat, and scalar spin chirality.
- Generation of p-h and h-T phase diagrams.

**Out of scope**

- Inclusion of conduction electron degrees of freedom.
- Extensions to other lattice geometries.
- Experimental data fitting beyond the scope of the original paper.

### Active Anchor Registry

- Ref-01: Phys. Rev. B 107, 094437 (2023)
  - Why it matters: Primary benchmark for all results, including phase boundaries and the "hidden" nature of topological transitions.
  - Carry forward: planning | execution | verification | writing
  - Required action: read | use | compare | cite

### Carry-Forward Inputs

- Provided txt version of Phys. Rev. B 107, 094437.

### Skeptical Review

- **Weakest anchor:** Convergence of steepest descent method on V100 hardware compared to A100.
- **Unvalidated assumptions:** Thermodynamic limit approximations in the steepest descent method.
- **Competing explanation:** None known (strict replication).
- **Disconfirming observation:** Failure to reproduce the 5Q intermediate phase at p ≈ 0.52.
- **False progress to reject:** Qualitative agreement without matching specific critical values.

### Open Contract Questions

- Optimization of JAX/Optax hyperparameters for convergence on V100 vs A100.

## Research Questions

### Answered

(None yet — investigate to answer)

### Active

- [ ] Can the 3Q, 4Q, and 5Q phases be stabilized and identified using the proposed Hamiltonian?
- [ ] Does the exact steepest descent method correctly capture the thermodynamic limit of the partition function?
- [ ] Are the hidden topological transitions (changes in Nm without C singularities) reproduced?
- [ ] Do the phase boundaries in the p-h and h-T planes match the original paper's results?

### Out of Scope

- Effect of conduction electron scatterers on topological transitions.
- Quantum effects in the 3D system.

## Research Context

### Physical System

Emergent magnetic monopole lattices in 3D spin systems (modeled after MnSi1-xGex and SrFeO3).

### Theoretical Framework

Exact steepest descent method applied to a spin lattice Hamiltonian with long-range interactions.

### Key Parameters and Scales

| Parameter | Symbol | Regime  | Notes   |
| --------- | ------ | ------- | ------- |
| Mixing Ratio | p     | [0, 1]  | Interpolates between 3Q (p=0) and 4Q (p=1) |
| Magnetic Field | h     | Varying | Studied in [100], [110], [111] directions |
| Temperature | T     | Varying | From 0 to PM transition |
| Interaction | J, K, D | Fixed | J=1, K=0.6, D=0.3 (as per paper) |
| Period | (cid:3) | 8       | Magnetic period of stable textures |

### Known Results

- 3Q and 4Q HLs are stabilized at p=0 and p=1 respectively.
- Hidden topological transitions occur with change of monopole density Nm.
- A 5Q intermediate phase exists near p ≈ 0.52.

### What Is New

This is a strict replication intended to verify the results and establish a baseline implementation.

### Target Venue

N/A (Replication project).

### Computational Environment

V100 GPU nodes using JAX and Optax. Implementation utilizes `jax.pmap` for explicit multi-GPU scaling and deterministic seeding for reproducibility.

## Notation and Conventions

See `.gpd/CONVENTIONS.md` for all notation and sign conventions.
See `.gpd/NOTATION_GLOSSARY.md` for symbol definitions.

## Unit System

Natural units (energy unit J = 1, lattice constant = 1).

## Requirements

See `.gpd/REQUIREMENTS.md` for the detailed requirements specification.

## Key References

- Ref-01: Phys. Rev. B 107, 094437 (2023)

## Constraints

- **Computational**: Must utilize V100 GPUs effectively via JAX.
- **Accuracy**: Critical transition points must be matched within 1%.

## Key Decisions

| Decision | Rationale | Outcome   |
| -------- | --------- | --------- |
| Strict replication | Establish baseline before any extensions | — Pending |
| JAX/Optax on V100 | Match author's toolchain for consistency | — Pending |
| Simulated Annealing | Avoid local minima in rugged energy landscape (especially 5Q) | — Implemented |
| Guided Seed Injection | Inject a priori 1Q and 3Q states to ensure global convergence | — Implemented |
| Deterministic Seeding | Ensure exact reproducibility via MD5 hash of (p, h, T) | — Implemented |
| Dual-Search Strategy | Combined Global Random Search and Local Phase Tracking to avoid symmetry traps | — Implemented |
| Basin-Safe Derivative | Specific Heat $C$ calculation using $\pm \Delta T$ steps with topological jump aversion | — Implemented |
| Real-Space Mapping | 3D spin reconstruction and 2D slice projections with plaquette markers | — Implemented |

---

_Last updated: 2026-04-23 after implementation of Fig 6 pipeline and high-resolution p-sweeps_
