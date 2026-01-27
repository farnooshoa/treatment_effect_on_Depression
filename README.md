# Treatment Effect on Depression | Causal Representation Learning + DR ITE Supervision + Conditional Diffusion

This repo implements the **applicant-facing methodology** for estimating treatment effects on next-visit depression severity using a simulated visit-transition dataset (`data_generated.csv`).

The implementation includes:
- Patient-level leakage-free splitting
- Preprocessing (standardization, categorical encoding with unknown, treatment ID mapping)
- Representation learning with **stable (S)** and **confounding (C)** latents
- Factual outcome prediction
- **ITE head** supervised with **cross-fitted doubly robust (DR) pseudo-outcomes**
- Conditional diffusion denoising loss in confounder space (C), conditioned on (S, treatment)

---

## Dataset & Problem Setup

Each row represents a **transition from visit v → visit v+1**:
- Covariates `x` and treatment `T` are taken at visit `v`
- Outcome `y` is measured at visit `v+1`

**Outcome (`y`)**
- `y` is defined as the **next-visit HAMD total score**, computed as the sum of symptom items:
  - `HAMD01` … `HAMD17`

**Treatments**
- Single administered treatment `T ∈ {0,...,K−1}` (K = number of unique therapies in train split)
- Treatments are mapped to integer IDs deterministically from the training split.

**Leakage prevention**
- Train/val/test splits are performed **at the patient level** (`UNIQUEID`), so all samples from a patient stay in one split.
