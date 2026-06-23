# A Tutorial on Variational Inference for Single-Cell Genomics

Demonstration code and notebooks for the paper **Variational Inference Methods for Single-Cell Genomics**.

Reproducing simulation results, single-cell probabilistic inference examples, and temporal single-cell analyses used in the tutorial.


## Repository structure

```text
01_simulation/
  01_LMM/
    LMM.py
    LMM_simulation.ipynb
  02_scLDA/
    LDA.py
    ScLdaSimData.py
    LDA_simulation.ipynb
  03_GLMM/
    GLMM.py
    GLMM_simulation.ipynb

02_scPI/
  FA.py
  ZIFA.py
  01_computational_time.ipynb
  02_performance_comparison.ipynb


03_TemporalGP/
  GP.py
  utils.py
  01_compare.ipynb
  02_leave_cohort.ipynb
```

## Notebook

- Follow the notebooks in `01_simulation/` to reproduce the simulation tutorials for LMM, scLDA, and GLMM.
- Follow the notebooks in `02_scPI/` to run the single-cell probabilistic inference examples, including computational-time and performance comparisons.
- Follow the notebooks in `03_TemporalGP/` to compare TemporalGP models and run the leave-cohort microglia experiment.

