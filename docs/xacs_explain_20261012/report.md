# XACs/ACES explanation-supervision dev pilot

completed 24/24; paired blocks 12/12; independent reload audit=True; frozen weight w=0.01.
Supervision uses development-train pairs only; official test was not used for training, mining or selection, and was not evaluated.

| Task | seed | MSE RMSE | explain RMSE | MSE Cliff | explain Cliff |
|---|---:|---:|---:|---:|---:|
| CHEMBL234_Ki | 42 | 0.610958 | 0.617289 | 0.670537 | 0.671296 |
| CHEMBL234_Ki | 43 | 0.617191 | 0.644058 | 0.673767 | 0.704810 |
| CHEMBL234_Ki | 44 | 0.606676 | 0.611134 | 0.655567 | 0.626732 |
| CHEMBL2047_EC50 | 42 | 0.440175 | 0.375082 | 0.508261 | 0.458292 |
| CHEMBL2047_EC50 | 43 | 0.432279 | 0.438537 | 0.413429 | 0.467609 |
| CHEMBL2047_EC50 | 44 | 0.442160 | 0.426012 | 0.444841 | 0.448701 |
| CHEMBL2147_Ki | 42 | 0.593904 | 0.584792 | 0.623104 | 0.608984 |
| CHEMBL2147_Ki | 43 | 0.613114 | 0.638743 | 0.588047 | 0.600417 |
| CHEMBL2147_Ki | 44 | 0.681613 | 0.673501 | 0.599276 | 0.627419 |
| CHEMBL244_Ki | 42 | 0.752133 | 0.768134 | 0.878515 | 0.886706 |
| CHEMBL244_Ki | 43 | 0.770446 | 0.784583 | 0.930194 | 0.921932 |
| CHEMBL244_Ki | 44 | 0.740273 | 0.781221 | 0.856656 | 0.894417 |

```json
{
  "completed": 24,
  "total": 24,
  "paired_blocks": 12,
  "independent_reload_audited": true,
  "weight": 0.01,
  "rmse": {
    "mse": 0.6084102439886194,
    "explain": 0.6119237603385713,
    "relative_improvement": -0.005774913201523235
  },
  "cliff": {
    "mse": 0.6535160266495473,
    "explain": 0.6597762258719516,
    "relative_improvement": -0.00957925891197986
  },
  "tasks_winning_at_least_two_seeds": 0,
  "go": false
}
```

Both arms share the identical per-batch id sequence; matched (task, seed) initialisation hashes agree. Validation selects the checkpoint; three seeds share one split and estimate initialisation variance, not independent datasets.
