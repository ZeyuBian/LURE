# LURE

Tabular off-policy evaluation simulation code in R.

## Files

- `simulation_tabular.R`: full simulation script.
- `run_sim_quick.R`: quick smoke test with a small number of replications.
- `Methods.R`: baseline OPE method implementations.

## Requirements

- R
- Package: `rpart`

Install the package in R if needed:

```r
install.packages("rpart")
```

## Usage

Run these commands from the project directory.

Quick check:

```r
source("run_sim_quick.R")
```

Full simulation:

```r
source("simulation_tabular.R")
```

Generated outputs are written into the project folder.