# Fluid-Inspired-StratCom

Simulating a **fluid-inspired strategic communications model**.

The code runs three example scenarios and reproduces the appendix figure from the paper, along with simple console diagnostics. The current repository contains one main Python script: `Fluid-inspired_StratCom_Appendix_A.py`.

## What this does

This repository models how communication dynamics evolve over time using a fluid-style formal framework. In the current script, it:

- builds three predefined scenarios
- simulates them over a 28-day window
- computes measures such as effective communicative flow and turbulence risk
- saves a figure as a PNG
- prints summary diagnostics to the console

## Repository contents

- `Fluid-inspired_StratCom_Appendix_A.py` — main simulation and figure-generation script
- `README.md` — project overview file in the repository

## Requirements

Install Python 3.10+ and the following packages:

```bash
pip install numpy matplotlib
```

## How to run

From the repository folder:

```bash
python Fluid-inspired_StratCom_Appendix_A.py
```

This will generate a default output image:

```text
Fluid-inspired_StratCom_Appendix_A_figure.png
```

You can also specify a custom output path:

```bash
python Fluid-inspired_StratCom_Appendix_A.py --out outputs/figure1.png
```

## Scenarios included

The script currently includes three built-in scenarios:

- **A:** CDC/WHO + health influencers
- **B:** Hospital internal campaign
- **C:** Open Twitter ecosystem

## Output

When you run the script, it:

1. simulates the three scenarios
2. saves a two-panel figure
3. prints key values such as fragmentation, peak flow, peak turbulence risk, and instability window to the console

## Use case

This repo is useful if you want to:

- reproduce the appendix figure from the paper
- inspect the scenario setup used in the paper
- adapt the model for your own communication scenarios
- experiment with how network structure, amplification, friction, and damping affect narrative stability

## Notes

The repository is currently lightweight and code-focused. There is no package structure, notebook, or test suite yet; the main entry point is the single Python script in the root of the repo.

## Suggested next improvements

Useful additions later could include:

- a `requirements.txt`
- a sample output image committed to the repo
- clearer mapping between code variables and paper equations
- a notebook version for easier exploration
- parameter files for custom scenarios
