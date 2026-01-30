# MCM Problem C: Data With The Stars

Analysis of Dancing with the Stars competition data (2026 MCM Problem C).

## Structure

```
repo/
  data/
    2026_MCM_Problem_C_Data.csv
  src/
    io.py
    preprocess.py
    rules.py
    models/
      vote_latent.py
      meanfield.py
    fit/
      fit_elimination.py
      uncertainty.py
    analysis/
      season_compare.py
      controversy_cases.py
      pro_dancer_effects.py
      new_system.py
  reports/
    figures/
    tables/
  main.py
  requirements.txt
  README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```
