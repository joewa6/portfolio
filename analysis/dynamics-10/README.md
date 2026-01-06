# DYNAMICS-10 Analysis Code

This folder contains **analysis-only** code for the DYNAMICS-10 blog series.

## Rules

1. **Analysis code produces figures only**
2. **Blog posts never contain executable code**
3. **Notebooks are temporary—delete after converting to .py**
4. **All figures go to `../../assets/img/blog/dynamics-10/dayXX/`**

## Structure

```
analysis/dynamics-10/
├── utils.py                    # Shared utilities
├── day01_double_well.py        # Day 01 figure generation
├── day02_state_definition.py  # Day 02 figure generation
└── ...
```

## Usage

From this directory:

```bash
python day01_double_well.py
```

This will generate all figures for Day 01 in the correct location.

## Dependencies

Standard scientific stack:
- numpy
- matplotlib
- scipy

Install with:
```bash
pip install numpy matplotlib scipy
```

## Non-Negotiables

- One script per day
- Script names match blog post filenames
- Figures saved with `utils.save_figure()`
- No hardcoded paths
- No interactive plots in scripts
