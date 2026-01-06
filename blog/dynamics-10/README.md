# DYNAMICS-10 Series

**What can and cannot be inferred from finite trajectories of noisy dynamical systems when decisions depend on that inference.**

## Non-Negotiable Constraints

- One question per post
- One reusable rule per post  
- Explicit failure modes only
- Decision quantities stated upfront
- No motivation, no history, no hype

## The 10 Questions

| Day | Title | Question | System |
|-----|-------|----------|--------|
| 00 | Scope and Intent | — | — |
| 01 | Distributions Are Not Dynamics | Does fitting the correct distribution imply correct transition dynamics? | Double-well Langevin |
| 02 | What Is a State? | TBD | TBD |
| 03 | Usable Reaction Coordinates | TBD | TBD |
| 04 | Identifiability Limits | TBD | TBD |
| 05 | Controllability | TBD | TBD |
| 06 | When ML Helps | TBD | TBD |
| 07 | Molecular Reality Check | TBD | TBD |
| 08 | Inference to Intervention | TBD | TBD |
| 09 | Materials Translation | TBD | TBD |

## Workflow

### For Each Day:

1. **Write** in `blog/dynamics-10/dayXX-title.md`
2. **Analyze** in `analysis/dynamics-10/dayXX_script.py`
3. **Generate figures** → `assets/img/blog/dynamics-10/dayXX/`
4. **Embed figures** with relative paths in markdown
5. **Export to HTML** only when ready to publish

### Files Never Change

- Filenames are stable
- Titles can change, filenames do not
- One file = one day = one question

## Folder Structure

```
portfolio/
├── blog/dynamics-10/           # Markdown source (truth)
│   ├── template.md
│   ├── day00-scope-and-intent.md
│   └── day01-distributions-vs-dynamics.md
│       ...
├── assets/img/blog/dynamics-10/   # Generated figures
│   ├── day01/
│   │   ├── double_well_potential.png
│   │   └── mfpt_comparison.png
│   └── day02/
│       ...
└── analysis/dynamics-10/       # Analysis code
    ├── utils.py                # Shared utilities
    ├── day01_double_well.py    # Figure generation
    └── README.md
```

## Publishing Rule

**Markdown is the source of truth.**  
HTML is a rendering artefact.

Do not touch HTML until the markdown is complete and figures are generated.

## Template Structure

Every post follows this structure:

1. What this post is actually about (anti-framing)
2. The Question (falsifiable)
3. The System (minimal)
4. Models / Representations Compared
5. What Works (baseline competence)
6. Where It Fails (concrete divergence)
7. Failure Mode Analysis (why it fails)
8. Decision Consequences (real errors)
9. Reusable Rule (generalization)
10. Bridge to Next Question (constraint, not hype)

## Publishability Test

A post is **not publishable** if:
- The question is not falsifiable
- No failure mode is demonstrated
- The failure mode cannot be explained
- No decision consequence is shown
- The reusable rule hedges or equivocates

If any of these apply, **do not publish**.
