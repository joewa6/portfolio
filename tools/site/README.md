# Site Build Tools

This folder contains the Markdown-to-HTML build system for GitHub Pages.

## Structure

```
tools/site/
├── build.py           # Main build script
└── templates/         # Jinja2 templates
    ├── md_base.html   # Base template with styling
    └── post.html      # Post layout template
```

## Usage

### Local Build

```bash
# Install dependencies
pip install -r requirements-site.txt

# Build all markdown → HTML
python tools/site/build.py
```

This converts all `.md` files in `blog/dynamics-10/` to `.html` in the same directory.

### What It Does

1. Parses YAML frontmatter from markdown files
2. Converts markdown content to HTML
3. Applies Jinja2 templates with site styling
4. Outputs HTML next to markdown files (e.g., `day01-....md` → `day01-....html`)
5. Creates `.nojekyll` to prevent Jekyll interference

### GitHub Actions

The `.github/workflows/pages.yml` workflow automatically:
1. Installs Python dependencies
2. Runs `build.py` on every push to `main`
3. Deploys to GitHub Pages

No manual build needed for deployment.

## Template System

**md_base.html** — Base template
- Contains full CSS from main site
- Provides navigation header
- Sets up semantic HTML structure

**post.html** — Post template (extends md_base.html)
- Displays title from frontmatter
- Shows metadata line (day · system · question)
- Renders markdown content

## Frontmatter Fields

All DYNAMICS-10 posts use YAML frontmatter:

```yaml
---
title: "Your Title"
series: "DYNAMICS-10"
day: "01"
question: "Your question?"
decision_quantities:
  - "MFPT"
  - "state populations"
system: "double-well Langevin"
---
```

The build script uses these to generate metadata lines and structured content.

## Adding New Posts

1. Create `blog/dynamics-10/dayXX-title.md`
2. Add frontmatter header
3. Write in markdown
4. Run `python tools/site/build.py` locally to test
5. Commit both `.md` and generated `.html`
6. Push to trigger GitHub Actions deployment

## Non-Negotiables

- Markdown is source of truth
- HTML is generated artifact (committed for GitHub Pages)
- Template files never edited directly
- All blog CSS lives in `md_base.html` template
