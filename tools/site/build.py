#!/usr/bin/env python3
"""
Build script: Markdown → HTML for GitHub Pages

Converts all .md files in blog/dynamics-10/ to .html using Jinja2 templates.
Preserves frontmatter metadata for title, day, system, question.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List

import frontmatter
from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown


ROOT = pathlib.Path(__file__).resolve().parents[2]  # portfolio/
TEMPLATES = ROOT / "tools" / "site" / "templates"

# Inputs (Markdown)
MD_DIRS = [
    ROOT / "blog" / "dynamics-10",
]


@dataclass
class Post:
    md_path: pathlib.Path
    html_path: pathlib.Path
    title: str
    meta_line: str
    html: str


def compute_base_url(from_path: pathlib.Path) -> str:
    """
    For linking assets from nested pages.
    If page is portfolio/blog/dynamics-10/x.html, base_url should be ../../
    """
    rel = from_path.relative_to(ROOT)
    depth = len(rel.parents) - 1  # number of folders between file and root
    return "../" * depth


def render_markdown(md_text: str) -> str:
    """
    Convert markdown to HTML with code highlighting and tables.
    """
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "codehilite",  # requires pygments
        ],
        extension_configs={
            "codehilite": {"guess_lang": False},
            "toc": {"permalink": True},
        },
        output_format="html5",
    )
    return md.convert(md_text)


def load_posts() -> List[Post]:
    """
    Load all markdown files from MD_DIRS and parse frontmatter.
    """
    posts: List[Post] = []

    for md_dir in MD_DIRS:
        if not md_dir.exists():
            continue

        for md_path in sorted(md_dir.glob("*.md")):
            # Skip template and README
            if md_path.stem in ("template", "README"):
                continue

            fm = frontmatter.load(md_path)
            title = str(fm.get("title") or md_path.stem.replace("-", " ").title())

            # Optional meta line: day/system/question if present
            day = fm.get("day")
            system = fm.get("system")
            question = fm.get("question")
            meta_bits = []
            if day:
                meta_bits.append(f"Day {day}")
            if system:
                meta_bits.append(str(system))
            if question:
                meta_bits.append(str(question))
            meta_line = " · ".join(meta_bits)

            html = render_markdown(fm.content)

            html_path = md_path.with_suffix(".html")
            posts.append(
                Post(
                    md_path=md_path,
                    html_path=html_path,
                    title=title,
                    meta_line=meta_line,
                    html=html,
                )
            )

    return posts


def build() -> None:
    """
    Main build function: convert all markdown files to HTML.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("post.html")

    posts = load_posts()
    if not posts:
        print("No posts found.")
        return

    for p in posts:
        base_url = compute_base_url(p.html_path)
        rendered = tpl.render(
            title=p.title,
            meta=p.meta_line,
            content=p.html,
            base_url=base_url,
        )
        p.html_path.write_text(rendered, encoding="utf-8")
        print(f"✓ {p.html_path.relative_to(ROOT)}")

    # Prevent Jekyll from interfering with GitHub Pages
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print("✓ .nojekyll")
    
    print(f"\nBuilt {len(posts)} posts.")


if __name__ == "__main__":
    build()
