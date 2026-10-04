"""Build simple static documentation; no hosted model inference."""

import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site"
OUTPUT.mkdir(exist_ok=True)
PAGES = [
    "usage",
    "artifacts",
    "training",
    "evaluation",
    "migration",
    "limits",
    "delivery-0.3.0",
]
NAV = (
    '<a href="index.html">PairJudge</a>'
    + "".join(
        f'<a href="{name}.html">{html.escape(name.replace("-", " ").title())}</a>'
        for name in PAGES[:6]
    )
    + '<a href="https://github.com/DaoyuanLi2816/pairjudge">GitHub</a>'
)
STYLE = """body{margin:0;background:#f3f6fa;color:#172e42;font-family:system-ui,sans-serif;line-height:1.65}nav{padding:18px 24px;background:#10243a;display:flex;gap:20px;flex-wrap:wrap}nav a{color:#e6f3ff;text-decoration:none;font-size:14px}main{max-width:1000px;margin:24px auto;padding:28px;background:white;border-radius:14px}a{color:#087f70}img{max-width:100%;height:auto}pre{padding:18px;background:#10243a;color:#eef6ff;overflow:auto;border-radius:9px}code{font-size:13px}table{border-collapse:collapse;width:100%;display:block;overflow:auto}th,td{padding:12px;border:1px solid #d2dce5}blockquote{border-left:4px solid #087f70;margin-left:0;padding-left:20px;color:#496379}h1,h2,h3{line-height:1.25}h2{margin-top:36px}@media(max-width:640px){main{margin:12px;padding:18px}nav{gap:12px}}"""


def render(source, name):
    text = source.read_text(encoding="utf-8")
    text = text.replace('<div align="center">', '<div align="center" markdown="1">')
    text = re.sub(
        r"```mermaid[\s\S]*?```",
        '<img src="workflow.svg" alt="Human-label training, teacher probabilities, soft-label student and swap inference">',
        text,
    )
    for page in PAGES:
        text = text.replace(f"docs/{page}.md", f"{page}.html").replace(
            f"({page}.md)", f"({page}.html)"
        )
    # Non-page repository references remain real GitHub links.
    for prefix in ("competition/", "examples/", "tests/", "src/", "docs/reports/"):
        text = re.sub(
            r"\]\((" + re.escape(prefix) + r"[^)]+)\)",
            lambda match: (
                "](https://github.com/DaoyuanLi2816/pairjudge/blob/main/"
                + match.group(1)
                + ")"
            ),
            text,
        )
    text = text.replace(
        "](CONTRIBUTING.md)",
        "](https://github.com/DaoyuanLi2816/pairjudge/blob/main/CONTRIBUTING.md)",
    ).replace(
        "](LICENSE)", "](https://github.com/DaoyuanLi2816/pairjudge/blob/main/LICENSE)"
    )
    body = markdown.markdown(
        text, extensions=["fenced_code", "tables", "toc", "md_in_html"]
    )
    title = name.replace("-", " ").title()
    (OUTPUT / f"{name}.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · PairJudge</title><style>{STYLE}</style><nav>{NAV}</nav><main>{body}</main></html>',
        encoding="utf-8",
    )


render(ROOT / "README.md", "index")
for page in PAGES:
    render(ROOT / "docs" / f"{page}.md", page)
shutil.copy2(ROOT / "docs/banner.svg", OUTPUT / "banner.svg")
shutil.copy2(ROOT / "docs/workflow.svg", OUTPUT / "workflow.svg")
if (ROOT / "docs/reports").exists():
    shutil.copytree(ROOT / "docs/reports", OUTPUT / "reports", dirs_exist_ok=True)
(OUTPUT / ".nojekyll").write_text("", encoding="utf-8")
print(f"Built {len(PAGES) + 1} static pages; all inference remains local.")
