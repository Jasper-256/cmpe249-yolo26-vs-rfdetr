"""Render report/REPORT.md to a print-styled PDF (report/REPORT.pdf) via headless Google Chrome.

  uv run --with markdown-it-py --with mdit-py-plugins --with playwright python scripts/build_pdf.py

Markdown -> HTML with markdown-it (CommonMark + GFM tables, GitHub-style heading anchors), a print stylesheet,
then Playwright drives the system Chrome (channel="chrome") to print Letter pages with a running footer.
"""
import os
import re
import unicodedata

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "report", "REPORT.md")
HTML = os.path.join(ROOT, "report", "REPORT.html")
PDF = os.path.join(ROOT, "report", "REPORT.pdf")


def github_slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


CSS = r"""
@page { size: Letter; margin: 0.62in 0.6in 0.7in 0.6in; }
:root {
  --ink: #16181d; --ink2: #4a4f59; --muted: #6b717c; --rule: #d9dce1; --accent: #2a5db0;
  --shade: #f4f6f9; --callout: #eef3fb;
}
html { font-size: 9.6pt; }
body { font-family: "Bitstream Charter", "Charter", "DejaVu Serif", serif; color: var(--ink);
       line-height: 1.38; margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact;
       hyphens: auto; text-rendering: optimizeLegibility; }
h1, h2, h3, h4 { font-family: "Noto Sans", "DejaVu Sans", sans-serif; color: var(--ink);
                 break-after: avoid; page-break-after: avoid; line-height: 1.22; }
h1 { font-size: 19pt; font-weight: 700; margin: 0 0 4pt; letter-spacing: -0.2pt; }
h2 { font-size: 13pt; font-weight: 700; margin: 16pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1.2pt solid var(--accent); }
h3 { font-size: 10.8pt; font-weight: 700; margin: 12pt 0 4pt; color: #1f3a6b; }
p { margin: 0 0 6pt; text-align: justify; }
a { color: var(--accent); text-decoration: none; }
strong { font-weight: 700; }
hr { display: none; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; }
li { margin: 0 0 3pt; text-align: justify; }
li > p { margin-bottom: 3pt; }

/* title block = first h1 + following paragraph */
.title-block { border-bottom: 0.6pt solid var(--rule); padding-bottom: 6pt; margin-bottom: 10pt; }
.title-block p { font-family: "Noto Sans", sans-serif; color: var(--ink2); font-size: 9.2pt;
                 text-align: left; margin: 0; }

/* TL;DR callout */
blockquote { margin: 0 0 10pt; padding: 7pt 10pt 4pt; background: var(--callout);
             border-left: 3pt solid var(--accent); border-radius: 2pt; font-size: 9.1pt; }
blockquote p, blockquote li { text-align: left; }
blockquote ol { padding-left: 15pt; }

code { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 0.86em;
       background: var(--shade); padding: 0.5pt 2.5pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { background: var(--shade); border: 0.6pt solid var(--rule); border-radius: 3pt;
      padding: 6pt 8pt; margin: 4pt 0 9pt; overflow: hidden; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 6.55pt; line-height: 1.36; white-space: pre-wrap;
           overflow-wrap: anywhere; }

table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt;
        font-family: "Noto Sans", "DejaVu Sans", sans-serif; font-size: 7.15pt; line-height: 1.28;
        font-variant-numeric: tabular-nums; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { background: #e7ebf2; color: var(--ink); font-weight: 700; text-align: left;
     border-bottom: 0.9pt solid #9aa3b2; padding: 2.6pt 4pt; vertical-align: bottom; }
td { border-bottom: 0.5pt solid var(--rule); padding: 2.2pt 4pt; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafbfc; }
td code, th code { font-size: 0.92em; }
table.wide { font-size: 6.45pt; }
table.wide th, table.wide td { padding: 2pt 2.6pt; }
table.numeric td:not(:first-child):not(:nth-child(2)),
table.numeric th:not(:first-child):not(:nth-child(2)) { text-align: right; }
table.numeric td:first-child, table.numeric td:nth-child(2) { white-space: nowrap; }

figure, p.figure { text-align: center; margin: 6pt 0 2pt; break-inside: avoid; }
img { max-width: 94%; display: block; margin: 0 auto; }
img[src*="qual_"] { max-width: 74%; }
p.caption { font-family: "Noto Sans", sans-serif; font-size: 8pt; color: var(--ink2);
            text-align: center; margin: 2pt 6% 10pt; }
li img { margin: 5pt auto 4pt; }

.refs ol, .refs li { font-size: 8.4pt; }
h4.minor { font-size: 9.6pt; margin: 9pt 0 3pt; color: var(--ink); }
p.rag, li.rag { text-align: left; }
ol { padding-left: 21pt; }
section.refs li { text-align: left; font-size: 8.8pt; margin-bottom: 2pt; }
table, div.keep { break-inside: avoid; page-break-inside: avoid; }
.pagebreak { break-before: page; page-break-before: always; }
section.appendix { font-size: 9pt; }
"""

FOOTER = """
<div style="width:100%; font-family:'Noto Sans',sans-serif; font-size:7pt; color:#6b717c;
            padding:0 0.6in; display:flex; justify-content:space-between;">
  <span>YOLO26 vs RF-DETR — CMPE 249, Track A</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>"""


def postprocess(html: str) -> str:
    # title block: first <h1> + the byline paragraph after it
    html = re.sub(r"(<h1[^>]*>.*?</h1>\s*)<p>(.*?)</p>",
                  lambda m: m.group(1) + '<p class="byline">' + m.group(2).replace("\n", "<br>") + "</p>",
                  html, count=1, flags=re.S)
    html = re.sub(r'(<h1[^>]*>.*?</h1>\s*<p class="byline">.*?</p>)', r'<div class="title-block">\1</div>', html, count=1, flags=re.S)
    # standalone images -> figure paragraphs; the italic paragraph right after a figure -> caption
    html = re.sub(r"<p>(<img [^>]+>)</p>", r'<p class="figure">\1</p>', html)
    html = re.sub(r'(<p class="figure">.*?</p>\s*)<p><em>(.*?)</em></p>', r'\1<p class="caption">\2</p>', html, flags=re.S)
    # tables: mark wide ones (many columns) and numeric ones
    def tag_table(m):
        t = m.group(0)
        head = re.search(r"<thead>.*?</thead>", t, re.S)
        ncol = head.group(0).count("<th") if head else 0
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", t, re.S)]
        num = sum(bool(re.fullmatch(r"[−+\-–]?[\d.,]+ ?%?|–|—", c)) for c in cells)
        cls = []
        if ncol >= 11:
            cls.append("wide")
        if ncol >= 7 and cells and num / len(cells) > 0.45:
            cls.append("numeric")
        return t.replace("<table>", f'<table class="{" ".join(cls)}">', 1) if cls else t
    html = re.sub(r"<table>.*?</table>", tag_table, html, flags=re.S)
    # standalone bold paragraphs act as minor headings -> keep with next block
    html = re.sub(r"<p><strong>([^<]{2,60})</strong></p>", r'<h4 class="minor">\1</h4>', html)
    # paragraphs / list items dense with inline code: ragged-right to avoid stretched justification
    html = re.sub(r"<(p|li)>((?:(?!</\1>).)*?<code>(?:(?!</\1>).)*?<code>(?:(?!</\1>).)*?)</\1>",
                  r'<\1 class="rag">\2</\1>', html, flags=re.S)
    # keep a short lead-in paragraph (ending in ':') together with the table right after it
    html = re.sub(r'(<p(?: class="[^"]*")?>(?:(?!</p>).){0,400}?:</p>\s*<table.*?</table>)',
                  r'<div class="keep">\1</div>', html, flags=re.S)
    # reference lists: wrap the References section so it can be styled (left-aligned, smaller)
    html = re.sub(r'(<h2 id="references">.*?)(<h2 )', r'<section class="refs">\1</section>\2', html, count=1, flags=re.S)
    return html


def main():
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")
    md.use(anchors_plugin, max_level=3, slug_func=github_slug)
    text = open(SRC, encoding="utf-8").read()
    body = postprocess(md.render(text))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>YOLO26 vs RF-DETR: COCO and KITTI</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    open(HTML, "w", encoding="utf-8").write(html)

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page()
            page.goto("file://" + HTML, wait_until="networkidle")
            page.pdf(path=PDF, format="Letter", print_background=True, prefer_css_page_size=True,
                     display_header_footer=True, header_template="<div></div>", footer_template=FOOTER,
                     margin={"top": "0.62in", "bottom": "0.7in", "left": "0.6in", "right": "0.6in"})
        finally:
            browser.close()
    print("wrote", PDF)


if __name__ == "__main__":
    main()
