"""Build report/Project_Report.pdf from results/summary.json + figures (HTML -> PDF with headless Chromium)."""
import base64, json, os, sys

sys.path.insert(0, "src")
from common import FAMILIES  # noqa

S = json.load(open("results/summary.json"))
B = json.load(open("results/baselines.json"))
STY = json.load(open("results/rq4_stylo.json"))
TOP = json.load(open("results/rq4_top_ngrams.json"))
OPN = json.load(open("results/rq4_openers.json"))


def J(n):
    p = f"results/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def pct(v, d=1):
    return "n/a" if v is None else f"{100 * v:.{d}f}"


def img(path, width="100%"):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def fig(path, cap, n, width="100%"):
    return f'<figure>{img(path, width)}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>'


exec(open("report/report_text.py").read())  # defines HTML_BODY(S, B, STY, TOP, OPN, helpers)

CSS = """
@page { size: Letter; margin: 0.8in 0.8in 0.8in 0.8in; }
body { font-family: 'Times New Roman', 'Liberation Serif', serif; font-size: 10.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt 0; }
.sub { text-align: center; font-size: 10.5pt; margin-bottom: 2pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt 0; }
p { margin: 4pt 0 6pt 0; text-align: justify; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 9pt; }
th, td { border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 2.5pt 6pt; text-align: center; }
th { background: #f1f1ee; }
td.l, th.l { text-align: left; }
caption { caption-side: top; font-size: 9pt; margin-bottom: 3pt; text-align: left; }
figure { margin: 6pt 0 8pt 0; page-break-inside: avoid; }
figcaption { font-size: 9pt; color: #333; margin-top: 2pt; }
.abstract { margin: 8pt 24pt; font-size: 10pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; }
ul { margin: 2pt 0 6pt 0; padding-left: 18pt; } li { margin: 1pt 0; }
.ex td { text-align: left; font-size: 8.4pt; vertical-align: top; }
.small { font-size: 8.8pt; color: #333; }
.box { border: 1px solid #ccc; background: #fafaf7; padding: 5pt 8pt; margin: 6pt 0; font-size: 9.6pt; }
"""

html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{HTML_BODY()}</body></html>"
open("report/Project_Report.html", "w").write(html)

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page()
    pg.goto("file://" + os.path.abspath("report/Project_Report.html"))
    pg.pdf(path="report/Project_Report.pdf", format="Letter", print_background=True,
           margin=dict(top="0.75in", bottom="0.75in", left="0.8in", right="0.8in"),
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#666'>"
                           "<span class='pageNumber'></span></div>")
    br.close()
print("wrote report/Project_Report.pdf")
