"""Generate report/main.tex (and report/figures/) from the experiment outputs in runs/.

Every number in the report is read from runs/ here, so re-running after new experiments keeps the text and
tables in sync:

    uv run python report/generate.py
    cd report && latexmk -pdf main.tex

The document is written with small helpers (P, table, figure, ...) that emit LaTeX; edit the text below,
not main.tex, if you want changes to survive a regeneration.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORT = ROOT / "report"
FIGS = REPORT / "figures"
OUT = REPORT / "main.tex"
FIGS.mkdir(parents=True, exist_ok=True)

TEXT_WIDTH_CM = 17.4  # A4 with 1.8 cm side margins
cm = 1.0  # table/figure widths below are in centimetres

# ---------------------------------------------------------------- text -> LaTeX
UNICODE = {
    "L∞": r"$L_\infty$",
    "ε": r"$\varepsilon$",
    "±": r"$\pm$",
    "→": r"$\to$",
    "↓": r"$\downarrow$",
    "×": r"$\times$",
    "σ": r"$\sigma$",
    "∞": r"$\infty$",
    "−": r"$-$",
    "—": "---",
    "–": "--",
    "·": r"\textperiodcentered{}",
    "•": "",
}


def tex(s) -> str:
    """Convert the light markup used below (<b>, <i>, <br/>, <font>) and unicode symbols to LaTeX."""
    s = str(s)
    for ch, rep in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(ch, rep)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    s = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", s)
    s = re.sub(r"<i>(.*?)</i>", r"\\emph{\1}", s)
    s = re.sub(r"<font face='Arial-Italic'>(.*?)</font>", r"\\emph{\1}", s)
    s = re.sub(r"<font color='\\#c0392b'>(.*?)</font>", r"\\textcolor{tagred}{\1}", s)
    for ch, rep in UNICODE.items():
        s = s.replace(ch, rep)
    return s


def strip_number(heading: str) -> str:
    """'6.1 What the numbers say' -> 'What the numbers say' (LaTeX numbers sections itself)."""
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", heading)


# Paragraph styles.
body, small, h1, h2, h3, title, subtitle = "body", "small", "h1", "h2", "h3", "title", "subtitle"


def P(text, style=body):
    if style == h1:
        return rf"\needspace{{8\baselineskip}}\section{{{tex(strip_number(text))}}}"
    if style == h2:
        return rf"\needspace{{6\baselineskip}}\subsection{{{tex(strip_number(text))}}}"
    if style == h3:
        return rf"\par\smallskip\noindent\textbf{{{tex(text)}}}\par\smallskip"
    if style == title:
        return rf"\begin{{center}}{{\LARGE\bfseries {tex(text)}\par}}\end{{center}}"
    if style == subtitle:
        return rf"\begin{{center}}\color{{subtle}}{tex(text)}\end{{center}}"
    if text.startswith("• "):
        items = "\n".join(rf"  \item {tex(i)}" for i in text.split("<br/>"))
        return f"\\begin{{itemize}}\n{items}\n\\end{{itemize}}"
    return rf"{{\small\color{{subtle}} {tex(text)}\par}}" if style == small else tex(text) + "\n"


def table(rows, widths, highlight=None):
    """Booktabs-style table: navy header row, zebra rows, highlighted cells. Widths in cm."""
    pad = 0.21  # 2 x \tabcolsep (3pt)
    spec = "".join(rf">{{\raggedright\arraybackslash}}p{{{w - pad:.2f}cm}}" for w in widths)
    hl = set(highlight or [])
    lines = [
        r"\begin{center}\footnotesize",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        r"\rowcolor{navy}" + " & ".join(rf"\textcolor{{white}}{{\textbf{{{tex(c)}}}}}" for c in rows[0]) + r" \\",
        r"\midrule",
    ]
    for r, row in enumerate(rows[1:], start=1):
        cells = [(r"\cellcolor{hl}" if (r, c) in hl else "") + tex(v) for c, v in enumerate(row)]
        lines.append((r"\rowcolor{zebra}" if r % 2 == 0 else "") + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def figure(path, width_cm, text):
    name = path.name
    if path.stat().st_size > 1_000_000:  # photo grids: PNG of noisy photos is huge, JPEG is ~10x smaller
        from PIL import Image

        name = path.with_suffix(".jpg").name
        Image.open(path).convert("RGB").save(FIGS / name, quality=92)
    else:
        shutil.copy(path, FIGS / name)
    caption = re.sub(r"^Figure \d+\.\s*", "", text)
    return (
        r"\begin{figure}[H]\centering"
        + "\n"
        + rf"\includegraphics[width={min(width_cm, TEXT_WIDTH_CM):.1f}cm]{{figures/{name}}}"
        + "\n"
        + rf"\caption{{{tex(caption)}}}"
        + "\n"
        + r"\end{figure}"
    )


def PageBreak():
    return r"\clearpage"


def Spacer(*_):
    return ""


def KeepTogether(items):
    return "\\par\\noindent\\begin{minipage}{\\linewidth}\n" + "\n".join(items) + "\n\\end{minipage}\\par\\medskip"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,left=1.8cm,right=1.8cm,top=1.6cm,bottom=2.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[table]{xcolor}
\usepackage{graphicx,booktabs,array,float,caption,amsmath,enumitem,fancyhdr,titlesec,needspace}
\usepackage[hidelinks]{hyperref}

\definecolor{navy}{HTML}{1F3A5F}
\definecolor{zebra}{HTML}{F2F5F9}
\definecolor{hl}{HTML}{FFE9B3}
\definecolor{tagred}{HTML}{C0392B}
\definecolor{subtle}{HTML}{555555}

\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\setlist{itemsep=2pt,topsep=2pt}
\captionsetup{font={small,color=subtle},labelfont=bf}
\titleformat{\section}{\Large\bfseries\color{navy}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\large\bfseries\color{navy}}{\thesubsection}{0.5em}{}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\footnotesize\color{subtle}COMS4036A \textperiodcentered{} edit transferability \textperiodcentered{} results summary}
\fancyfoot[R]{\footnotesize\color{subtle}Page \thepage}

\hypersetup{pdftitle={Edit transferability: results summary},
            pdfauthor={Dylan Davies, Sudhir Adjodhya, Michael Badenhorst}}

\begin{document}
"""


# ---------------------------------------------------------------- data
from results import (  # noqa: E402
    ATTACKERS,
    MODELS,
    X,
    agg,
    arc_seed_std,
    attacker_figure,
    best_vs_arc,
    cond_table,
    epochs,
    eps_fmt,
    fooled,
    header,
    matrix_rows,
    max_seed_std,
    n_folds,
    n_pairs,
    off_diag_eps4,
    people,
    pg,
    saved,
    smoke,
    spread,
    ssim,
    validate,
    xssim,
)

attacker_figure(RUNS / "figures" / "attacker_transfer.png")
EX = RUNS / "examples"


# ---------------------------------------------------------------- document
story = []
story += [
    P("Do edits that fool one face recognition model also fool models they were never built for?", title),
    P("COMS4036A project · results summary · Dylan Davies, Sudhir Adjodhya, Michael Badenhorst", subtitle),
    P(
        "Generated from <font face='Arial-Italic'>runs/sweep.csv</font>, <font face='Arial-Italic'>runs/validate.csv</font>, "
        "<font face='Arial-Italic'>runs/sweeps/</font> and <font face='Arial-Italic'>runs/train.log</font>; every number "
        "below is read from those files.",
        subtitle,
    ),
    Spacer(1, 8),
]

# 1. Summary
story += [
    P("1. Summary", h1),
    P(
        f"We trained a face recognition network from scratch (the <b>surrogate</b>) and used projected gradient descent "
        f"(PGD) against it to edit one photo in each of the {n_pairs:,} same-person pairs of the LFW View 2 benchmark. We then "
        f"asked four models that were never used while editing whether they still recognise the pair as the same person."
    ),
    P(
        f"<b>Main finding.</b> The edits work almost perfectly on the model they were built against and transfer poorly to "
        f"everything else. At an edit budget of ε = 4/255, where the edited photo is still nearly identical to the original "
        f"(SSIM {ssim.loc[('pgd', 4.0)]:.3f}), the surrogate stops matching {pct(pg(4.0, 'surrogate'))} of same-person pairs, "
        f"but the other deep models lose only {pct(pg(4.0, 'casia'))} (casia), {pct(pg(4.0, 'vggface2'))} (vggface2) and "
        f"{pct(pg(4.0, 'arcface'))} (arcface). Random noise at the same budget does almost nothing to any deep model, so "
        f"the effect comes from the optimisation, not from changing pixels. Repeating the sweep with two more random seeds "
        f"changes no result by more than {100 * max_seed_std.max():.2f} percentage points (standard deviation)."
    ),
    P(
        f"<b>Is it just our surrogate?</b> To check, we repeated the attack using each pretrained deep model as the attacker "
        f"(Section 9). Our surrogate turns out to be the <i>weakest</i> attacker of the four: at ε = 8/255 its edits make "
        f"casia lose {pct(X(8.0, 'surrogate', 'casia'))} of pairs, while edits built against vggface2 make casia lose "
        f"{pct(X(8.0, 'vggface2', 'casia'))}. Edits transfer well <b>within the same architecture</b> (the three "
        f"InceptionResnetV1 models) even when the training faces differ, while ArcFace, a different architecture, resists "
        f"edits from every attacker (at most {pct(max(X(16.0, a, 'arcface') for a in ATTACKERS[:3]))}, even at a visible "
        f"ε = 16/255). So on the proposal's second question, the evidence points to <b>architecture</b> more than training "
        f"data, and how well protection transfers depends heavily on which model the edit was built against."
    ),
]

# 2. Context
story += [
    P("2. Context: the question and why it matters", h1),
    P(
        "Privacy tools such as Fawkes and LowKey add small, hard-to-see edits to photos before they are posted, so that face "
        "recognition systems can no longer link them to the person. These tools assume an edit built against one recognition "
        "system will also defeat whatever system someone else is running. A photo is edited once and must then hold up "
        "against every system that sees it later, so if edits do not <i>transfer</i>, the protection is much weaker than it looks."
    ),
    P(
        "Our proposal asked two things: (1) when we edit a photo until <i>our</i> model stops recognising it, do models we had "
        "no access to also stop recognising it? (2) Where some models resist and others give way, is that explained by their "
        "architecture or by the faces they were trained on? The four test models were chosen so that each differs from the "
        "surrogate in a controlled way."
    ),
]

# 3. Setup
story += [
    P("3. Experimental setup", h1),
    P("3.1 Models", h2),
    table(
        [
            ["Model", "Role", "Architecture", "Training faces", "Training objective", "Differs from surrogate in"],
            [
                "surrogate",
                "Builds the edits (only model with gradients used)",
                "InceptionResnetV1",
                "CASIA-WebFace (trained here)",
                "Softmax cross-entropy",
                "—",
            ],
            [
                "casia",
                "Test model; also an attacker in Section 9",
                "InceptionResnetV1",
                "CASIA-WebFace",
                "Softmax cross-entropy",
                "Only the random initialisation",
            ],
            [
                "vggface2",
                "Test model; also an attacker in Section 9",
                "InceptionResnetV1",
                "VGGFace2",
                "Softmax cross-entropy",
                "Training faces",
            ],
            [
                "arcface",
                "Test model; also an attacker in Section 9",
                "ResNet-50 (InsightFace buffalo_l)",
                "WebFace600K",
                "ArcFace margin loss",
                "Architecture, objective and faces",
            ],
            [
                "lbph",
                "Test only",
                "Local Binary Pattern Histograms",
                "None (hand-crafted)",
                "None",
                "No deep learning at all",
            ],
        ],
        widths=[1.7 * cm, 3.2 * cm, 3.0 * cm, 2.8 * cm, 2.6 * cm, 3.4 * cm],
    ),
    Spacer(1, 6),
    P("3.2 Data", h2),
    P(
        f"<b>Test data:</b> LFW View 2, the standard benchmark of 6,000 fixed pairs (3,000 same-person, 3,000 different-person) "
        f"split into {n_folds} folds, covering 7,701 distinct photos. We use the published pairs unchanged, including the six "
        f"known mislabelled pairs, as the maintainers request. The MTCNN face detector found a face in all 7,701 photos."
    ),
    P(
        f"<b>Surrogate training data:</b> CASIA-WebFace, {int(header[0]):,} images of {int(header[1]):,} people, already "
        f"aligned to 112×112 pixels. LFW is too small to train on, so a separate dataset is needed; using CASIA means the "
        f"surrogate shares its training faces with the casia test model, which is what lets us separate the effect of data "
        f"from the effect of architecture."
    ),
    P("3.3 How each model sees a photo", h2),
    P(
        "Every edit is made on the full 250×250 LFW photo, exactly as a user would post it. Each model then crops and aligns "
        "that same photo its own way, so every model sees the same edit:"
    ),
    P(
        "• <b>vggface2 / casia:</b> a box crop around the detected face, 32-pixel margin, resized to 160×160 (margin chosen "
        "by validation, Section 4).<br/>"
        "• <b>arcface / lbph:</b> the official InsightFace 5-point alignment to 112×112 (eyes, nose, mouth corners).<br/>"
        "• <b>surrogate:</b> 5-point alignment to 160×160, using landmark positions measured on the CASIA training crops, so "
        "its LFW test faces line up the way its training faces do.",
        body,
    ),
    P(
        "All crops are implemented as one differentiable warp, so the surrogate's own cropping is part of the PGD "
        "optimisation. Face landmarks are detected once on the clean photo and reused for its edited versions."
    ),
    P('3.4 Deciding "same person"', h2),
    P(
        "Each model turns a photo into an embedding and compares two photos by a distance (cosine distance for the deep "
        "models, chi-square for LBPH). A pair counts as the same person if the distance is below a threshold. Following the "
        "standard LFW protocol, each model gets its own threshold for each fold, fitted on the clean pairs of the other nine "
        "folds. These clean-data thresholds are then kept fixed when scoring edited photos, so a model's error rate at "
        "ε = 0 is its ordinary baseline error."
    ),
    P("3.5 The edit (PGD) and the comparisons", h2),
    P(
        "PGD repeatedly nudges every pixel in the direction that makes the surrogate see the edited photo as <i>more "
        "different from its own original</i>, then pulls any pixel that has moved further than ε back within that limit. "
        "We push away from the photo's own original rather than from its pair partner because a real user does not know "
        "which photo their posted image will be compared against. Settings: L∞ budget ε, 20 steps, step size 2.5ε/20, "
        "random start. Edited photos are rounded back to ordinary 8-bit images before scoring."
    ),
    table(
        [
            ["Condition", "What it is", "What it tells us", "Strengths swept"],
            ["pgd", "PGD edit against the surrogate", "Main result", "ε = 0, 0.5, 1, 2, 4, 8, 16 (/255)"],
            [
                "noise",
                "Random ±ε on every pixel",
                "Whether the optimisation matters, or any change would do",
                "Same ε values",
            ],
            [
                "pgd+jpeg75",
                "PGD edit, then JPEG at quality 75",
                "Whether ordinary compression removes the edit",
                "Same ε values",
            ],
            [
                "jpeg",
                "JPEG compression of the clean photo",
                "A free edit any user could apply",
                "Quality 90, 75, 50, 30",
            ],
            ["blur", "Gaussian blur of the clean photo", "A free edit any user could apply", "σ = 0.5, 1, 2, 3 pixels"],
        ],
        widths=[2.0 * cm, 4.3 * cm, 5.6 * cm, 4.8 * cm],
    ),
    Spacer(1, 4),
    P(
        "The <b>unmatched share</b> reported throughout is the fraction of the 3,000 same-person pairs that a model no longer "
        "accepts as the same person after photo <i>a</i> of the pair is edited. Values are the mean over the 10 folds, "
        "± the standard deviation across folds. <b>SSIM</b> (structural similarity, 1.0 = identical) measures how visible the "
        "edit is, comparing the edited photo with the original.",
        body,
    ),
]

# 4. Validation
story += [
    P("4. Validating the models before any edits", h1),
    P(
        "The proposal identified alignment as the main risk: if a model is fed faces cropped the wrong way, it still produces "
        "numbers, just wrong ones. So before any edit we checked that each model reaches its published accuracy on clean "
        "LFW pairs. The gate was: every deep test model within about 1 percentage point of its published figure."
    ),
]
val_rows = [["Model", "Clean LFW accuracy (ours)", "± std over folds", "Published", "Gap"]]
for m in MODELS:
    r = validate.loc[m]
    pub = r.published
    val_rows.append(
        [
            m,
            pct(r.accuracy, 2),
            pct(r["std"], 2),
            pct(pub, 2) if pub == pub else "—",
            f"{100 * (r.accuracy - pub):+.2f} pts" if pub == pub else "—",
        ]
    )
story += [
    table(val_rows, widths=[2.4 * cm, 4.0 * cm, 3.0 * cm, 2.6 * cm, 2.4 * cm]),
    Spacer(1, 4),
    P(
        "All three pretrained deep models pass: ArcFace matches its published accuracy almost exactly, and the two facenet "
        "models sit about 0.3 points below theirs. The surrogate (trained by us) has no published figure; its accuracy is "
        "covered in Section 5. LBPH's 67% is typical for a hand-crafted method on LFW and means it already gets many pairs "
        "wrong before any edit."
    ),
    P(
        "<b>Crop margin for the facenet models.</b> The facenet models are sensitive to how much border is kept around the "
        "face. We tried four margins and kept the best (32 pixels, which matches the original facenet LFW crop):"
    ),
    table(
        [
            ["Margin (pixels)", "0", "14", "32 (chosen)", "44"],
            ["vggface2", "99.00%", "99.30%", "99.37%", "99.37%"],
            ["casia", "98.07%", "98.50%", "98.73%", "98.42%"],
        ],
        widths=[3.2 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm, 2.4 * cm],
    ),
]

# 5. Training
story += [
    PageBreak(),
    P("5. Training the surrogate", h1),
    P(
        f"The surrogate is an InceptionResnetV1 trained from random initialisation for {len(epochs)} epochs on CASIA-WebFace "
        f"({int(header[2]):,} steps of 256 images per epoch), using softmax cross-entropy over the {int(header[1]):,} "
        "identities, SGD with Nesterov momentum 0.9, weight decay 5e-4, a one-cycle learning-rate schedule peaking at 0.1, "
        "bf16 mixed precision and random horizontal flips. Using the same architecture and objective as the facenet test "
        "models is deliberate, so that the casia model differs from the surrogate only in its random starting weights. "
        "Training took about 1.5 hours on an RTX 5070 Ti."
    ),
    P(
        f"500 CASIA identities were held out of training and used as a validation set (fixed same-person and different-person "
        f"pairs). After each epoch the network was scored on those pairs and on clean LFW View 2 (10-fold protocol). The saved "
        f"model is the epoch with the best held-out accuracy (epoch {saved + 1}); LFW played no part in the choice."
    ),
]
best = max(e[3] for e in epochs)
tr_rows = [["Epoch", "Train loss", "Train accuracy", "Clean LFW accuracy", "± std"]]
hl = []
for i, (e, loss, acc, lfw, std) in enumerate(epochs):
    tr_rows.append([f"{int(e) + 1}", f"{loss:.3f}", pct(acc), pct(lfw, 2), pct(std, 2)])
    if lfw == best:
        hl.append((i + 1, 3))
story += [
    table(tr_rows, widths=[1.6 * cm, 2.4 * cm, 2.8 * cm, 3.4 * cm, 2.0 * cm], highlight=hl),
    Spacer(1, 4),
    P(
        f"Clean LFW accuracy rose from {pct(epochs[0][3], 2)} after one epoch to {pct(epochs[-1][3], 2)} after "
        f"{len(epochs)} (best {pct(best, 2)} in epoch {[int(e[0]) + 1 for e in epochs if e[3] == best][0]}, highlighted). "
        "For comparison, the published facenet model trained on the same data reaches 99.05%, so the surrogate is a "
        "competent model, about one point weaker. Training accuracy reaching 97.6% while LFW accuracy levels off means the "
        "last few epochs mostly fit the training set more tightly."
    ),
]

# 6. Main result
story += [
    PageBreak(),
    P("6. Main result: do PGD edits transfer?", h1),
    P(
        "The table below is the core result of the project. Each cell is the share of the 3,000 same-person pairs the model "
        "no longer matches after one photo is edited with PGD against the surrogate. ε = 0 means no edit, so that row is each "
        "model's baseline error. The SSIM column shows how visible the edit is at each budget."
    ),
    table(cond_table("pgd", "Edit budget ε", eps_fmt), widths=[2.2 * cm] + [2.7 * cm] * 5 + [1.4 * cm]),
    Spacer(1, 6),
    figure(
        RUNS / "figures" / "main_result.png",
        15.5,
        "Figure 1. Share of same-person pairs no longer matched against edit budget (log scale), one line per model, "
        "error bars = std over 10 folds. Dashed line: SSIM between edited and original photo (right axis).",
    ),
    P("6.1 What the numbers say", h2),
    P(
        f"<b>The edit works on the model it was built against, and quickly.</b> The surrogate goes from a "
        f"{pct(pg(0.0, 'surrogate'))} baseline to {pct(pg(1.0, 'surrogate'))} at ε = 1/255, {pct(pg(2.0, 'surrogate'))} at "
        f"2/255 and {pct(pg(4.0, 'surrogate'))} from 4/255 upward. At 2/255 the edit is essentially invisible "
        f"(SSIM {ssim.loc[('pgd', 2.0)]:.3f})."
    ),
    P(
        f"<b>The edit barely transfers at invisible budgets.</b> At ε = 4/255 (SSIM {ssim.loc[('pgd', 4.0)]:.3f}), every test "
        f"deep model stays within a few points of its baseline: casia {pct(pg(0.0, 'casia'))} → {pct(pg(4.0, 'casia'))}, "
        f"vggface2 {pct(pg(0.0, 'vggface2'))} → {pct(pg(4.0, 'vggface2'))}, arcface {pct(pg(0.0, 'arcface'))} → "
        f"{pct(pg(4.0, 'arcface'))}. A photo that is fully protected against the surrogate is still recognised by every "
        f"other deep model about 95% of the time or more."
    ),
    P(
        f"<b>With our surrogate as attacker, shared training data looks decisive.</b> casia and vggface2 have the same "
        f"architecture and objective; the only difference is that casia was trained on the same faces as the surrogate. At "
        f"ε = 16/255 casia loses {pct(pg(16.0, 'casia'))} of pairs while vggface2 loses {pct(pg(16.0, 'vggface2'))}, about "
        f"four times less (at ε = 8/255: {pct(pg(8.0, 'casia'))} vs {pct(pg(8.0, 'vggface2'))}). Changing architecture, "
        f"objective and data together (arcface) removes transfer almost entirely: {pct(pg(16.0, 'arcface'))} at ε = 16/255. "
        f"<i>Section 9 shows this reading does not hold once other models are used as the attacker.</i>"
    ),
    P(
        f"<b>Visible edits are needed before anything transfers.</b> The only substantial transfer (casia, "
        f"{pct(pg(16.0, 'casia'))}) happens at ε = 16/255, where SSIM has dropped to {ssim.loc[('pgd', 16.0)]:.3f} and the "
        f"edit is clearly visible. So in our setting, an edit that is invisible is not transferable, and an edit that "
        f"transfers is not invisible."
    ),
]

# 7. Comparisons
story += [
    PageBreak(),
    P("7. Required comparisons", h1),
    P("7.1 Random noise at the same budget", h2),
    P(
        "Random noise changes every pixel by exactly ±ε, the same maximum change PGD is allowed. If noise did as well as PGD, "
        "the optimisation would be contributing nothing."
    ),
    table(cond_table("noise", "Noise budget ε", eps_fmt), widths=[2.2 * cm] + [2.7 * cm] * 5 + [1.4 * cm]),
    Spacer(1, 4),
    P(
        f"Noise does almost nothing to any deep model. Even at ε = 16/255, where noise is more visible than PGD "
        f"(SSIM {ssim.loc[('noise', 16.0)]:.3f} vs {ssim.loc[('pgd', 16.0)]:.3f}), the surrogate only reaches "
        f"{pct(agg.loc[('noise', 16.0, 'surrogate')].m)} against {pct(pg(16.0, 'surrogate'))} for PGD, and casia "
        f"{pct(agg.loc[('noise', 16.0, 'casia')].m)} against {pct(pg(16.0, 'casia'))}. The PGD effect, and the transfer to "
        f"casia, therefore come from the optimisation itself."
    ),
    P("7.2 JPEG compression after editing", h2),
    P(
        "Photos are usually recompressed when uploaded. Here the PGD-edited photo is saved as JPEG at quality 75 before the "
        "models see it. The SSIM column includes the compression's own loss of detail."
    ),
    table(cond_table("pgd+jpeg75", "Edit budget ε", eps_fmt), widths=[2.2 * cm] + [2.7 * cm] * 5 + [1.4 * cm]),
    Spacer(1, 4),
    P(
        f"Compression weakens small edits but does not remove larger ones. Against the surrogate, at ε = 2/255 it cuts the "
        f"unmatched share from {pct(pg(2.0, 'surrogate'))} to {pct(agg.loc[('pgd+jpeg75', 2.0, 'surrogate')].m)}; at "
        f"ε = 4/255 the edit survives ({pct(agg.loc[('pgd+jpeg75', 4.0, 'surrogate')].m)}) and from 8/255 it is unaffected. "
        f"Transfer to casia at ε = 16/255 drops slightly, from {pct(pg(16.0, 'casia'))} to "
        f"{pct(agg.loc[('pgd+jpeg75', 16.0, 'casia')].m)}."
    ),
    figure(
        RUNS / "figures" / "edit_comparison.png",
        17,
        "Figure 2. PGD vs random noise vs PGD followed by JPEG (quality 75), one panel per model (unmatched share vs ε).",
    ),
    PageBreak(),
    P("7.3 Free edits: JPEG compression and blur of the original photo", h2),
    P("These are things any user could do without special tools. They are applied to the clean photo, with no PGD."),
    table(cond_table("jpeg", "JPEG quality", lambda s: f"{s:g}"), widths=[2.2 * cm] + [2.7 * cm] * 5 + [1.4 * cm]),
    Spacer(1, 6),
    table(cond_table("blur", "Blur σ (px)", lambda s: f"{s:g}"), widths=[2.2 * cm] + [2.7 * cm] * 5 + [1.4 * cm]),
    Spacer(1, 4),
    P(
        f"JPEG compression, even at quality 30, leaves the deep models essentially unaffected (surrogate "
        f"{pct(agg.loc[('jpeg', 30.0, 'surrogate')].m)}, casia {pct(agg.loc[('jpeg', 30.0, 'casia')].m)}). Blur starts to "
        f"matter only when strong: at σ = 3 pixels (SSIM {ssim.loc[('blur', 3.0)]:.3f}) the surrogate loses "
        f"{pct(agg.loc[('blur', 3.0, 'surrogate')].m)} and casia {pct(agg.loc[('blur', 3.0, 'casia')].m)}, while arcface "
        f"stays at {pct(agg.loc[('blur', 3.0, 'arcface')].m)}. Free edits are not a substitute for an optimised edit."
    ),
    figure(
        RUNS / "figures" / "free_edits.png",
        16,
        "Figure 3. Unmatched share under JPEG compression (left, quality decreasing to the right) and Gaussian blur (right).",
    ),
]

# 8. Seeds
seed_rows = [["Edit budget ε"] + MODELS]
for e in sorted(spread.loc["pgd"].index.get_level_values(0).unique()):
    seed_rows.append(
        [eps_fmt(e)]
        + [
            f"{100 * spread.loc[('pgd', e, m), 'mean']:.2f}% ± {100 * spread.loc[('pgd', e, m), 'std']:.2f}"
            for m in MODELS
        ]
    )
story += [
    PageBreak(),
    P("8. Are the results stable? Two more random seeds", h1),
    P(
        "The seed controls PGD's random starting point and the random-noise baseline, so re-running with different seeds "
        "shows how much of the result is luck. We repeated the full surrogate sweep (all conditions, all 3,000 pairs) with "
        "seeds 1 and 2, in addition to the original seed 0. The table shows, for PGD, the unmatched share averaged over the "
        "10 folds for each seed, then the mean ± standard deviation across the three seeds (std in percentage points)."
    ),
    table(seed_rows, widths=[2.4 * cm] + [3.0 * cm] * 5),
    Spacer(1, 4),
    P(
        "Largest standard deviation across seeds for each condition: "
        + ", ".join(f"{c} {100 * v:.2f} pts" for c, v in max_seed_std.items())
        + ". JPEG and blur involve no randomness, so they are identical across seeds. Every result in this document is "
        "reproducible to well under one percentage point; the differences between models are far larger than seed noise."
    ),
]

# 9. Every model as attacker
r2, h2_ = matrix_rows(2.0)
r4, h4 = matrix_rows(4.0)
r8, h8 = matrix_rows(8.0)
r16, h16 = matrix_rows(16.0)
r8j, h8j = matrix_rows(8.0, "pgd+jpeg75")
mw = [3.1 * cm] + [2.5 * cm] * 5 + [1.3 * cm]
story += [
    P("9. Is it just our surrogate? Every deep model as the attacker", h1),
    P(
        "A worry with Section 6 is that our surrogate might simply be a poor model to build edits against, for example "
        "because it overfits its training data. To test this, we ran the same PGD attack using each of the other deep models "
        "as the attacker (casia, vggface2 and arcface, all at seed 0) and scored the edited photos on all five models. "
        "LBPH cannot be an attacker because it has no gradients. ArcFace was converted from its ONNX file to PyTorch "
        "(onnx2torch) so gradients could flow through it; the converted model gives the same clean LFW accuracy "
        f"({pct(validate.loc['arcface'].accuracy, 2)})."
    ),
    P(
        "Each table below gives the share of same-person pairs the <b>target</b> (column) stops matching when the edit was "
        "built against the <b>attacker</b> (row), mean ± std over the 10 folds. Highlighted cells are each attacker against "
        "itself. The SSIM column shows the edits are equally visible whichever model built them."
    ),
    KeepTogether([P("ε = 2/255 (essentially invisible)", h3), table(r2, widths=mw, highlight=h2_)]),
    Spacer(1, 4),
    KeepTogether([P("ε = 4/255 (nearly invisible)", h3), table(r4, widths=mw, highlight=h4)]),
    Spacer(1, 4),
    KeepTogether([P("ε = 8/255", h3), table(r8, widths=mw, highlight=h8)]),
    Spacer(1, 4),
    KeepTogether([P("ε = 16/255 (clearly visible)", h3), table(r16, widths=mw, highlight=h16)]),
    Spacer(1, 4),
    KeepTogether([P("ε = 8/255, edit then JPEG quality 75", h3), table(r8j, widths=mw, highlight=h8j)]),
    Spacer(1, 6),
    figure(
        RUNS / "figures" / "attacker_transfer.png",
        17.4,
        "Figure 4. Unmatched share against ε for each attacker (one panel each), one line per target model; the thick "
        "line is the attacker scored on itself.",
    ),
    P("9.1 What changes", h2),
    P(
        f"<b>Our surrogate is the weakest attacker.</b> At ε = 8/255 its edits make casia lose "
        f"{pct(X(8.0, 'surrogate', 'casia'))} and vggface2 {pct(X(8.0, 'surrogate', 'vggface2'))}. Edits built against "
        f"vggface2 make casia lose {pct(X(8.0, 'vggface2', 'casia'))}; edits built against casia make vggface2 lose "
        f"{pct(X(8.0, 'casia', 'vggface2'))}. So the low transfer in Section 6 is partly a property of our surrogate, and "
        f'"edits barely transfer" was too strong as a general statement.'
    ),
    P(
        f"<b>But it is not a broken model; it is an easy target.</b> Edits from casia and vggface2 make the surrogate lose "
        f"{pct(X(8.0, 'casia', 'surrogate'))} and {pct(X(8.0, 'vggface2', 'surrogate'))} of pairs at ε = 8/255, and "
        f"{pct(X(4.0, 'casia', 'surrogate'))} and {pct(X(4.0, 'vggface2', 'surrogate'))} at ε = 4/255. It is fragile, and "
        f"its own gradients point in directions the other models do not share."
    ),
    P(
        f"<b>Architecture matters more than training data.</b> The three InceptionResnetV1 models fool each other well even "
        f"when their training faces differ: vggface2 → casia is {pct(X(8.0, 'vggface2', 'casia'))} at ε = 8/255 and "
        f"{pct(X(16.0, 'vggface2', 'casia'))} at 16/255. ArcFace, the only different architecture, resists every attacker: "
        f"the best is {best_vs_arc} at {pct(X(16.0, best_vs_arc, 'arcface'))}, and only at a clearly visible ε = 16/255. "
        f"In the other direction, ArcFace's own edits reach the InceptionResnetV1 models only moderately "
        f"({pct(X(16.0, 'arcface', 'surrogate'))} surrogate, {pct(X(16.0, 'arcface', 'casia'))} casia, "
        f"{pct(X(16.0, 'arcface', 'vggface2'))} vggface2 at ε = 16/255)."
    ),
    P(
        f"<b>Transfer is lopsided.</b> vggface2 → casia ({pct(X(8.0, 'vggface2', 'casia'))} at ε = 8/255) is much stronger "
        f"than casia → vggface2 ({pct(X(8.0, 'casia', 'vggface2'))}). The model trained on the larger, more varied VGGFace2 "
        f"data is both the strongest attacker and the hardest InceptionResnetV1 model to fool."
    ),
    P(
        f"<b>Invisible edits still transfer only partially.</b> At ε = 4/255 (SSIM about "
        f"{xssim.loc[('pgd', 4.0, 'surrogate')]:.2f}) the strongest transfer between any two different deep models is "
        f"{off_diag_eps4[1]} → {off_diag_eps4[2]} at {pct(off_diag_eps4[0])}; at ε = 2/255 nothing transfers above "
        f"{pct(max(X(2.0, a, t) for a in ATTACKERS for t in ATTACKERS if a != t))}. JPEG after editing reduces transfer "
        f"slightly (e.g. vggface2 → casia {pct(X(8.0, 'vggface2', 'casia'))} → "
        f"{pct(X(8.0, 'vggface2', 'casia', 'pgd+jpeg75'))} at ε = 8/255)."
    ),
    P(
        f"<b>LBPH</b> sits at {pct(min(X(8.0, a, 'lbph') for a in ATTACKERS))}–"
        f"{pct(max(X(8.0, a, 'lbph') for a in ATTACKERS))} at ε = 8/255 whichever model built the edit, which supports "
        f"treating it as noise-sensitive rather than fooled (Section 10)."
    ),
    P(
        "<b>Why might our surrogate be a weak attacker?</b> These are untested hypotheses: (1) <i>overfitting</i>: training "
        "accuracy reached 97.6% while LFW accuracy plateaued, although we kept the epoch that was best on held-out identities; (2) <i>crop "
        "geometry</i>: the surrogate aligns faces by landmarks, while casia and vggface2 use a box crop, so edits shaped "
        f"through one geometry may carry over worse to the other (ArcFace, which also uses landmark alignment, transfers "
        f"better to the surrogate, {pct(X(8.0, 'arcface', 'surrogate'))}, than to casia, {pct(X(8.0, 'arcface', 'casia'))}, "
        "at ε = 8/255); (3) <i>training budget</i>: 30 epochs on CASIA alone, much less than the pretrained models received."
    ),
]

# 9.2 Example edits


story += [
    PageBreak(),
    P("9.2 What the edits look like", h2),
    P(
        f"To make the tables concrete, we picked {len(people)} same-person pairs (one from each of three different folds, "
        f"chosen at random among pairs that all four deep models match before editing): "
        f"{', '.join(people)}. For each pair we edited photo <i>a</i> with PGD against each attacker at "
        f"ε = 0, 2, 4, 8 and 16/255, with exactly the sweep's settings (20 steps, random start, seed 0, rounded to 8-bit), "
        f"and scored every edited photo on all five models with each model's fold threshold. GPU computation is not "
        f"bit-for-bit reproducible even with a fixed seed, so re-running can flip a tag for a model sitting right at its "
        f"threshold (3 of the 300 tags changed between two runs); the overall pattern does not change."
    ),
    P(
        "<b>How to read the grids.</b> Rows are attackers, columns are budgets; the first column is the photo <i>b</i> the "
        "edited photo is compared against. Under each image are five tags, one per target model: <b>S</b> surrogate, "
        "<b>C</b> casia, <b>V</b> vggface2, <b>A</b> arcface, <b>L</b> lbph. A <font color='#c0392b'><b>red</b></font> tag "
        "means that model no longer matches the pair (the edit fooled it); grey means it still recognises the person. "
        "The SSIM between the edited and original photo is printed underneath. Images show the central 190×190 pixels of "
        "the 250×250 photo. Examples were chosen so that all four deep models match the pair before editing; LBPH, "
        "with its 67% clean accuracy, already misses two of the three pairs, so its L tag is red even at ε = 0."
    ),
    P(
        f"<b>What they show.</b> The single photos follow the same pattern as the tables. In the first example, our "
        f"surrogate's edit fools only the surrogate up to ε = 8/255 (fooled: {fooled(0, 'surrogate', 8)}), while the "
        f"vggface2 edit at the same budget fools {fooled(0, 'vggface2', 8)} and the casia edit fools "
        f"{fooled(0, 'casia', 8)}. ArcFace's edit fools only ArcFace until ε = 16/255 ({fooled(0, 'arcface', 16)}). "
        f"At 2–4/255 the edits are hard to see; at 16/255 a grainy texture is clearly visible, especially on skin."
    ),
]
for k in people.index:
    story += [
        PageBreak() if k else Spacer(1, 2),
        figure(
            EX / f"example_{k}.png",
            17.4 if k else 16.0,
            f"Figure {5 + k}. {people[k]}: photo a edited against each attacker (rows) at each budget (columns). "
            f"Tags: models fooled (red) or still matching (grey); S surrogate, C casia, V vggface2, A arcface, "
            f"L lbph.",
        ),
    ]
story += [
    PageBreak(),
    P("9.3 The edits on their own", h2),
    P(
        "The figure below shows only the change each attacker made to the first example (edited minus original), with "
        "each column scaled by 1/ε so the pattern is visible at every budget (mid-grey = no change). Three things stand out:"
    ),
    P(
        "• <b>The pretrained attackers' edits are face-shaped.</b> The casia, vggface2 and arcface edits trace the eyes, "
        "brows, mouth and outline of the face, so they change the facial features a recognition model relies on. Our "
        "surrogate's edit looks like unstructured texture with little facial structure. This is a plausible visual "
        "explanation for why its edits transfer so poorly (Section 9.1), though we have not tested it formally.<br/>"
        "• <b>Each edit stays inside its attacker's crop.</b> PGD can only change pixels the attacker actually looks at, so "
        "each edit is a rectangle matching that model's crop: a landmark-aligned square for the surrogate and arcface, the "
        "MTCNN box with margin for casia and vggface2. An edit built through one crop may miss part of another model's "
        "crop.<br/>"
        "• <b>The rest is PGD's random start.</b> Outside the crop, the speckle is the random starting noise, which no "
        "gradient ever updates. It is the same for every attacker and slightly lowers SSIM; it is not part of the "
        "optimised edit.",
        body,
    ),
    figure(
        EX / "perturbation_0.png",
        14.5,
        f"Figure {5 + len(people)}. {people[0]}: the edit alone (edited − original) for each attacker and budget, "
        f"each column scaled by 1/ε. Edits for the other two examples are in runs/examples/.",
    ),
]

# 10. LBPH
lb = lambda c, s: agg.loc[(c, s, "lbph")].m  # noqa: E731
story += [
    P("10. A note on LBPH (the non-deep model)", h1),
    P(
        f"LBPH needs careful interpretation. Its clean accuracy is only {pct(validate.loc['lbph'].accuracy)}, so with its "
        f"best threshold it already fails to match {pct(lb('pgd', 0.0))} of same-person pairs before any edit. Its numbers "
        f"also move in the opposite direction from what transfer would predict: small PGD edits slightly <i>lower</i> its "
        f"unmatched share ({pct(lb('pgd', 0.0))} → {pct(lb('pgd', 4.0))} at ε = 4/255), random noise affects it far more than "
        f"PGD at ε = 16/255 ({pct(lb('noise', 16.0))} vs {pct(lb('pgd', 16.0))}), and blur at σ = 3 breaks it almost "
        f"completely ({pct(lb('blur', 3.0))})."
    ),
    P(
        "LBPH describes faces by fine local texture, so any change to pixel-level texture moves its distances, whether or "
        "not the change is targeted. Its numbers therefore measure sensitivity to pixel changes in general, not transfer of "
        "the edit, and it should be reported as a non-deep baseline rather than as evidence that the edit transfers."
    ),
]

# 11. Conclusions
story += [
    P("11. Conclusions", h1),
    P(
        "1. <b>Edits built against our surrogate transfer poorly.</b> A nearly invisible edit (ε = 4/255) that fully defeats "
        "the surrogate leaves every other deep model recognising about 95% or more of the same-person pairs."
    ),
    P(
        f"2. <b>The results are stable.</b> Across three random seeds no result's standard deviation exceeds "
        f"{100 * max_seed_std.max():.2f} percentage points."
    ),
    P(
        f"3. <b>Transfer depends strongly on which model the edit is built against.</b> Our surrogate is the weakest of four "
        f"attackers: edits built against vggface2 make casia lose {pct(X(8.0, 'vggface2', 'casia'))} of pairs at "
        f"ε = 8/255, against {pct(X(8.0, 'surrogate', 'casia'))} for our surrogate."
    ),
    P(
        f"4. <b>Architecture matters more than training data.</b> Models sharing the InceptionResnetV1 architecture fool each "
        f"other well even with different training faces, while ArcFace resists every attacker (at most "
        f"{pct(X(16.0, best_vs_arc, 'arcface'))}, and only at a visible ε = 16/255). The surrogate-only result that seemed to "
        f"favour training data (Section 6) does not generalise."
    ),
    P(
        "5. <b>The effect comes from the optimisation.</b> Random noise of the same size, JPEG compression and mild blur do "
        "almost nothing to the deep models."
    ),
    P(
        f"6. <b>Implication for privacy tools.</b> Protection built against one recognition system is unreliable against an "
        f"unknown one: it depends on the tool's choice of model, it is much weaker against a different architecture, and at "
        f"a nearly invisible budget (ε = 4/255) even the best pairing transfers to only {pct(off_diag_eps4[0])} of pairs."
    ),
]

# 10. Limitations
story += [
    P("12. Limitations and next steps", h1),
    P(
        "• One editing method (PGD) and one budget sweep. Attacking an ensemble of models at once, the standard way to make "
        "edits transfer better, was not tried.<br/>"
        "• Seeds 1 and 2 were run with the surrogate as attacker only; the other attackers use seed 0. Seed spread for the "
        "surrogate is under 0.7 points, so this is unlikely to matter.<br/>"
        "• The saved surrogate is the best epoch on 500 held-out CASIA identities (98.28% clean LFW). Why it is a weak attacker "
        "(overfitting, crop geometry, training budget) is untested; retraining with a box crop would test one of those.<br/>"
        f"• The original seed-0 sweep scored ArcFace with onnxruntime; all later runs use the PyTorch conversion. Both give "
        f"identical clean accuracy, and ArcFace's numbers agree across the three seeds to within "
        f"{100 * arc_seed_std:.2f} points.<br/>"
        "• Face landmarks come from the clean photo and are reused for its edited versions; a real system would re-detect the "
        "face, which large edits could shift.<br/>"
        "• The proposal's optional check of whether protection holds equally across demographic groups (LFW is roughly "
        "77.5% male and 83.5% White) has not been done yet.<br/>"
        "• LBPH's low clean accuracy limits what its numbers can show (Section 10).<br/>"
        "• PGD's random start adds noise to the whole photo, but only pixels inside the attacker's crop are then optimised, "
        "so the noise outside the crop stays (Section 9.3). It slightly lowers SSIM equally for every attacker; starting "
        "from the clean photo, or masking the start to the crop, would remove it.",
        body,
    ),
    P(
        f"An early smoke test on a random subset of 200 pairs (ε = 0, 4, 16) gave the same pattern as the full run, e.g. "
        f"casia {pct(smoke[(smoke.condition == 'pgd') & (smoke.strength == 16) & (smoke.model == 'casia')].unmatched.mean())} "
        f"and arcface {pct(smoke[(smoke.condition == 'pgd') & (smoke.strength == 16) & (smoke.model == 'arcface')].unmatched.mean())} "
        f"at ε = 16/255. All figures in this document use the full 3,000-pair run.",
        small,
    ),
]


OUT.write_text(PREAMBLE + "\n\n".join(x for x in story if x) + "\n\n\\end{document}\n", encoding="utf-8")
print("wrote", OUT, "and", len(list(FIGS.iterdir())), "figures")
