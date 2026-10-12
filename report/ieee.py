"""Generate the IEEE-format paper (report/ieee/paper.tex) from the results in runs/.

    uv run python report/ieee.py
    cd report/ieee && latexmk -pdf paper.tex

Numbers in the text are written as @key@ placeholders and filled from `results.py`, so the paper stays in
sync with the experiments. Edit the template here, not paper.tex.
"""

import re
import shutil

from PIL import Image
from results import (
    ATTACKERS,
    MODELS,
    ROOT,
    RUNS,
    X,
    agg,
    best_vs_arc,
    epochs,
    header,
    max_seed_std,
    off_diag_eps4,
    people,
    pg,
    ssim,
    validate,
    xssim,
)

OUT = ROOT / "report" / "ieee"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def p(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def eps(e):
    return f"{e:g}/255"


# ---------------------------------------------------------------- values used in the text
best = max(e[3] for e in epochs)
gaps = [abs(validate.loc[m].accuracy - validate.loc[m].published) for m in ("vggface2", "casia", "arcface")]
ssim_range = max(xssim.loc[("pgd", e)].max() - xssim.loc[("pgd", e)].min() for e in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0))
V = {
    "npairs": "3,000",
    "nimages": f"{int(header[0]):,}",
    "nids": f"{int(header[1]):,}",
    "final": p(epochs[-1][3], 2),
    "best": p(best, 2),
    "bestep": str([int(e[0]) + 1 for e in epochs if e[3] == best][0]),
    "trainacc": p(epochs[-1][2]),
    "maxgap": f"{100 * max(gaps):.2f}",
    "maxstd": f"{100 * max_seed_std.max():.2f}",
    "ssimrange": f"{ssim_range:.3f}",
    "ratio16": f"{pg(16.0, 'casia') / pg(16.0, 'vggface2'):.1f}",
    "pg4maxother": p(max(pg(4.0, m) for m in ("casia", "vggface2", "arcface"))),
    "nssim16": f"{ssim.loc[('noise', 16.0)]:.3f}",
    "bssim3": f"{ssim.loc[('blur', 3.0)]:.3f}",
    "bestarc": best_vs_arc,
    "arcmax": p(X(16.0, best_vs_arc, "arcface")),
    "od4": p(off_diag_eps4[0]),
    "od4a": off_diag_eps4[1],
    "od4t": off_diag_eps4[2],
    "max2": p(max(X(2.0, a, t) for a in ATTACKERS for t in ATTACKERS if a != t)),
    "person0": people[0],
    "lbacc": p(validate.loc["lbph"].accuracy),
}
for e in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
    V[f"ssim{e:g}"] = f"{ssim.loc[('pgd', e)]:.3f}"
    for m, k in zip(MODELS, "scval", strict=True):
        V[f"pg{e:g}{k}"] = p(pg(e, m))
short = dict(zip(ATTACKERS, "scva", strict=True))
for e in (4.0, 8.0, 16.0):
    for a in ATTACKERS:
        for t in ATTACKERS:
            V[f"X{e:g}{short[a]}{short[t]}"] = p(X(e, a, t))
for cond, key, e, m in [
    ("noise", "n16s", 16.0, "surrogate"),
    ("noise", "n16c", 16.0, "casia"),
    ("pgd+jpeg75", "j2s", 2.0, "surrogate"),
    ("pgd+jpeg75", "j4s", 4.0, "surrogate"),
    ("jpeg", "q30s", 30.0, "surrogate"),
    ("blur", "b3s", 3.0, "surrogate"),
    ("blur", "b3c", 3.0, "casia"),
    ("noise", "lbn16", 16.0, "lbph"),
    ("blur", "lbb3", 3.0, "lbph"),
]:
    V[key] = p(agg.loc[(cond, e, m)].m)


# ---------------------------------------------------------------- tables
def tabular(head, rows, spec):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


T_MODELS = tabular(
    ["Model", "Architecture", "Training data", "Loss", "Input"],
    [
        ["surrogate", "InceptionResnetV1", "CASIA-WebFace (ours)", "softmax", r"5-pt, 160$^2$"],
        ["casia", "InceptionResnetV1", "CASIA-WebFace", "softmax", r"box, 160$^2$"],
        ["vggface2", "InceptionResnetV1", "VGGFace2", "softmax", r"box, 160$^2$"],
        ["arcface", "ResNet-50", "WebFace600K", "ArcFace", r"5-pt, 112$^2$"],
        ["lbph", "LBP histograms", "none", "none", r"5-pt, 112$^2$"],
    ],
    "lllll",
)

T_VALID = tabular(
    ["Model", "Ours", "Published", "Gap (pts)"],
    [
        [
            m,
            p(validate.loc[m].accuracy, 2) + r" $\pm$ " + p(validate.loc[m]["std"], 2),
            p(validate.loc[m].published, 2) if validate.loc[m].published == validate.loc[m].published else "--",
            f"{100 * (validate.loc[m].accuracy - validate.loc[m].published):+.2f}"
            if validate.loc[m].published == validate.loc[m].published
            else "--",
        ]
        for m in MODELS
    ],
    "lrrr",
)

T_MAIN = tabular(
    [r"$\varepsilon$"] + MODELS + ["SSIM"],
    [
        [eps(e)]
        + [p(agg.loc[("pgd", e, m)].m) + r" $\pm$ " + p(agg.loc[("pgd", e, m)].s) for m in MODELS]
        + [f"{ssim.loc[('pgd', e)]:.3f}"]
        for e in sorted(agg.loc["pgd"].index.get_level_values(0).unique())
    ],
    "l" + "r" * 6,
)

BASELINES = [
    ("Noise", "noise", 8.0, rf"$\varepsilon$ = {eps(8)}"),
    ("Noise", "noise", 16.0, rf"$\varepsilon$ = {eps(16)}"),
    ("PGD + JPEG 75", "pgd+jpeg75", 2.0, rf"$\varepsilon$ = {eps(2)}"),
    ("PGD + JPEG 75", "pgd+jpeg75", 4.0, rf"$\varepsilon$ = {eps(4)}"),
    ("PGD + JPEG 75", "pgd+jpeg75", 16.0, rf"$\varepsilon$ = {eps(16)}"),
    ("JPEG", "jpeg", 50.0, "q = 50"),
    ("JPEG", "jpeg", 30.0, "q = 30"),
    ("Blur", "blur", 2.0, r"$\sigma$ = 2"),
    ("Blur", "blur", 3.0, r"$\sigma$ = 3"),
]
T_BASE = tabular(
    ["Edit", "Strength", "S", "C", "V", "A", "L", "SSIM"],
    [
        [name, strength] + [p(agg.loc[(c, e, m)].m) for m in MODELS] + [f"{ssim.loc[(c, e)]:.3f}"]
        for name, c, e, strength in BASELINES
    ],
    "llrrrrrr",
)

matrix_rows = []
for e in (4.0, 8.0, 16.0):
    matrix_rows.append([rf"\multicolumn{{7}}{{l}}{{\emph{{$\varepsilon$ = {eps(e)}}}}}"])
    for a in ATTACKERS:
        cells = [(r"\textbf{" + p(X(e, a, t)) + "}") if t == a else p(X(e, a, t)) for t in MODELS]
        matrix_rows.append([a] + cells + [f"{xssim.loc[('pgd', e, a)]:.3f}"])
T_MATRIX = tabular([r"Attacker $\downarrow$ / target $\rightarrow$"] + MODELS + ["SSIM"], matrix_rows, "l" + "r" * 6)


# ---------------------------------------------------------------- figures
def fig(name):
    src = RUNS / ("examples" if name.startswith(("example", "perturbation")) else "figures") / f"{name}.png"
    if src.stat().st_size > 1_000_000:  # noisy photo grids: JPEG is ~10x smaller than PNG
        Image.open(src).convert("RGB").save(FIGS / f"{name}.jpg", quality=92)
        return f"figures/{name}.jpg"
    shutil.copy(src, FIGS / src.name)
    return f"figures/{src.name}"


F = {k: fig(k) for k in ("main_result", "edit_comparison", "attacker_transfer", "example_0", "perturbation_0")}

# ---------------------------------------------------------------- paper
TEMPLATE = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,cite,url}
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{Do Edits That Fool One Face Recognition Model Fool Others? Measuring the Transfer of PGD Perturbations on LFW}

\author{\IEEEauthorblockN{Dylan Davies (2672598), Sudhir Adjodhya (2652330) and Michael Badenhorst (2652150)}
\IEEEauthorblockA{COMS4036A Computer Vision}}

\maketitle

\begin{abstract}
Photo-cloaking tools assume that an edit built against one face recognition model will also fool models its
author never saw. We test that assumption on the @npairs@ same-person pairs of LFW View~2. We trained an
InceptionResnetV1 surrogate from scratch on CASIA-WebFace, edited one photo in each pair with $L_\infty$ projected
gradient descent (PGD) against it, and scored the edits on four models that played no part in building them: two
pretrained InceptionResnetV1 networks, an ArcFace ResNet-50 and an LBPH matcher. At $\varepsilon = 4/255$ (SSIM
@ssim4@) the edit stops the surrogate matching @pg4s@ of the pairs, but no other deep model loses more than
@pg4maxother@. Repeating the attack with each pretrained network as the attacker shows that our surrogate is the
weakest source of transferable edits: at $\varepsilon = 8/255$, edits built against the VGGFace2 model break
@X8vc@ of the CASIA model's pairs, against @X8sc@ for our surrogate. Edits transfer well between networks that
share an architecture, even when their training data differ, while ArcFace resists every attacker (at most
@arcmax@, at a clearly visible $\varepsilon = 16/255$). Random noise, JPEG compression and blur barely affect the
deep models. Across three random seeds, no result has a standard deviation above @maxstd@ percentage points.
\end{abstract}

\begin{IEEEkeywords}
adversarial examples, transferability, face recognition, privacy, projected gradient descent
\end{IEEEkeywords}

\section{Introduction}
Fawkes~\cite{fawkes} and LowKey~\cite{lowkey} protect people from face recognition by adding small perturbations to
photos before they are posted. A photo is edited once, against whatever model the tool's designers had, and must
then defeat every system that sees it later. Radiya-Dixit \emph{et al.}~\cite{radiya} point out how weak that
position is: the defender commits to an edit, and the recogniser can be replaced or retrained afterwards.

Both tools change the photos a recogniser later learns from or searches. We ask a simpler question about a
recogniser that is already trained. If a photo is edited
until one model no longer recognises its owner, do models the editor never had access to also stop recognising
them? And where some models resist while others give way, does the difference follow their architecture or the
faces they were trained on?

We answer with a controlled transfer experiment on LFW~\cite{lfw}. The four test models were chosen so that each
differs from our surrogate in a known way. The \emph{casia} model matches its architecture, objective and training
data, so only the random initialisation differs; \emph{vggface2} keeps the architecture but saw different faces;
ArcFace changes all of these. A hand-crafted LBPH matcher~\cite{lbph} adds a baseline without deep learning. Our
contributions are:
\begin{itemize}
\item a measurement of how often PGD edits built against a from-scratch surrogate stop four unseen models from
matching same-person pairs, over seven edit budgets, with random-noise, JPEG and blur baselines;
\item an attacker--target matrix in which every differentiable model takes a turn as the attacker, showing that
shared architecture predicts transfer better than shared training data;
\item a pipeline in which every model crops the same edited photo its own way, with each model's clean accuracy
checked against its published figure before any attack.
\end{itemize}

\section{Related Work}
Small, targeted input changes can flip the output of a deep network~\cite{szegedy2014}, and gradient-sign
methods~\cite{fgsm} and PGD~\cite{madry} find them efficiently. Such perturbations often transfer between models
trained for the same task~\cite{papernot}, which makes black-box attacks practical. Liu \emph{et
al.}~\cite{liu2017} found that targeted perturbations rarely transfer from one model to another, and that
optimising against an ensemble of models makes them transfer much more often.

Face recognition models map an aligned face crop to an embedding and compare embeddings by distance.
FaceNet~\cite{facenet} trained such embeddings with a triplet loss; ArcFace~\cite{arcface} adds an angular margin to
a softmax loss. Widely used pretrained models include InceptionResnetV1~\cite{inceptionv4} networks trained on
VGGFace2~\cite{vggface2} and CASIA-WebFace~\cite{casia}, distributed with facenet-pytorch~\cite{facenetpytorch},
and InsightFace's ArcFace ResNet-50 trained on WebFace600K, a subset of WebFace260M~\cite{webface260m}. LFW remains
the standard verification benchmark, although it is too small to train on~\cite{deepid2} and its subjects are
mostly male and White~\cite{hanjain}.

Fawkes and LowKey use adversarial perturbations as a privacy defence, and both depend on transfer from the model
they optimise against to the one a third party runs. Radiya-Dixit \emph{et al.} show that recognisers trained
later can learn to ignore such perturbations. We isolate the inference-time question and vary architecture and
training data separately.

\section{Method}
\subsection{Threat model}
A user edits one photo of themselves before posting it. They do not know which other photo it will later be
compared with, so the attack pushes the edited photo away from the embedding of its own original. The edit
succeeds against a recogniser if that recogniser no longer matches the edited photo with a different, unedited
photo of the same person.

\subsection{Models}
Table~\ref{tab:models} lists the five models. The surrogate is an InceptionResnetV1 that we trained from scratch
with a softmax cross-entropy loss on CASIA-WebFace, the same architecture and objective as the two facenet-pytorch
models. The \emph{casia} model therefore differs from it only in initialisation, \emph{vggface2} in training data,
and \emph{arcface} in architecture, objective and data. LBPH compares an $8\times8$ grid of uniform local binary
pattern histograms with the chi-square distance.

\begin{table}[t]
\centering
\caption{Models. ``5-pt'' is five-landmark alignment; ``box'' is an MTCNN box crop with a 32-pixel margin.}
\label{tab:models}
\scriptsize
\setlength{\tabcolsep}{3pt}
@T_MODELS@
\end{table}

\subsection{Preprocessing}
Each model expects faces cropped and aligned its own way. A wrong crop raises no error; it only lowers accuracy,
which is easy to miss. We run MTCNN~\cite{mtcnn} once on every LFW photo (it found a face in all 7,701) and express each model's crop
as a $2\times3$ affine map from the $250\times250$ photo to the model input: a box crop at $160\times160$ for the
facenet models, the standard InsightFace five-point alignment at $112\times112$ for ArcFace and LBPH, and a
five-point alignment at $160\times160$ for the surrogate, using the landmark positions MTCNN finds on its CASIA
training crops. The crops are applied with a differentiable bilinear warp, so the surrogate's own cropping sits
inside the attack. Every edit is made on the full photo, and every model crops the same edited photo.

\subsection{Verification protocol}
A pair is accepted as the same person when the distance between embeddings (cosine for the deep models,
chi-square for LBPH) is below a threshold. Following the LFW protocol, each model's threshold for a fold is fitted
on the clean pairs of the other nine folds. We keep these clean thresholds fixed when scoring edited photos, so
$\varepsilon = 0$ gives each model's ordinary error rate.

\subsection{Attack}
For a photo $x$ with surrogate embedding $f(x)$, PGD~\cite{madry} starts from $x$ plus uniform noise in
$[-\varepsilon, \varepsilon]$ and repeats
\begin{equation}
x^{t+1} = \Pi_{\varepsilon}\!\left(x^{t} + \alpha \,\mathrm{sign}\, \nabla_{x}\, d\big(f(x^{t}), f(x)\big)\right),
\end{equation}
where $d$ is the cosine distance and $\Pi_\varepsilon$ projects onto the $L_\infty$ ball of radius $\varepsilon$
around $x$ and onto valid pixel values. We use 20 steps with $\alpha = 2.5\varepsilon/20$, and round the result to
8 bits before scoring, as a posted photo would be. We sweep $\varepsilon \in \{0, 0.5, 1, 2, 4, 8, 16\}/255$.

\subsection{Baselines and metrics}
Random sign noise of size $\varepsilon$ on every pixel tests whether the optimisation matters. Saving the edited
photo as JPEG at quality 75 tests whether ordinary recompression removes the edit. JPEG compression (quality 90 to
30) and Gaussian blur ($\sigma$ = 0.5 to 3 pixels) of the clean photo are edits any user could make for free.

Our measure is the \emph{unmatched share}: the fraction of the @npairs@ same-person pairs that a model no longer
accepts after photo $a$ of each pair is edited, averaged over the ten folds. SSIM~\cite{ssim} between the edited
and original photo measures how visible the edit is.

\section{Experimental Setup}
LFW View~2 holds 6,000 fixed pairs in ten folds. We use them unchanged, including the six pairs its maintainers
list as mislabelled. The surrogate was trained for 30 epochs on CASIA-WebFace (@nimages@ aligned images of
@nids@ people) with SGD (Nesterov momentum 0.9, weight decay $5\times10^{-4}$), a one-cycle schedule peaking at a
learning rate of 0.1, batch size 256, bf16 mixed precision and random horizontal flips. Training took about
1.5~hours on one RTX~5070~Ti. The final epoch reached @final@ clean LFW accuracy (best @best@, at epoch
@bestep@), against 99.05\% published for the pretrained \emph{casia} model trained on the same data.

Because a misaligned crop lowers accuracy without any error, we checked every model against its published LFW
accuracy before attacking (Table~\ref{tab:valid}). We chose the facenet crop margin from $\{0, 14, 32, 44\}$
pixels by clean accuracy; 32 was best for both models. The three pretrained deep models land within @maxgap@
points of their published figures. For the cross-attacker experiments we converted ArcFace from ONNX to PyTorch
with onnx2torch so that gradients flow through it; the converted model reproduces the original's accuracy.

\begin{table}[t]
\centering
\caption{Clean LFW View 2 accuracy (mean $\pm$ std over 10 folds) against published figures.}
\label{tab:valid}
\footnotesize
@T_VALID@
\end{table}

\section{Results}
\subsection{Transfer from the surrogate}
Table~\ref{tab:main} and Fig.~\ref{fig:main} give the main result. The surrogate stops matching quickly: from
@pg0s@ of pairs with no edit to @pg2s@ at $\varepsilon = 2/255$, where the edit is essentially invisible (SSIM
@ssim2@), and @pg4s@ from $4/255$ up. The other deep models hardly move at these budgets. At $4/255$ \emph{casia}
goes from @pg0c@ to @pg4c@, \emph{vggface2} from @pg0v@ to @pg4v@ and ArcFace from @pg0a@ to @pg4a@. Substantial
transfer appears only at $16/255$, where SSIM has fallen to @ssim16@ and the edit is plain to see. There
\emph{casia} loses @pg16c@ of pairs, \emph{vggface2} @pg16v@ and ArcFace @pg16a@.

With our surrogate as the attacker, shared training data looks decisive: \emph{casia}, trained on the same faces,
loses @ratio16@ times as many pairs at $16/255$ as \emph{vggface2}. Section~\ref{sec:cross} shows that this reading
does not survive a change of attacker.

\begin{table*}[t]
\centering
\caption{Unmatched share of same-person pairs (mean $\pm$ std over folds) after PGD against the surrogate.
$\varepsilon = 0$ is each model's baseline error. SSIM is between edited and original photo.}
\label{tab:main}
\footnotesize
@T_MAIN@
\end{table*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@F_main_result@}
\caption{Unmatched share against edit budget for PGD against the surrogate, one line per model (error bars: std
over folds). Dashed: SSIM (right axis).}
\label{fig:main}
\end{figure}

\subsection{Baselines}
Random noise does little even though it is more visible than PGD at the same budget (Table~\ref{tab:base},
Fig.~\ref{fig:compare}). At $\varepsilon = 16/255$ (SSIM @nssim16@) it moves the surrogate to only @n16s@ and
\emph{casia} to @n16c@. JPEG at quality 75 after editing weakens small edits, cutting the surrogate's unmatched
share at $2/255$ from @pg2s@ to @j2s@, but edits of $4/255$ and above survive it (@j4s@ at $4/255$). Free edits
barely register: JPEG at quality 30 leaves the surrogate at @q30s@, and only strong blur ($\sigma = 3$ pixels, SSIM
@bssim3@) has an effect, @b3s@ for the surrogate and @b3c@ for \emph{casia}.

\begin{table}[t]
\centering
\caption{Unmatched share under baseline edits (S surrogate, C casia, V vggface2, A arcface, L lbph).}
\label{tab:base}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
@T_BASE@
\end{table}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{@F_edit_comparison@}
\caption{PGD against the surrogate compared with random noise of the same budget and with PGD followed by JPEG at
quality 75, one panel per scored model.}
\label{fig:compare}
\end{figure*}

\subsection{Stability across seeds}
The random seed sets PGD's starting point and the noise baseline. We repeated the full surrogate sweep with two
more seeds. Across the three seeds the standard deviation of every averaged result stayed at or below @maxstd@
percentage points, far smaller than the differences between models.

\subsection{Every deep model as the attacker}\label{sec:cross}
Low transfer could be a property of our surrogate rather than of transfer in general. To test this we repeated the
attack with \emph{casia}, \emph{vggface2} and ArcFace as the attacker and scored all five models
(Table~\ref{tab:matrix}, Fig.~\ref{fig:matrix}). Edits are equally visible whichever model builds them: at every
budget their SSIM differs by at most @ssimrange@.

Our surrogate is the weakest attacker. At $\varepsilon = 8/255$ its edits break @X8sc@ of \emph{casia}'s pairs,
while edits built against \emph{vggface2} break @X8vc@. It is also the easiest model to fool: edits from
\emph{casia} and \emph{vggface2} break @X8cs@ and @X8vs@ of its pairs at the same budget.

Architecture predicts transfer better than training data. The three InceptionResnetV1 models fool one another
even when trained on different faces: \emph{vggface2}~$\to$~\emph{casia} reaches @X8vc@ at $8/255$ and @X16vc@ at
$16/255$. ArcFace resists every attacker; the strongest, \emph{@bestarc@}, reaches @arcmax@ and only at $16/255$.
ArcFace's own edits transfer moderately to the others (@X16as@ surrogate, @X16ac@ \emph{casia}, @X16av@
\emph{vggface2} at $16/255$).

Transfer is also lopsided. At $8/255$, \emph{vggface2}~$\to$~\emph{casia} (@X8vc@) is much stronger than
\emph{casia}~$\to$~\emph{vggface2} (@X8cv@). At the nearly invisible $4/255$, the strongest transfer between two
different deep models is \emph{@od4a@}~$\to$~\emph{@od4t@} at @od4@, and at $2/255$ none exceeds @max2@.

\begin{table}[t]
\centering
\caption{Unmatched share when the edit is built against the attacker (row) and scored on the target (column).
Bold: attacker scored on itself.}
\label{tab:matrix}
\scriptsize
\setlength{\tabcolsep}{2pt}
@T_MATRIX@
\end{table}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{@F_attacker_transfer@}
\caption{Unmatched share against $\varepsilon$ for each attacker (one panel each), one line per target. The thick
line is the attacker scored on itself.}
\label{fig:matrix}
\end{figure*}

\subsection{What the edits look like}
Fig.~\ref{fig:example} shows one LFW pair (@person0@) edited by each attacker. At 2--4/255 the edits are hard to
see; at $16/255$ they add a visible grain, mostly on skin. Fig.~\ref{fig:pert} shows the change alone. The
pretrained attackers' edits follow the eyes, brows, mouth and outline of the face, while the surrogate's looks like
texture with little facial structure. Each edit stays inside its attacker's crop; outside it, only PGD's random
start remains.

\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{@F_example_0@}
\caption{Photo $a$ of one pair edited against each attacker (rows) at each budget (columns), with the comparison
photo $b$ on the left. Tags under each photo: models fooled (red) or still matching (grey); S surrogate, C casia,
V vggface2, A arcface, L lbph.}
\label{fig:example}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@F_perturbation_0@}
\caption{The edit alone (edited minus original) for the pair in Fig.~\ref{fig:example}, each column scaled by
$1/\varepsilon$.}
\label{fig:pert}
\end{figure}

\section{Discussion}
Our first experiment, with only our surrogate as the attacker, suggested that edits transfer mainly to models that
share training data. The cross-attacker results overturn that reading. Edits transfer readily among the three
InceptionResnetV1 models whatever their training data, and ArcFace, the one model with a different architecture,
is barely touched by any of them. On the proposal's question of architecture against data, the evidence favours
architecture.

We did not test why our surrogate is a weak attacker, but three explanations fit what we saw. It may overfit:
training accuracy reached @trainacc@ while LFW accuracy levelled off, and we kept the final epoch rather than the
best. Its crop geometry differs from the facenet models', which use a box crop; ArcFace, which shares the
landmark-based alignment, transfers better to the surrogate (@X8as@) than to \emph{casia} (@X8ac@) at
$8/255$. It also had far less training than the pretrained models.

LBPH needs care. With a clean accuracy of @lbacc@ it already misses @pg0l@ of same-person pairs before any edit.
Random noise at $16/255$ breaks @lbn16@ of its pairs against @pg16l@ for PGD, and blur at $\sigma = 3$ breaks
@lbb3@. It reacts to any change in fine texture, so we treat it as a baseline rather than as evidence of transfer.

This is discouraging for privacy tools. Whether an edit protects someone depends on which model it was built
against, and the protection is much weaker against a model with a different architecture. At a nearly invisible
budget, even the best-matched pair of models transfers to only @od4@ of pairs.

\section{Limitations}
We used one attack and did not try ensemble attacks~\cite{liu2017}, which usually transfer better. Only the
surrogate attacker was repeated over three seeds. Landmarks are detected on the clean photo and reused for its
edited versions, whereas a deployed system would detect the face again. PGD on the GPU is not bit-for-bit
reproducible even with a fixed seed; re-running the example edits changed 3 of 300 fooled-or-matched decisions,
all for models sitting at their threshold. PGD's random start leaves noise outside the attacker's crop, which
lowers SSIM slightly and equally for every attacker. Finally, we did not check whether protection is equally
effective across demographic groups, which matters given LFW's skew~\cite{hanjain}.

\section{Conclusion}
PGD edits built against a from-scratch face recognition model transfer poorly to models its author never used.
Changing the attacker shows that this is partly a weakness of our surrogate: edits built against a pretrained
InceptionResnetV1 transfer strongly to other networks with that architecture, even across training sets. A
different architecture resists edits from every attacker we tried until the edit is clearly visible. A privacy
edit made once against one model should not be expected to protect against an unknown recogniser.

\begin{thebibliography}{99}
\bibitem{fawkes} S.~Shan, E.~Wenger, J.~Zhang, H.~Li, H.~Zheng, and B.~Y. Zhao, ``Fawkes: Protecting privacy
against unauthorized deep learning models,'' in \emph{Proc. 29th USENIX Security Symp.}, 2020, pp. 1589--1604.
\bibitem{lowkey} V.~Cherepanova, M.~Goldblum, H.~Foley, S.~Duan, J.~Dickerson, G.~Taylor, and T.~Goldstein,
``LowKey: Leveraging adversarial attacks to protect social media users from facial recognition,'' in \emph{Proc.
Int. Conf. Learn. Represent. (ICLR)}, 2021.
\bibitem{radiya} E.~Radiya-Dixit, S.~Hong, N.~Carlini, and F.~Tram\`er, ``Data poisoning won't save you from facial
recognition,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2022.
\bibitem{szegedy2014} C.~Szegedy, W.~Zaremba, I.~Sutskever, J.~Bruna, D.~Erhan, I.~Goodfellow, and R.~Fergus,
``Intriguing properties of neural networks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2014.
\bibitem{fgsm} I.~J. Goodfellow, J.~Shlens, and C.~Szegedy, ``Explaining and harnessing adversarial examples,'' in
\emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2015.
\bibitem{madry} A.~Madry, A.~Makelov, L.~Schmidt, D.~Tsipras, and A.~Vladu, ``Towards deep learning models resistant
to adversarial attacks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2018.
\bibitem{papernot} N.~Papernot, P.~McDaniel, and I.~Goodfellow, ``Transferability in machine learning: From
phenomena to black-box attacks using adversarial samples,'' arXiv:1605.07277, 2016.
\bibitem{liu2017} Y.~Liu, X.~Chen, C.~Liu, and D.~Song, ``Delving into transferable adversarial examples and
black-box attacks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2017.
\bibitem{lfw} G.~B. Huang, M.~Ramesh, T.~Berg, and E.~Learned-Miller, ``Labeled Faces in the Wild: A database for
studying face recognition in unconstrained environments,'' Univ. Massachusetts, Amherst, Tech. Rep. 07-49, 2007.
\bibitem{casia} D.~Yi, Z.~Lei, S.~Liao, and S.~Z. Li, ``Learning face representation from scratch,''
arXiv:1411.7923, 2014.
\bibitem{vggface2} Q.~Cao, L.~Shen, W.~Xie, O.~M. Parkhi, and A.~Zisserman, ``VGGFace2: A dataset for recognising
faces across pose and age,'' in \emph{Proc. IEEE Int. Conf. Autom. Face Gesture Recognit. (FG)}, 2018,
pp. 67--74.
\bibitem{facenet} F.~Schroff, D.~Kalenichenko, and J.~Philbin, ``FaceNet: A unified embedding for face recognition
and clustering,'' in \emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2015, pp. 815--823.
\bibitem{inceptionv4} C.~Szegedy, S.~Ioffe, V.~Vanhoucke, and A.~A. Alemi, ``Inception-v4, Inception-ResNet and the
impact of residual connections on learning,'' in \emph{Proc. AAAI Conf. Artif. Intell.}, 2017, pp. 4278--4284.
\bibitem{arcface} J.~Deng, J.~Guo, N.~Xue, and S.~Zafeiriou, ``ArcFace: Additive angular margin loss for deep face
recognition,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2019, pp. 4690--4699.
\bibitem{webface260m} Z.~Zhu, G.~Huang, J.~Deng, Y.~Ye, J.~Huang, X.~Chen, J.~Zhu, T.~Yang, J.~Lu, D.~Du, and
J.~Zhou, ``WebFace260M: A benchmark unveiling the power of million-scale deep face recognition,'' in \emph{Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2021.
\bibitem{lbph} T.~Ahonen, A.~Hadid, and M.~Pietik\"ainen, ``Face description with local binary patterns:
Application to face recognition,'' \emph{IEEE Trans. Pattern Anal. Mach. Intell.}, vol.~28, no.~12,
pp. 2037--2041, 2006.
\bibitem{mtcnn} K.~Zhang, Z.~Zhang, Z.~Li, and Y.~Qiao, ``Joint face detection and alignment using multitask
cascaded convolutional networks,'' \emph{IEEE Signal Process. Lett.}, vol.~23, no.~10, pp. 1499--1503, 2016.
\bibitem{ssim} Z.~Wang, A.~C. Bovik, H.~R. Sheikh, and E.~P. Simoncelli, ``Image quality assessment: From error
visibility to structural similarity,'' \emph{IEEE Trans. Image Process.}, vol.~13, no.~4, pp. 600--612, 2004.
\bibitem{facenetpytorch} T.~Esler, ``facenet-pytorch,'' GitHub repository, 2019. [Online]. Available:
\url{https://github.com/timesler/facenet-pytorch}
\bibitem{deepid2} Y.~Sun, Y.~Chen, X.~Wang, and X.~Tang, ``Deep learning face representation by joint
identification-verification,'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, 2014.
\bibitem{hanjain} H.~Han and A.~K. Jain, ``Age, gender and race estimation from unconstrained face images,''
Michigan State Univ., Tech. Rep. MSU-CSE-14-5, 2014.
\end{thebibliography}

\end{document}
"""

TABLES = {"T_MODELS": T_MODELS, "T_VALID": T_VALID, "T_MAIN": T_MAIN, "T_BASE": T_BASE, "T_MATRIX": T_MATRIX}
FIGS_V = {f"F_{k}": v for k, v in F.items()}
values = {**V, **TABLES, **FIGS_V}


def fill(match):
    return values[match.group(1)]  # KeyError on a typo'd placeholder


(OUT / "paper.tex").write_text(re.sub(r"@([A-Za-z0-9_.]+)@", fill, TEMPLATE), encoding="utf-8")
print("wrote", OUT / "paper.tex")
