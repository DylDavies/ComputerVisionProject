# Mock review: IEEE paper and repository

**Mock grade: 70/100 (a borderline first)**

This is a strong, well-engineered project with an honest write-up. Its central conclusion, though, claims more than
the experimental design can isolate, and a few factual inconsistencies would cost marks with a careful marker.

| Area | Mark | Comment |
|---|---|---|
| Problem and motivation | 8/10 | A clear, real-world question, and it follows the proposal faithfully. |
| Method and experimental design | 14/25 | Careful protocol, but the architecture-vs-data claim is confounded (items 1–2). |
| Results and analysis | 17/25 | Rich results, good baselines, and an honest correction of our own first reading. The causal language runs ahead of the evidence. |
| Writing and presentation | 13/20 | Readable, but the abstract is overloaded, some sentences contradict each other, and the figures are stranded after the references. |
| Code and reproducibility | 18/20 | Clean, small and tested code with a validation gate. The paper-generation pipeline had fallen out of sync (item 3, now fixed). |

**To reach 80+:** fix items 1 (at least the soft fix), 2 and 4, add one ensemble attack row (item 7), and trim the
abstract (item 11). These are mostly writing changes plus about 30 minutes of GPU time.

Status key: ☐ open · ☑ done

---

## Serious problems (fix these first)

### 1. ☐ The "architecture beats data" conclusion is confounded

**Problem.** casia and vggface2 share more than an architecture. They also share the same crop (MTCNN box, margin 32),
input size, normalisation and preprocessing pipeline. ArcFace differs in architecture *and* loss, data, crop, input
size and accuracy. When edits move between casia and vggface2 but not to ArcFace, shared preprocessing or ArcFace's
angular-margin loss explains that just as well. Our own Discussion admits crop geometry matters (ArcFace→surrogate
29.6% vs ArcFace→casia 16.2% at 8/255), which undermines the main claim. There is also only one non-Inception
model, so "architecture" rests on a sample of one.

**How to fix.**
- *Cheap (writing only):* soften the claim everywhere (abstract, contributions, Results V-D, Discussion,
  Conclusion) to "transfer tracks shared architecture *and preprocessing*". Add the confound to Limitations:
  architecture, crop, input size and loss vary together, and only one non-Inception model was tested.
- *Proper (one training run):* train a second surrogate, a ResNet-50 with a softmax loss on CASIA-WebFace, using the
  same crop as the current surrogate. It shares the current surrogate's data, loss and crop, so architecture is
  the only difference. This is the controlled experiment the research question actually needs.

### 2. ☐ The paper contradicts itself about the casia model

**Problem.** Method → Models says casia "differs from it only in initialisation". The Introduction, however, says it
was "trained separately, with a different crop and training procedure", and Table I shows a box crop for casia vs a
5-pt crop for the surrogate. The same false claim appears in `src/facetransfer/train.py` (docstring, lines 3–4) and
in `README.md` ("nothing but the seed").

**How to fix.** In all three places, say that casia differs in crop and training procedure but not in architecture,
loss or training data. In the paper, edit the template in `report/ieee.py` (see item 3).

### 3. ☑ Hand edits to `paper.tex` would be wiped on regeneration (fixed)

**Problem.** `report/ieee.py` generates `paper.tex` from a template, but everyone had been editing `paper.tex`
directly. Running `ieee.py` would have thrown away the abstract/introduction rewrite and all later wording changes.

**Fix applied.** The template in `report/ieee.py` was rebuilt from the current `paper.tex`, with every result number
turned back into its `@placeholder@`. It reproduces the current `paper.tex` exactly.

**From now on:** edit the text in `report/ieee.py`, not in `paper.tex`, then run
`uv run python report/ieee.py && cd report/ieee && latexmk -pdf paper.tex`.

### 4. ☐ ε = 0.5/255 isn't really a 0.5/255 budget

**Problem.** Edits are rounded to 8-bit before scoring. A pixel moved by exactly ±0.5 levels rounds to ±1, and PGD
pushes many pixels to the ±ε bound. A numerical check with pixels at the ±0.5 bound found that 62% change by a full
level, with a maximum change of 1/255. The "0.5/255" row is therefore really "≤1/255 with fewer pixels changed". The
SSIM column is correct, because SSIM is measured after rounding.

**How to fix.** Either drop ε = 0.5/255 from the sweep, tables and figure, or keep it and add one sentence to Method →
Attack: after 8-bit rounding, the 0.5/255 budget allows changes of up to 1/255.

---

## Moderate issues

### 5. ☐ Causal language is stronger than the evidence

**Problem.** The paper says edits "survive" JPEG, ArcFace "resists", and architecture "predicts" transfer, with 4
attackers and only one seed for three of them.

**How to fix.** Use "is consistent with", "suggests" or "in our experiments" for the architecture claim in the abstract,
Results V-D and Conclusion. Descriptive verbs for measured outcomes are fine; the causal claims are the problem.

### 6. ☐ The cross-attacker matrix is single-seed

**Problem.** Only the surrogate sweep was repeated over 3 seeds. Table IV, the headline result, is from one seed.

**How to fix.** Re-run the casia, vggface2 and arcface attackers with 2 more seeds, at least at ε = 8/255:
`uv run ft sweep --attacker <name> --seed <s> --conditions pgd --eps 8`. Alternatively, state in the Table IV caption
that it comes from a single seed.

### 7. ☐ No transfer-boosting attacks

**Problem.** A marker will ask whether standard transfer techniques change the picture. Examples are momentum
(MI-FGSM), input diversity (DI) or an ensemble attack (Liu et al., which we already cite).

**How to fix.** Add an ensemble attacker: in `attack.pgd`, sum the distances from 2–3 models (each with its own
affines) instead of one. A single extra row, e.g. casia+vggface2 → ArcFace at 8/255 and 16/255, is enough to report.

### 8. ☐ "Kept the final epoch rather than the best" reads like a mistake

**Problem.** It is actually the correct choice, because choosing the best epoch by LFW accuracy would mean selecting
on the test set. As written, though, it sounds like an apology.

**How to fix.** In the Discussion, reword it as a deliberate choice: "we kept the final epoch, because selecting the
best epoch on LFW would tune on the test set."

### 9. ☐ SSIM is computed on the full 250×250 photo

**Problem.** The photo includes background the edit barely touches, so SSIM looks better than it would on the
face itself.

**How to fix.** Compute SSIM on each attacker's face crop as well (one extra `warp` call in `evaluate.ssim`). As a
minimum, state in Method → Baselines and metrics that SSIM is measured on the full photo.

### 10. ☐ Proposal items not delivered: demographic check and LPIPS

**Problem.** The proposal promised a check of whether protection is equally effective across demographic groups
("if time allows") and offered LPIPS as an alternative to SSIM. Both are only mentioned in Limitations.

**How to fix.** A per-group breakdown is cheap: join the per-pair results with LFW's public gender/race attribute
files and report the unmatched share per group at one budget. If there is no time, leave it in Limitations as is.

---

## Writing and presentation

### 11. ☐ The abstract is too long

**Problem.** It is about 300 words (IEEE asks for 150–250) and contains too many numbers.

**How to fix.** Keep three results: the 4/255 surrogate result, vggface2→casia 82.2% vs surrogate→casia 16.2% at
8/255, and ArcFace ≤ 18.8%. Move the rest to the Results section.

### 12. ☐ Figures 6–7 come after the references

**How to fix.** Move the example and perturbation figures into Results → "What the edits look like", even if they
are smaller there. Then remove the `[p]` placement and the comment about putting them after the references.

### 13. ☐ The third contribution is engineering, not research

**Problem.** "A pipeline in which the same edited photo is cropped separately for each model…" is good practice, but
listing it as a contribution invites "that's not a contribution".

**How to fix.** Drop it from the contributions list. It is already described in Method → Preprocessing.

### 14. ☐ "Discouraging for privacy tools" over-generalises

**Problem.** The paper generalises from plain PGD on LFW with four models to Fawkes and LowKey, which use different
attacks (feature-space targets, ensembles, LPIPS constraints).

**How to fix.** Qualify it, e.g. "For edits built like ours, against a single model, the protection is much weaker…".

### 15. ☐ No significance testing

**Problem.** The paper reports fold standard deviations, but none of the key comparisons has a significance test.

**How to fix.** Add a paired McNemar test or a bootstrap confidence interval for the main comparisons, e.g.
vggface2→casia vs casia→vggface2 at 8/255, and surrogate→casia vs vggface2→casia. This takes a few lines of
code using the per-pair unmatched flags.

---

## Repository

### 16. ☐ Thin tests and no CI

**Problem.** There are 5 tests (thresholds, warp, PGD budget, log parsing). LBPH, the JPEG/quantise round trip and
`sweep` are untested, and there is no CI.

**How to fix.** Add tests for LBPH (output shape, zero distance to itself), `attack.jpeg`/`quantise` (shape, dtype,
range) and one `sweep` smoke test on random tensors with a tiny model. Add a GitHub Action that runs `ruff check` and
`pytest` (about 15 lines).

### 17. ☐ Committed build artefacts and two parallel reports

**Problem.** `report/main.pdf`, `results_summary.pdf` and `report/ieee/paper.pdf` are committed. Figures are
duplicated in `report/figures/` and `report/ieee/figures/`. Two reports exist (`report/main.tex` and
`report/ieee/paper.tex`), and it is unclear which one is the submission.

**How to fix.** Decide which report is the submission and delete or archive the other, together with its generator.
Keep only the final submitted PDF in git and ignore the rest.
