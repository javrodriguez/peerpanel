# Credibility map — this repository read against FDA's seven-step framework

PeerPanel is a demonstration system: a local, open-source manuscript-review pipeline built so that its own measurements can be checked by a stranger.
This page maps what this repository records onto the seven steps of the risk-based credibility assessment framework in the FDA draft guidance saved under `docs/fda/`, using that framework's own vocabulary, so that a reader who works in regulated science can see in their own terms which parts of this design are evidenced here and which are not.
This system is not validated for any context of use, nothing here has been submitted to any regulator, and this page is a mapping exercise on a demonstration system rather than a statement about regulatory standing.

**Source.** Considerations for the Use of Artificial Intelligence To Support Regulatory Decision-Making for Drug and Biological Products (U.S. Food and Drug Administration, January 2025, docket FDA-2024-D-4689) — the PDF was fetched from https://www.fda.gov/media/184830/download on 2026-09-04T02:38:53Z, the landing page read "Draft Guidance" when it was checked at 2026-09-04T02:39:59Z, the saved copy has sha256 `62352ed0fc17…`, and the whole record — the landing URL, the extraction command, and the text file's own sha256 — is in `docs/fda/PROVENANCE.json`.

Each heading below is quoted verbatim from that saved text.
Under each one: the files in this repository that bear on the step and what they show, or a plain statement that the step is not evidenced here.

## Step 1: Define the Question of Interest

The question this repository's evaluation asks is written down beside the numbers it produced.
`results/RESULTS.md` states, for each measurement, what was asked, of which manuscripts, and over which corpus: the retrieval ladder asks which rung finds the documents a manuscript's own reference list says are relevant, and the planted-error evaluation asks whether a review panel asserts defects that a single agent, run alone on each of the two named local models under the same protocol, does not.
Each planted-error record also states the budget its question was asked under, and says where the comparison falls short of the equal-compute one it was designed as: both single-agent arms hit their sampling ceiling before reaching the panel's token spend, and the record's own note says so rather than leaving a reader to work it out from the totals.
`README.md` carries that second question and the answer the records show — a null result, in which no arm asserted any planted defect — including where that answer does not favour the architecture this repository was built to demonstrate.
No question asked here concerns a patient, a product, or a decision anyone acts on.

## Step 2: Define the Context of Use for the AI Model

`LIMITATIONS.md` opens with a scope block that states the intended use — a pre-submission self-check on already-public manuscripts, run by their own authors against a pinned open-access corpus — and the uses ruled out, including manuscripts under confidential review.
`README.md` repeats that scope where a reader meets the project first, and `DECISIONS.md` records why the corpus, the reviewed excerpt and the model roles are what they are, so the setting every number was taken in is legible rather than implied.
That setting is a demonstration one: there is no drug, no submission and no decision downstream of this system's output, so what those files describe is the scope of an experiment and not a context of use in the sense this step means.

## Step 3: Assess the AI Model Risk

Not evidenced here — this is a demonstration system.
Model risk in this framework combines model influence with decision consequence, and both are properties of a decision that this system's output never reaches.
What would be needed is a named question of interest with a real decision attached, an assessment of how much this system's output contributes to that decision relative to other evidence, and a judgement of the consequence of an incorrect decision — reached with the parties who own the decision, not inside a repository like this one.

## Step 4: Develop a Plan to Establish AI Model Credibility Within the Context of Use

The evaluation protocol here is committed code rather than a description of one.
`src/peerpanel/evals/planted.py` defines the defects planted into a held-out manuscript, the text a grader is allowed to score, and the rule that decides whether a finding counts as a catch — `assertion-v1`, named in the code and carried in every record it scored.
That rule was written and frozen before it was run against anything, and `tests/test_planted_soundness.py` pins its cue list as frozen text, so changing what the rule credits means changing a test whose whole point is that the rule does not move.
`tests/fixtures/round3_scored_strings.json` is the negative control it was frozen ahead of: all 230 strings the previous round's records scored, which that same test re-derives from those records at the named commit rather than trusting the fixture as a copy, and not one of which the rule may credit.
The test then measures why that control is a floor rather than evidence — no assertion cue occurs in any of the 230 strings — so the weight is carried by controls built from real text instead: manuscript sentences containing phrases a careless rule would fire on, which must score nothing, and real record strings with a genuine assertion appended, which must score.
`DECISIONS.md` carries the reasoning behind the protocol: that both arms of the comparison are scored by one function, that a prompt is sized so the model reads it whole, and that a small-n aggregate publishes its per-unit numbers or publishes nothing.
`LIMITATIONS.md` states, beside each known failure mode of language-model review, whether this design mitigates it, measures it, or simply carries it — which is the part of the plan that says in advance what this evaluation cannot settle.

## Step 5: Execute the Plan

`results/planted-eval-caprin-heterochromatin.json` and `results/planted-eval-met17-auxotroph.json` are the executed planted-error runs: each names the errors planted, the rule that scored them, and, for each of the three arms — the panel, and one single-agent arm per named local model — which planted defects that arm's own prose asserted, which planted tokens it named at all, which it missed, the tokens and wall-clock it spent, and the run conditions of its own calls.
Those two counts are not interchangeable and the record does not allow them to be read as one: the assertion count is the published number even at zero, and the naming count sits beside it as a labelled upper bound, because a finding that names a planted token while asserting nothing is a restatement of the manuscript.
The run conditions are recorded per call rather than per run — the calls made, the largest prompt, the smallest window served, the smallest margin between the two, and the field each call's window was read from — so a reader can tell from the record alone that no prompt was cut.
The raw capture of each run is committed beside it, at `results/planted-eval-caprin-heterochromatin.log` and `results/planted-eval-met17-auxotroph.log`, and the text of every finding a grader scored is committed inside the record rather than summarised, so a reader can re-score the run by hand.
`results/panel-review-met17-auxotroph.json` and `results/panel-review-met17-auxotroph-run2.json` apply the same discipline to a full panel run on a real manuscript, `results/summaries-stats-demo.json` does it for the community-report layer the global retrieval rung reads, and `results/build-stats-ci.json` and `results/build-stats-demo.json` do it for the two index builds; every one of those records names the model and the wire that served its calls, alongside the same per-call window figures.
`results/derived-fields.json` names the fields that were written by derivation rather than measured at the call, and the command that re-measures each, so a weaker field is labelled instead of passing as the rest.
Every record under `results/` names the one command that regenerates it, in the table at the top of `results/RESULTS.md`.

## Step 6: Document the Results of the Credibility Assessment Plan and Discuss Deviations From the Plan

`results/RESULTS.md` is the report: the tables, the reading of each one, and the readings that do not flatter the system, in the same voice as the ones that do.
What it has to report is a null result — under the frozen rule no arm asserted any planted defect in either manuscript — reached by a panel that spent several times the tokens either single-agent arm spent.
`tests/test_published_numbers.py` and `tests/test_planted_published_numbers.py` read figures back out of the committed records and compare them to the text of that page, the second parsing each table into rows and columns first, so a table that renamed an arm or dropped a column fails on its shape before any number is read.
The deviations are documented in the same place as the results rather than in a footnote: `DECISIONS.md` D19 records that prompts were being cut by a window this project had mis-assumed, that every record produced before 2 September 2026 was withdrawn rather than kept, what it cost to re-run them all, and the measurement that ruled out the cheaper road of keeping some.
Regenerating those records exposed two further defects in this repository's own machinery — a cached community report served under another community's membership, and a cold run whose reports outnumbered its calls because it read back entries it had written minutes earlier — and `tests/test_graph_summaries.py` now builds both situations deliberately, so each is a deviation fixed and pinned rather than a number quietly absorbed.
`LIMITATIONS.md` states what the numbers do not prove, and `README.md` reports the headline result as the record shows it.

## Step 7: Determine the Adequacy of the AI Model for the Context of Use

Not evidenced here — this is a demonstration system.
Determining adequacy is a judgement a sponsor and a regulator make against a defined context of use and an assessed model risk, and no result in this repository is offered toward such a judgement.
What would be needed is a credibility assessment report tied to a real context of use, read by the people who own the decision, together with the life-cycle monitoring this framework asks for where a model stays in use — none of which exists here.

---

**Where the map runs out.**
Two of the seven steps are not evidenced here, and that is the honest shape of a demonstration system: the steps this repository can show are the ones about stating the question, writing the protocol down before running it, executing it, and reporting what came back — including the runs that went against the architecture this repository was built to demonstrate.
The two it cannot show are the ones that need a real decision, a real sponsor and a real regulator.
No sentence on this page says this system is validated, compliant or qualified for anything, because none of that would be true.
