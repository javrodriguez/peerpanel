# Credibility map — this repository read against FDA's seven-step framework

PeerPanel is a demonstration system: a local, open-source manuscript-review pipeline built so that its own measurements can be checked by a stranger.
This page maps what this repository records onto the seven steps of the risk-based credibility assessment framework in the FDA draft guidance saved under `docs/fda/`, using that framework's own vocabulary, so that a reader who works in regulated science can see in their own terms which parts of this design are evidenced here and which are not.
This system is not validated for any context of use, nothing here has been submitted to any regulator, and this page is a mapping exercise on a demonstration system rather than a statement about regulatory standing.

**Source.** Considerations for the Use of Artificial Intelligence To Support Regulatory Decision-Making for Drug and Biological Products (U.S. Food and Drug Administration, January 2025, docket FDA-2024-D-4689) — the PDF was fetched from https://www.fda.gov/media/184830/download on 2026-09-04T02:38:53Z, the landing page read "Draft Guidance" when it was checked at 2026-09-04T02:39:59Z, the saved copy has sha256 `62352ed0fc17…`, and the whole record — the landing URL, the extraction command, and the text file's own sha256 — is in `docs/fda/PROVENANCE.json`.

Each heading below is quoted verbatim from that saved text.
Under each one: the files in this repository that bear on the step and what they show, or a plain statement that the step is not evidenced here.

## Step 1: Define the Question of Interest

The question this repository's evaluation asks is written down beside the numbers it produced.
`results/RESULTS.md` states, for each measurement, what was asked, of which manuscripts, and over which corpus: the retrieval ladder asks which rung finds the documents a manuscript's own reference list says are relevant, and the planted-error evaluation asks whether a review panel finds defects that a single agent given the panel's token spend does not.
`README.md` carries that second question and the answer the records show, including where the answer does not favour the architecture this repository was built to demonstrate.
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
`src/peerpanel/evals/planted.py` defines the defects planted into a held-out manuscript, the text a grader is allowed to score, and the rule that decides whether a finding counts as a catch.
`tests/test_planted_soundness.py` holds that rule to its controls — strings that must score, strings that must not — so a reader can see what the rule credits and what it refuses before reading any number the rule produced.
`DECISIONS.md` carries the reasoning behind the protocol: that both arms of the comparison are scored by one function, that a prompt is sized so the model reads it whole, and that a small-n aggregate publishes its per-unit numbers or publishes nothing.
`LIMITATIONS.md` states, beside each known failure mode of language-model review, whether this design mitigates it, measures it, or simply carries it — which is the part of the plan that says in advance what this evaluation cannot settle.

## Step 5: Execute the Plan

`results/planted-eval-caprin-heterochromatin.json` and `results/planted-eval-met17-auxotroph.json` are the executed planted-error runs: each names the errors planted, what each arm found and missed, the tokens and wall-clock it spent, and the run conditions of its own calls — how many were made and the smallest margin between a prompt and the window that prompt ran in.
The raw capture of each run is committed beside it, at `results/planted-eval-caprin-heterochromatin.log` and `results/planted-eval-met17-auxotroph.log`, and the text of every finding a grader scored is committed inside the record rather than summarised, so a reader can re-score the run by hand.
`results/panel-review-met17-auxotroph.json` and `results/panel-review-met17-auxotroph-run2.json` apply the same discipline to a full panel run on a real manuscript, and `results/build-stats-ci.json` and `results/build-stats-demo.json` do it for the two index builds, each naming the provider that served every call.
Every record under `results/` names the one command that regenerates it, in the table at the top of `results/RESULTS.md`.

## Step 6: Document the Results of the Credibility Assessment Plan and Discuss Deviations From the Plan

`results/RESULTS.md` is the report: the tables, the reading of each one, and the readings that do not flatter the system, in the same voice as the ones that do.
`tests/test_published_numbers.py` reads figures back out of the committed records and compares them to the text of that page, so a number that drifts from its record fails a test rather than standing.
The deviations are documented in the same place as the results rather than in a footnote: `DECISIONS.md` D19 records that prompts were being cut by a window this project had mis-assumed, that every record produced before 2 September 2026 was withdrawn rather than kept, what it cost to re-run them all, and the measurement that ruled out the cheaper road of keeping some.
`LIMITATIONS.md` states what the numbers do not prove, and `README.md` reports the headline result as the record shows it, a loss included.

## Step 7: Determine the Adequacy of the AI Model for the Context of Use

Not evidenced here — this is a demonstration system.
Determining adequacy is a judgement a sponsor and a regulator make against a defined context of use and an assessed model risk, and no result in this repository is offered toward such a judgement.
What would be needed is a credibility assessment report tied to a real context of use, read by the people who own the decision, together with the life-cycle monitoring this framework asks for where a model stays in use — none of which exists here.

---

**Where the map runs out.**
Two of the seven steps are not evidenced here, and that is the honest shape of a demonstration system: the steps this repository can show are the ones about stating the question, writing the protocol down before running it, executing it, and reporting what came back — including the runs that went against the architecture this repository was built to demonstrate.
The two it cannot show are the ones that need a real decision, a real sponsor and a real regulator.
No sentence on this page says this system is validated, compliant or qualified for anything, because none of that would be true.
