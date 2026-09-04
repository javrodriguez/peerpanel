"""Shared prose-rule constants for the result-sentence guards.

Two test files enforce the same rule from opposite ends — `test_first_screen.py`
over the README's first screen and `test_planted_published_numbers.py` over the
published tables — and the rule is only meaningful if both read the SAME word
lists. Holding a second copy in each file is the drift shape this repository
keeps catching: one list gets a new synonym, the other does not, and the guard
that still matters silently stops covering the sentence that matters.

The leading underscore keeps pytest from collecting this module as a test file.

The rule these lists serve (requirement 10, and the contract's "a loss or a null
result, never a win"): when no manuscript has the panel arm asserting more
planted errors than every single-agent arm, the published prose may not claim the
panel won, and must name the outcome as the loss or null it is.

`WIN_WORDS` is deliberately anchored on `panel` and bounded to the same sentence
(at most 40 non-period characters between the subject and the verb), so a
sentence about some other subject winning — a retrieval rung, a baseline — does
not trip it, and a period between them ends the reach.
"""

from __future__ import annotations

import re

WIN_WORDS = re.compile(
    r"\bpanel\b[^.]{0,40}\b(wins?|won|beats?|beat|outperform(s|ed)?|found more|"
    r"came out ahead|did better)\b",
    re.I,
)

LOSS_WORDS = re.compile(r"\b(loss|loses|lost|null result|neither)\b", re.I)
