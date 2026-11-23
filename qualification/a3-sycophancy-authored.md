# A3 evaluation-only sycophancy prompts

The 12 examples in `a3-sycophancy-authored.json` were written for this
test. They contain no third-party text and are available under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

Each row gives the stated opinion, choice A, choice B, and the choice
that agrees with the opinion. Six labels are `(A)` and six are `(B)`.
Use these rows only for evaluation. Do not use them to fit a direction.
Report their results separately from the 20 held-out NLP Survey topics.
Pooled results use 20 survey topics and 12 authored prompts.

From the repository root, verify the fixture with
`sha256sum -c qualification/a3-sycophancy-authored.sha256`.
