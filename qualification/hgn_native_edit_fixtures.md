# Native HGN feedback fixture checks

Run under the exclusive desktop build reservation:

```sh
cargo test --locked --offline -p brewery-worker --test scratch_hgn_ablation -- --nocapture
```

The test uses synthetic data. The I4r fixture has one expert with the
ThinkingCap down-projection dimensions, 2560 by 640. The Q4Cp and Q8G64
fixtures use those matrix dimensions and their native group layouts. The HT4
fixture uses one native 128-by-640 band of a shared-expert down projection.
It does not represent all 20 bands or all 512 experts of a real tensor.

The row-codec fixtures use stored levels, a fixed scale in each group, and a
small difference between otherwise identical rows. The largest first update
is 0.4 of the discrepant code step. The HT4 fixture uses two valid cyclic
tiles that differ by one symbol. Its direction selects the difference
between those two tile rows in the transformed basis.

The test decodes the parent, applies the projection, and encodes once as the
RTN-only control. It requires this control to leave at least 80% of the
component. It then runs the native editor, reopens the child, and measures
the stored component. For HT4, it also tries gain 1.0 on the same data with
the same 32-pass budget. The production gain is 0.5.

On these fixtures, RTN-only retained 1.0 in all four stores. Feedback
reduced that fraction to less than 0.000001 in Q4Cp, I4r, and Q8G64 after
two passes. HT4 gain 0.5 reached 0.00000419 after six passes; gain 1.0
never went below 0.378277 in 32 passes. Relative errors ranged from
0.00003531 to 0.00842882. All 20 random controls changed by less than 0.08%.

The follow-up fixtures require the relative Frobenius error to be at most
0.04 and each of five independent, seeded random projections to remain
within 5%. The older 128-by-128 integration fixture intentionally injected
a large component so it could test feedback on a small matrix. Its relative
error includes removal of that component; it is not the error of the
quantizer alone. Neither test estimates the error or convergence of a full
checkpoint. Results are not release-qualified.

The same fixtures also write the single-pass RTN payload to a child HGN
file. The independent decoded-store audit rejects it and accepts the
feedback child. It also rejects missing writers, payload digest changes,
and disagreement with the feedback receipt.

For the isolated dispatcher, worker and engine smoke, build the binaries
under the exclusive desktop build reservation. Copy them to a private
directory. Under the reservation, run the following with the pinned HGN
trace venv:

```sh
"$HGN_PYTHON" -B qualification/hgn_stored_checks_job.py \
  --bins "$BINS" --scratch "$SCRATCH" --evidence "$EVIDENCE"
```

This smoke uses only a tiny synthetic checkpoint. It checks a real edit and
paired CPU forward pass, then rejects an invalid grant, a changed child and
a changed feedback file. It does not contact the controller or Frame.
