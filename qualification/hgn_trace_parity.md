# CPU HGN parity baseline

This check compares Qwen3.8 Flash Next v2 base with Halogen `ppl --rows-out`
for 64 positions, 192 through 255. It uses the first 256 token IDs as input.
The final ID in the reference file is a target token, not an input token.

The Brewery port and the unmodified peer `capture.py` both gave 62 of 64
top-1 matches and mean KL(Halogen || CPU) of `0.009392281638667033`. Both CPU
runs used 32 threads, Torch `2.15.0+cpu`, and Transformers `5.17.0`.

| Position | Halogen top ID | CPU top ID | Halogen logit margin | CPU logit margin |
|---|---|---|---|---|
| 200 | 321 | 1116 | 0.2979965 | 0.1984425 |
| 205 | 733 | 1216 | 0.6996222 | 0.2480326 |

Each margin is the logit for that runtime's top token minus the logit for
the other runtime's top token. The user-approved gate requires at least
62 of 64 top-1 matches, mean KL at most 0.01, and a Halogen margin less
than 1.0 at every miss. This baseline meets the gate. The earlier 63 of 64
measurement used ThinkingCap, not v2 base.

| Input | SHA-256 |
|---|---|
| HGN checkpoint | `71246c6ab3fc1de2cf06326f18e275fe9c2a18366d646ed3357d194c884fc687` |
| N-gram sidecar | `3450acada94e19aabad88bd49b45eb70304820949c2240fc178067b20ae5dffe` |
| Config | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| Token IDs | `5f5b18566fac9607e9ad5d73a029a4833588555b426e886dbbab6f83e785b16f` |
| Halogen image | `773073448dc9ff4a8c6b6c278802e254fd5f1546764e27531aeb3b77cb13482e` |
| Halogen rows | `e534fb47cd5a693e0a04d64857ae116ded3510a27dd30ea115cbd274ccf8e0ce` |

Run `hgn_trace_parity.py` in the pinned HGN environment with the matching
checkpoint, sidecar, config, ID file, and Halogen rows. This result does not
qualify a release.
