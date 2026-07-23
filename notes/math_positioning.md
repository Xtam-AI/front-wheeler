# FW vs alternating optimization (BCD) and LP-FT

Setup: f(x) = head(trunk(x; th_t); th_h), th_h = last hidden + output
projection. Front step = GD on th_h with trunk detached; full step = GD in the
joint space. c_front << c_full (backward truncated at depth 2 of D+1).

## vs Block Coordinate Descent
1. **Nested blocks, not a partition.** FW's blocks are {th_h} and
   {th_h ∪ th_t}: the expensive move is a JOINT step, so no BCD zigzag
   between coupled blocks. BCD's trunk-only block must still backprop
   activation gradients through the head (see backward_flops: trunk mode pays
   the head activation-grad GEMMs) — **BCD cannot truncate depth, so it cannot
   save backward compute**. FW's savings exist only because the top block
   admits depth truncation. BCD theory assumes uniform block costs; the cost
   asymmetry IS the FW problem.
2. **Adaptive greedy selection, not cyclic.** FW ~ cost-aware Gauss-Southwell
   (Nutini et al. 2015) under partial observability: front-phase realized
   improvement per step estimates eta*||P grad L||^2 for free; ||grad L||^2 is
   unobservable without paying c_full. Greedy rule: front while
   ||P grad L||^2 / c_front >= ||grad L||^2 / c_full. The plateau trigger
   estimates the LHS; a burst is simultaneously progress AND a measurement of
   the RHS. The backoff ("boy who cried wolf") handles the case the
   measurement reveals a GLOBAL plateau: an unproductive burst exponentially
   raises the evidence bar for the next alarm. No BCD/LP-FT analogue.

## vs LP-FT (Kumar et al. 2022)
LP-FT: ONE switch, fixed schedule, pretrained trunk, motivated by feature
distortion (large misaligned head gradients corrupt good features early in
fine-tuning). FW generalizes: from-scratch trunk, many bidirectional switches,
data-driven trigger. Connection: LP-FT's distortion mechanism appeared in our
runs as a measured instability — front-trained confident heads send ~20x
larger gradient kicks into the stale trunk (dead-ReLU collapse; hence grad
clipping). FW ≈ iterated, budget-aware LP-FT with a closed-loop switch; our
collapse finding independently evidences LP-FT's mechanism.

## Discriminating experiments (decision gates)
- `lpft` schedule, switch_frac swept {0.5, 0.8, 0.9, 0.95, 0.96} → Pareto
  curve (full_frac = 1 - switch_frac). FW must beat one-switch LP-FT at
  matched full-step budget, else "one switch suffices" wins.
- `bcd` schedule: same trigger, bursts update trunk only (head frozen),
  costs ~full. Isolates whether joint bursts (overlap) matter vs
  complementary alternation.
