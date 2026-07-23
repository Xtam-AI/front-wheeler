# Paper outline (target: arXiv, before Sept 17)

Working title ideas:
- "Wiring, Not Weights: How Local Can Supermask Search Get?"
- "Front Wheeler: Plateau-Triggered Backpropagation" (if exp2 becomes the lead)

Possibly two short notes rather than one paper — decide once results are in.

## Note A — learning is wiring, never updating
**Claim under test:** capacity pre-exists (random weights suffice); learning is
connectivity search. Supermasks prove the target subnetwork exists (edge-popup);
the open gap is whether a *local* rule can find it without smuggling BP back in.

**Axis of the experiment: how much global information does the learning signal
carry?**
1. `ep` — full BP gradient on mask scores (upper bound, known: Ramanujan et al. 2020)
2. `dfa` — fixed random projection of the output error vector (no weight
   transport; Nøkland 2016 applied to mask scores instead of weights — the novel cell)
3. `rmhebb` — one global scalar per sample (three-factor Hebbian, REINFORCE-flavored)
4. controls: random mask, readout-only, dense BP

**Expected figure:** accuracy degrades gracefully along the locality axis; the
gap between rows = "price of locality" for wiring-based learning.

Empirical notes so far:
- straight-through subtlety: selecting by |s| requires the sign(s) factor in
  the score update; dropping it kills learning entirely (worth a remark).
- naive reward-modulated Hebbian diverges: ReLU post*pre is all-positive, mask
  self-amplifies co-active synapses, logits explode, exploration dies. Fixes
  that made it learn: batch-centered (covariance) eligibility, RMS-normalized
  per-layer updates, exact local delta rule at the readout, loss-based scalar
  instead of sampled-action reward.

Related work to cite: Ramanujan et al. 2020 (What's Hidden in a Randomly
Weighted NN), Zhou et al. 2019 (supermasks), Nøkland 2016 (DFA), Lillicrap et
al. 2016 (feedback alignment), Frémaux & Gerstner 2016 (three-factor rules),
Gaier & Ha 2019 (weight-agnostic networks), Malach et al. 2020 (proving the
lottery ticket hypothesis / pruning is all you need).

## Note B — front wheeler
**Claim under test:** most training steps don't need deep credit assignment.
Bounce between the readout and the last hidden layer (backward touches only 2
of D weight matrices); trigger full-BP bursts only when the loss EMA plateaus.
Goal: match full-BP accuracy with a small fraction of full-BP steps.

**Schedules:** full / front-only / periodic-N (fixed-budget control) / fw
(adaptive). Key figure: final accuracy vs backward-FLOPs (Pareto front); fw
should sit above the periodic curve at matched budget if adaptivity matters.

**Honest framing:** related to greedy layer-wise training, decoupled/synthetic
gradients (Jaderberg 2017), "freeze-out"/layer freezing schedules, linear
probes. The specific plateau-triggered *burst* schedule and the BP-step-count
objective are the angle. Must show fw beats periodic at matched budget —
otherwise the honest conclusion is "a fixed schedule suffices."

## Timeline (deadline Sept 17, today Jul 23)
- Jul: toys working end-to-end (done-ish), sweeps
- Aug 1-15: lock experiments, CIFAR-10 replication if MNIST results hold
- Aug 15-31: writing, related-work pass
- Sep 1-10: polish, internal review
- Sep 15: submit buffer
