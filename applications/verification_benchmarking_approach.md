# Verification-Based Benchmarking of Quantum Computers

## 1. Core Idea

The goal is to benchmark quantum computers using **verification protocols** rather than only traditional statistical benchmarks such as cross-entropy benchmarking (XEB), randomized benchmarking, or random-circuit sampling.

The central question is:

\[
\text{Can this noisy quantum computer perform a computation that can be verified securely within a bounded resource budget?}
\]

Instead of asking whether a device reproduces the output statistics of a random circuit family, we ask whether the device can pass a rigorous verification protocol for computations from a target class \(\mathfrak C\).

A device passes the benchmark if there exists a verification-protocol configuration such that:

\[
\epsilon_{\mathrm{security}} \leq \epsilon_{\mathrm{target}},
\]

\[
N_{\mathrm{rounds}} \leq N_{\max},
\]

and the protocol does not reject simply because of the device's expected honest noise.

For example, one concrete benchmark instance could be:

\[
\epsilon_{\mathrm{target}} = 10^{-2},
\qquad
N_{\max} = 2000.
\]

The benchmark question becomes:

> Can this computer reach security level \(10^{-2}\) using fewer than 2000 verification rounds, while tolerating its honest noise?

If yes, the computer passes the benchmark for the chosen workload, noise model, and protocol.

---

## 2. Analytic Benchmark Formulation

In the analytic setting, we assume a simplified noise parameter, for example an effective honest failure probability \(p_{\mathrm{err}}\). A verification protocol provides a tolerance threshold \(w/s\), where:

- \(s\) is the number of test or trap rounds,
- \(w\) is the maximum number of tolerated failed test rounds,
- \(w/s\) is the tolerated honest failure rate,
- \(d\) is the number of computation-related rounds,
- \(d+s\) is the total number of rounds.

The benchmark pass condition is:

\[
\exists (d,s,w,\tilde\delta,\lambda_\phi)
\]

such that:

\[
\epsilon_{\mathrm{bound}} \leq \epsilon_{\mathrm{target}},
\]

\[
d+s \leq N_{\max},
\]

and

\[
\frac{w}{s} \geq p_{\mathrm{err}}.
\]

For the example:

\[
\epsilon_{\mathrm{target}} = 10^{-2},
\qquad
p_{\mathrm{err}} = 0.1,
\qquad
N_{\max}=2000,
\]

we ask:

\[
\exists (d,s,w,\tilde\delta,\lambda_\phi)
\quad
\text{such that}
\quad
\epsilon_{\mathrm{bound}} \leq 10^{-2},
\quad
d+s \leq 2000,
\quad
w/s \geq 0.1.
\]

If such a design exists, the device passes.

---

## 3. Why Simulation Is Needed

The analytic setting is useful, but it relies on an idealized noise parameter. In real devices, noise may not be captured by a single scalar probability.

Realistic noise can include:

- gate-level depolarizing noise,
- biased Pauli noise,
- dephasing,
- amplitude damping,
- measurement noise,
- coherent over-rotations,
- correlated errors,
- crosstalk,
- spatially non-uniform noise,
- temporal drift.

Therefore, instead of assuming a universal circuit-level error probability, we should use simulations of the actual verification protocol under different noise models.

The role of simulation is not necessarily to infer one global fake noise parameter. Rather, it is to estimate a **protocol-level honest failure probability**:

\[
\hat p_{\mathrm{honest}}.
\]

This quantity depends on the circuit, protocol, and noise model:

\[
\hat p_{\mathrm{honest}}
=
\hat p_{\mathrm{honest}}(\text{width},\text{depth},\text{noise model},\Pi),
\]

where \(\Pi\) is the verification protocol.

Thus, the effective honest failure probability is generally size-dependent. A fixed microscopic gate error rate can lead to larger protocol-level failure probability as circuit width or depth increases.

---

## 4. Simulation-Based Benchmark Pipeline

For a fixed noise model, the proposed pipeline is:

\[
\text{noise model}
\rightarrow
\text{simulate verification protocol}
\rightarrow
\hat p_{\mathrm{honest}}
\rightarrow
\text{round optimizer}
\rightarrow
\text{PASS/FAIL}.
\]

More concretely:

1. Choose a verification protocol \(\Pi\).
2. Choose a computation class or benchmark circuit family \(\mathfrak C\).
3. Choose a fixed noise model \(N\).
4. For each circuit shape, such as width-depth pair \((W,D)\), simulate the verification protocol under \(N\).
5. Estimate the honest failure probability:

   \[
   (W,D) \mapsto \hat p_{\mathrm{honest}}(W,D).
   \]

6. Feed \(\hat p_{\mathrm{honest}}(W,D)\) into the verification-round optimizer.
7. Determine whether security \(\epsilon_{\mathrm{target}}\) can be reached within the round budget \(N_{\max}\).
8. Mark the tile as PASS or FAIL.

---

## 5. First Landscape: Honest Failure Probability Heatmap

For a fixed noise model, we first build a heatmap:

\[
(W,D) \mapsto \hat p_{\mathrm{honest}}(W,D).
\]

Here:

- \(W\) is circuit width,
- \(D\) is circuit depth,
- each tile corresponds to a family of circuits of that size,
- the tile value is the simulated honest failure probability.

This heatmap answers:

> Under this noise model, how often does an honest noisy device fail the verification protocol as circuit size increases?

This is already useful because it shows where the protocol is robust or fragile with respect to realistic noise.

---

## 6. Second Landscape: Verification Pass/Fail Heatmap

The second heatmap is derived from the first.

For each width-depth tile, we ask:

> Given the simulated honest failure probability at this tile, can the verification protocol achieve target security \(\epsilon_{\mathrm{target}}\) using at most \(N_{\max}\) total rounds?

Formally:

\[
(W,D)
\rightarrow
\hat p_{\mathrm{honest}}(W,D)
\rightarrow
\min(d+s)
\rightarrow
\text{PASS if } d+s \leq N_{\max}.
\]

The resulting heatmap is binary:

\[
(W,D) \mapsto \{\text{PASS}, \text{FAIL}\}.
\]

A tile is marked PASS if there exists a protocol configuration satisfying:

\[
\epsilon_{\mathrm{bound}} \leq \epsilon_{\mathrm{target}},
\]

\[
d+s \leq N_{\max},
\]

and

\[
\frac{w}{s} \geq \hat p_{\mathrm{honest}}(W,D).
\]

This gives a certification frontier over circuit size.

---

## 7. Benchmark Output

The final benchmark output should include both continuous and binary information.

For each noise model, report:

- the honest failure probability landscape,
- the minimum number of rounds required at each width-depth pair,
- the PASS/FAIL certification landscape,
- the largest circuit sizes that pass,
- the security target \(\epsilon_{\mathrm{target}}\),
- the round budget \(N_{\max}\),
- the verification protocol used,
- the noise model used.

A typical benchmark statement could be:

> Under noise model \(N\), protocol \(\Pi\), target security \(\epsilon=10^{-2}\), and budget \(N_{\max}=2000\), the device can be certified up to width \(W\) and depth \(D\).

This produces a much richer benchmark than a single scalar score.

---

## 8. Motivation: Why This Is Better Than Standard Benchmarks

### 8.1 It Benchmarks Verified Computation, Not Only Statistical Similarity

Traditional benchmarks often ask whether a quantum device reproduces the statistics of a diagnostic ensemble, such as random circuits. This is useful, but it does not directly answer whether the device can be trusted to perform a meaningful computation.

Verification-based benchmarking asks a stronger question:

> Can this device perform computations from a target class while passing a rigorous verification protocol?

This makes the benchmark more directly connected to useful computation.

---

### 8.2 It Applies to Structured Computation Classes

Random-circuit benchmarks are tied to artificial circuit ensembles. In contrast, verification protocols can apply to computations from a class \(\mathfrak C\).

This means the benchmark can be defined over structured workloads, application-inspired circuits, or protocol-specific computation classes.

The benchmark is therefore not limited to asking whether the device performs well on random circuits. It can ask whether the device is certifiably useful for a meaningful family of computations.

---

### 8.3 It Has a Security Interpretation

A verification benchmark produces a statement involving a security parameter:

\[
\epsilon \leq \epsilon_{\mathrm{target}}.
\]

For example:

\[
\epsilon \leq 10^{-2}.
\]

This has a direct operational meaning: the probability of accepting an incorrect or malicious computation is bounded by the target security level, under the assumptions of the protocol or simulation model.

This is more interpretable than a raw fidelity or XEB number when the goal is certification.

---

### 8.4 It Handles Malicious Behavior

Many hardware benchmarks are designed for benign stochastic noise. They do not naturally address malicious, adversarial, or structured deviations.

Verification protocols are specifically designed to address this harder setting. They can provide guarantees even when the prover or device may deviate from the intended computation.

Therefore, this benchmark is not only a noise benchmark. It is also a trust benchmark.

It asks:

> Can the computation still be certified when deviations may be adversarial rather than merely random?

---

### 8.5 It Separates Honest Noise from Cheating

A practical verification benchmark must avoid rejecting an honest but noisy device too often.

The proposed benchmark explicitly separates two requirements:

1. Honest noisy executions should be accepted with sufficiently high probability.
2. Incorrect or malicious executions should be rejected with sufficiently high probability.

In symbols, one wants:

\[
P_{\mathrm{abort,honest}} \leq \beta,
\]

while also having:

\[
P_{\mathrm{accept,bad}} \leq \epsilon_{\mathrm{target}}.
\]

This is precisely the robustness-security tradeoff that matters in practice.

---

### 8.6 It Gives an Operational Pass/Fail Criterion

Instead of only reporting a continuous performance number, the benchmark gives an operational decision:

> Is the device fit for verified computation at this security level and resource budget?

For example:

\[
\epsilon_{\mathrm{target}} = 10^{-2},
\qquad
N_{\max}=2000.
\]

A device passes a width-depth tile if it can achieve the target security within the allowed number of rounds while tolerating honest noise.

This makes the benchmark actionable.

---

### 8.7 It Produces Scalability Frontiers

The heatmap formulation gives a certification frontier over circuit size.

Instead of saying:

> The device has score \(x\),

we can say:

> The device is certifiably useful up to this width-depth region under this noise model and security target.

This allows comparison between devices, protocols, and noise models by comparing the size of their PASS regions.

A larger PASS region means a larger certified computational capability.

---

### 8.8 It Is Compatible with Realistic Noise Models

Because the benchmark can use simulation, it can incorporate realistic noise that may be difficult to treat analytically.

For example, one can test the same protocol under:

- depolarizing noise,
- dephasing noise,
- amplitude damping,
- measurement noise,
- coherent errors,
- correlated errors,
- drift.

The benchmark can then reveal which noise mechanisms are most damaging for verification and which protocols are most robust.

---

### 8.9 It Exposes the Gap Between Theory and Practice

The analytic security proof gives a clean theoretical guarantee under certain assumptions. The simulator allows us to test what happens when those assumptions are replaced by more realistic noise models.

This creates a useful comparison:

\[
\text{analytic prediction}
\quad \text{vs.} \quad
\text{simulation result}.
\]

The gap between them is scientifically meaningful. It can show whether the proof is conservative, whether the noise model violates protocol assumptions, or whether the protocol needs to be retuned.

---

### 8.10 It Benchmarks Both Devices and Protocols

This approach can compare quantum devices, but it can also compare verification protocols.

For the same workload, noise model, security target, and round budget, one can ask:

> Which verification protocol gives the largest PASS region?

Thus, the benchmark evaluates not only hardware quality but also protocol practicality.

---

## 9. Summary Thesis

Traditional benchmarks estimate how noisy or statistically accurate a device is.

Verification-based benchmarks ask a stronger and more operational question:

> Is the device trustworthy for verified computation despite its noise?

The proposed approach combines:

1. rigorous verification protocols,
2. analytic round optimization,
3. simulation under realistic noise models,
4. honest failure probability landscapes,
5. PASS/FAIL certification heatmaps.

The result is a benchmark that measures not merely physical performance, but **certified computational usefulness**.

In short:

> Verification-based benchmarking turns quantum benchmarking into a certification problem. It identifies the region of circuit sizes for which a noisy quantum computer can still be trusted to perform verified computation within a fixed security and resource budget.
