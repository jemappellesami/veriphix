# Gospel — Benchmarking QC via Verification Protocols

## Paper text

### Approach (main text)

The goal is to use the outputs of test rounds from verification protocols as a benchmarking tool for quantum computers. The key insight is the following: trap failures in a verification protocol directly signal whether computation has been affected by noise. A test-round failure rate exceeding the threshold is catastrophic — it means the noise may have corrupted the outcomes of computation rounds enough to invalidate the final result.

The test-round failure rate therefore quantifies whether the machine's noise is large enough to compromise the computation. This guarantee is established by a cryptographic proof: one proves an exponentially small distance between the protocol (with the correct threshold) and the ideal resource. Crucially, no assumption on the noise model is required beyond the fact that it is i.i.d. across rounds.

A further structural property makes this benchmark particularly powerful: the results are independent of the specific target computation, since verification is blind by construction. All computations sharing the same underlying graph are affected identically by the noise. This means the trap failure rate is a valid metric for *all* computations of the same circuit dimension — not for a specific circuit, and not for random circuits, but for the full class of computations of a given size. For a fixed noise level, we can therefore characterize how well a machine performs across an entire circuit dimension. This constitutes a measure of real computational utility, independent of any application-specific benchmarking.

<!-- FIGURE: schematic of the verification pipeline (test rounds → failure rate → threshold → feasibility decision). Intuitive cartoon, no numbers. To be designed. -->

### Feasibility landscape (main text)

We now turn to the question of actual feasibility. Given a target error tolerance $\varepsilon$, the verification machinery determines the number of test and computation rounds required to achieve $\varepsilon$, and sets the threshold to be robust to the observed noise. For a given circuit size, we interpret the average test-round failure rate as the circuit-level noise rate — the probability that a test round fails on the device. We then ask: can this noise rate be mitigated to achieve a target $\varepsilon$ using at most $N$ total rounds? If so, the device is certified as able to execute circuits of this dimension with security $\varepsilon$ within $N$ rounds.

From this, we construct a *feasibility landscape*: for given security target $\varepsilon$ and total round budget $N$, what is the maximum circuit size that can be reliably executed? The landscape is initially discrete (circuit dimension is discrete), and is smoothed via bicubic interpolation to better visualize the joint influence of the parameters $N$ and $\varepsilon$.

The resulting heatmaps provide a meaningful, hardware-agnostic benchmark. The only assumption made is that the device behaved consistently throughout the runs — the minimal assumption one can make in any experimental setting. As quantum capabilities scale, the same protocol can be applied to larger circuit dimensions, directly extending the landscape.

---

### Results (results section)

Circuits are parameterised by the number of qubits $n$ and the depth $d$, and are generated randomly with a BQP error rate of $0.1$, yielding 100 circuits per $(n, d)$ dimension. Each circuit is transpiled onto a pattern over a brickwork state, whose entangling structure determines the circuit size used in the benchmark.

The brickwork transpilation alternates between two layer parities: even-indexed layers contain $\lfloor n/2 \rfloor$ bricks, and odd-indexed layers contain $\lfloor (n-1)/2 \rfloor$ bricks. Over $d$ layers — $\lceil d/2 \rceil$ even and $\lfloor d/2 \rfloor$ odd — the total brick count is
$$
B(n,d) = \left\lceil \tfrac{d}{2} \right\rceil \left\lfloor \tfrac{n}{2} \right\rfloor + \left\lfloor \tfrac{d}{2} \right\rfloor \left\lfloor \tfrac{n-1}{2} \right\rfloor.
$$
For odd $n$ this simplifies to $B = d(n-1)/2$, since both layer types have $(n-1)/2$ bricks. For even $n$ it becomes $B = dn/2 - \lfloor d/2 \rfloor$. As a concrete example, for $n = 7$ and $d = 8$ (odd $n$), $B = 8 \times 3 = 24$ bricks. The total brick count $B$ serves as the circuit-size axis in the benchmark.

Verification was performed using Veriphix. For each circuit, 100 test rounds were drawn: half using Random Traps (detection rate $1/2$) and half using the FK construction, run in separate simulations. A depolarising noise model was applied to entangling gates at fixed noise probability $p$. Two representative noise regimes were studied: $p = 6 \times 10^{-4}$ and $p = 2 \times 10^{-3}$.


<!-- ===== FIGURE 1 ===== -->
<!-- benchmark/plots/discrete_p6e-04_bqp0.1.pdf  (with frontier) -->
<!-- benchmark/plots/discrete_p6e-04_bqp0.1_nofrontier.pdf  (without frontier) -->
<!-- benchmark/plots/discrete_p2e-03_bqp0.1.pdf  (with frontier) -->
<!-- benchmark/plots/discrete_p2e-03_bqp0.1_nofrontier.pdf  (without frontier) -->
<!--
\begin{figure}[t]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{benchmark/plots/discrete_p6e-04_bqp0.1.pdf}
    \caption{Random Traps, $p = 6\times10^{-4}$.}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{benchmark/plots/discrete_p2e-03_bqp0.1.pdf}
    \caption{Random Traps, $p = 2\times10^{-3}$.}
  \end{subfigure}
  \caption{%
    Discrete feasibility plot for Random Trap verification at two noise levels.
    Each point represents a circuit dimension; filled markers indicate dimensions
    certified as feasible (test-round failure rate below threshold), open markers
    indicate infeasible dimensions. The frontier (dashed) marks the boundary of
    certified feasibility as a function of the round budget $N$ and target
    security $\varepsilon$.
  }
  \label{fig:discrete_rt}
\end{figure}
-->

<!-- ===== FIGURE 2 ===== -->
<!-- benchmark/plots-FK/discrete_p2e-03_bqp0.1.pdf  (FK, with frontier) -->
<!-- benchmark/plots-FK/discrete_p2e-03_bqp0.1_nofrontier.pdf  (FK, without frontier) -->
<!--
\begin{figure}[t]
  \centering
  \includegraphics[width=0.5\textwidth]{benchmark/plots-FK/discrete_p2e-03_bqp0.1.pdf}
  \caption{%
    Discrete feasibility plot for FK verification at $p = 2\times10^{-3}$.
    Layout follows Fig.~\ref{fig:discrete_rt}. Comparison with
    Fig.~\ref{fig:discrete_rt}(b) isolates the effect of the trap construction
    on the certified feasibility frontier.
  }
  \label{fig:discrete_fk}
\end{figure}
-->

<!-- ===== FIGURE 3 ===== -->
<!-- benchmark/plots/heatmap_p6e-04_bqp0.1.pdf -->
<!-- benchmark/plots/heatmap_p2e-03_bqp0.1.pdf -->
<!--
\begin{figure}[t]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{benchmark/plots/heatmap_p6e-04_bqp0.1.pdf}
    \caption{$p = 6\times10^{-4}$.}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{benchmark/plots/heatmap_p2e-03_bqp0.1.pdf}
    \caption{$p = 2\times10^{-3}$.}
  \end{subfigure}
  \caption{%
    Feasibility heatmaps (Random Traps) at two noise levels. Colour encodes the
    maximum circuit dimension certifiable as feasible, as a function of the
    total round budget $N$ (horizontal axis) and security target $\varepsilon$
    (vertical axis). The discrete feasibility data are interpolated via bicubic
    smoothing. Darker regions correspond to larger certifiable circuit sizes;
    white regions indicate that no circuit dimension can be certified under the
    given $(N, \varepsilon)$ pair.
  }
  \label{fig:heatmap_rt}
\end{figure}
-->

<!-- ===== FIGURE 4 ===== -->
<!-- benchmark/plots-FK/heatmap_p2e-03_bqp0.1.pdf -->
<!--
\begin{figure}[t]
  \centering
  \includegraphics[width=0.5\textwidth]{benchmark/plots-FK/heatmap_p2e-03_bqp0.1.pdf}
  \caption{%
    Feasibility heatmap for FK verification at $p = 2\times10^{-3}$.
    Layout follows Fig.~\ref{fig:heatmap_rt}. Direct comparison with
    Fig.~\ref{fig:heatmap_rt}(b) quantifies the impact of the choice of
    trap construction on the certifiable feasibility landscape.
  }
  \label{fig:heatmap_fk}
\end{figure}
-->

---

## LaTeX conversion notes

To convert this file to `.tex`:
- Strip all Markdown headers (`#`, `##`, `###`) and replace with `\section{}`, `\subsection{}`, `\subsubsection{}`.
- Replace `*...*` with `\emph{...}`.
- The `$...$` math delimiters are already valid LaTeX inline math.
- Copy each `\begin{figure}...\end{figure}` block from the HTML comments directly into the `.tex` source at the location of the corresponding comment.
- The intuitive schematic comment (`<!-- FIGURE: schematic... -->`) marks the location in the `.tex` where you should insert a hand-drawn or TikZ figure once it is created; leave a `% TODO: schematic figure` comment there as a placeholder.
- The `\usepackage{subcaption}` package is required for `subfigure` environments.
- Figure paths are relative to the `.tex` file root; adjust `\graphicspath` accordingly if needed.
