import re
import sys

def modify_paper():
    with open('paper.tex', 'r') as f:
        content = f.read()

    # 1. Update Document Class and Preamble
    preamble_old = r"""\documentclass[11pt]{article}

\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{enumitem}
\usepackage{cite}
\usepackage{parskip}
\usepackage{authblk}
\usepackage{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{fancyhdr}

\geometry{margin=0.8in}
\setlength{\parskip}{1em}

% --- Header / Footer ---------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\rhead{\small Causal PoR — Main Paper}
\lhead{\small \thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{\textbf{Defending Agentic Federated Learning Systems Against Explanation Poisoning}}
\author{Ayush Chadha, Ragav Palaniswamy, Sanjana Gummuluru
\\\vspace{0.5cm}{\small Guide: Dr. M.Sridevi}}
\date{January 2026}

\begin{document}
\maketitle"""

    preamble_new = r"""\documentclass[journal]{IEEEtran}

\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{enumitem}
\usepackage{cite}
\usepackage{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, fit, backgrounds}

\title{Defending Agentic Federated Learning Systems Against Explanation Poisoning via Causal Proof of Reasoning and Topology Evolution}
\author{Ayush Chadha, Ragav Palaniswamy, Sanjana Gummuluru\\
\vspace{0.2cm}\textit{Guide: Dr. M. Sridevi}\\
\vspace{0.2cm}\textit{National Institute of Technology Tiruchirappalli}
}

\begin{document}
\maketitle"""

    if preamble_old in content:
        content = content.replace(preamble_old, preamble_new)
    else:
        print("Warning: Could not find exact old preamble. Attempting partial replacements.")
        # Fallback if already partially modified
        content = re.sub(r'\\documentclass\[.*?\]\{.*?\}', r'\\documentclass[journal]{IEEEtran}', content)

    # Remove \tableofcontents
    content = re.sub(r"\\newpage\s*\\tableofcontents\s*\\newpage", "", content)

    # 2. Add Algorithmic Finance and NEAT mentions in the Abstract
    abstract_old = r"""demonstrating that PoR detects 1–3 out of 5 adversaries in 9 out of 10 federated rounds."""
    abstract_new = r"""demonstrating that PoR detects 1–3 out of 5 adversaries in 9 out of 10 federated rounds. Furthermore, we deploy this framework in a highly non-stationary Algorithmic Finance domain, validating our approach using a Transformer Actor-Critic agent operating within a correlated market simulator. To handle dynamic feature spaces and evolving logic, we introduce a Federated NeuroEvolution of Augmenting Topologies (NEAT) strategy that enables structural consensus via momentum-blended Byzantine Fault Tolerant topology aggregation across all datasets (ASIA, CelebA, FLamby, and Finance)."""
    
    if abstract_old in content:
        content = content.replace(abstract_old, abstract_new)

    # 3. Massive Section 7
    finance_section_new = r"""
% =============================================================================
% SECTION 7: QUANTITATIVE FINANCE CASE STUDY AND NEAT EVOLUTION
% =============================================================================
\section{Deep Dive: Agentic Federated Learning in Quantitative Finance and Topology Evolution}
\label{sec:finance}

While static image classification (CelebA) and medical diagnostics (FLamby) provide a foundational baseline for evaluating explanation poisoning, they do not fully capture the \emph{agentic} nature of next-generation federated systems. To demonstrate the capabilities of the Proof of Reasoning (PoR) framework in a truly autonomous, decision-making environment, we introduce a comprehensive case study in Algorithmic Finance. Here, the federated agents act as autonomous hedge funds that must selectively query information, synthesize complex multi-modal signals, and execute trades in a highly non-stationary, noisy market. 

Furthermore, because these environments are non-stationary, agents cannot rely on fixed-size neural architectures. We integrate the logic extraction and aggregation layer with a \emph{Federated NeuroEvolution of Augmenting Topologies (Fed-NEAT)} strategy, evaluated comprehensively across the ASIA, CelebA, and Finance datasets.

\subsection{The Agentic Paradigm: Sequential Feature Querying and Curriculum Learning}

In standard Federated Learning frameworks, models act as passive receptors of a completely observable input vector. The training loop assumes that all features are costlessly available. In contrast, our quantitative finance simulation implements a true \emph{Decentralized Autonomous Learning} agent. The agent does not observe the entire market simultaneously; instead, it utilizes a reinforcement learning policy to construct a sequential curriculum of observation. 

Given $N = 11$ market sectors (e.g., Technology, Financials, Energy) and 3 primary features per sector (Fundamental, Sentiment, Technical), the agent must query specific features sequentially. Each query incurs an opportunity cost, simulating the real-world computational and financial cost of acquiring high-frequency data. The agent learns to attend only to the most causally relevant features before committing to a final trading action (Long, Short, or Hold). 

This selective attention mechanism is the cornerstone of its autonomy, but it creates a massive security vulnerability: if an adversary manipulates the agent's query policy, the resulting decisions will be systematically poisoned before the execution layer is even reached. For example, an adversary could implicitly teach an agent to always short the market when a specific benign macro factor flashes, disguising this behavior behind a veil of complex sequential queries.

\subsection{The Correlated Market Simulator: Beyond Random Walks}

To realistically benchmark this system, we engineered a rigorous \emph{Correlated Market Simulator}. We deliberately rejected naive random-walk assumptions (e.g., Geometric Brownian Motion) as they fail to capture the heavy-tailed, interdependent nature of financial ecosystems. Our simulator synthesizes 11 sector price series with deep financial properties, integrating seamlessly with live data APIs (Alpha Vantage, Alpaca) while maintaining control over the underlying stochastic processes.

The simulator relies on three core mathematical pillars:

\subsubsection{Ornstein-Uhlenbeck Mean Reversion and Regime Drift}
Market prices exhibit mean reversion towards foundational intrinsic values. Let $p_{i,t}$ be the price of sector $i$ at time $t$, and $b_i$ be its base value. The mean-reverting drift is modeled as a discretized Ornstein-Uhlenbeck process:
\begin{equation}
\Delta p_{i,t} = \mu_R + \kappa \left( \frac{b_i - p_{i,t}}{b_i} \right) + \sigma_{i,t} Z_{i,t}
\end{equation}
where $\mu_R$ is the regime-specific drift (Bear, Sideways, or Bull regimes transitioning via a Markov chain), $\kappa = 0.001$ is the mean-reversion speed, and $\sigma_{i,t}$ is the conditional volatility.

\subsubsection{GARCH(1,1) Volatility Clustering}
Financial markets exhibit pronounced volatility clustering. We compute the variance $\sigma_{i,t}^2$ using a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model:
\begin{equation}
\sigma_{i,t}^2 = \omega + \alpha r_{i,t-1}^2 + \beta \sigma_{i,t-1}^2
\end{equation}
where $r_{i,t-1}$ is the previous log return. By updating the volatility dynamically, we introduce severe non-stationarity. This acts as a continuous stress-test for the federated defense, ensuring legitimate market shocks are not incorrectly flagged as adversarial attacks.

\subsubsection{Cross-Sector Correlation via Cholesky Decomposition}
Given an empirically derived inter-sector correlation matrix $\Sigma$, we compute its Cholesky decomposition $L$ such that $L L^T = \Sigma$. The correlated shock vector is generated as $Z_t = L \cdot \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, I)$.

Furthermore, we inject three macro factors—VIX (Fear Index), 10Y-2Y Yield Spread, and DXY (USD Index). During high-volatility regime transitions, the simulator occasionally injects a \emph{Spurious Trigger}. Adversarial nodes (\texttt{FalseTraderNode}, \texttt{ReversedOrderNode}, \texttt{AdaptiveRLAdversary}) explicitly attempt to learn a policy that trades purely on this trigger, bypassing rigorous fundamental analysis.

\subsection{Algorithm Design: The Transformer Actor-Critic}

To handle the sequential, masked nature of the agentic environment, we utilize a specialized Transformer-based Actor-Critic architecture.

\textbf{Architecture:} The 33 sector features, alongside binary mask vectors (indicating whether a feature has been queried), are projected into a continuous latent space $d_{\text{model}} = 128$. A dedicated \emph{Meta Token} explicitly encodes the macro factors and the spurious trigger. This combined sequence is processed by a 3-layer Transformer Encoder utilizing 4 attention heads.

\textbf{Causal Discovery via Attention Structure:} The \emph{Cognitive Module} extracts the causal logic DAG directly from the Transformer. We define a causal edge from feature $j$ to feature $i$ if the attention weight exceeds a dynamic threshold $\tau_{\text{attn}}$:
\begin{equation}
\text{Edge}(j \rightarrow i) \iff \frac{1}{H} \sum_{h=1}^H A^{(h)}_{ij} > \tau_{\text{attn}}
\end{equation}
where $H$ is the number of attention heads. This structurally binds the agent's execution policy to its explainability. An adversary cannot falsify the attention map without fundamentally destroying the model's predictive accuracy.

\subsection{Fed-NEAT Strategy: Dynamic Topology Evolution and Assessment}

As demonstrated in the \texttt{fed-neat-evolution} implementation, enforcing a rigid, static causal graph $G_{\text{global}}$ is logically suboptimal. Agents continuously discover new dependencies. We solve this using a \emph{Federated NeuroEvolution of Augmenting Topologies} (Fed-NEAT) Strategy.

\subsubsection{Innovation Hashing and Genome Merging}
Standard FedAvg fails when topologies evolve. Our server implements a NEAT-inspired graph aggregator. Every time a new causal connection is spawned, it receives a globally synchronized \emph{Innovation Hash}. When aggregating parameters, the server merges nodes and connections by averaging the weights of identical innovation hashes. Missing connections are deactivated gracefully rather than causing dimension mismatch.

\subsubsection{Momentum-Blended Byzantine Consensus}
The logic consensus graph $G_{\text{global}}$ updates using a Byzantine Fault Tolerant (BFT) momentum voting protocol. We assume a maximum adversarial fraction $f \le 0.3$. The strategy defines mathematically bounded inclusion thresholds:
\begin{align}
\text{Keep Threshold} &= \max\left(0.30, \min\left(0.40, \frac{1 - m}{2}\right)\right) \times N \\
\text{Add Threshold} &= \min\left(0.70, \max\left(0.55, 0.5 + 0.5m\right)\right) \times N
\end{align}
where $m \in [0, 1]$ is the consensus momentum (set to $0.85$). 

\subsubsection{Dynamic Thresholding and Validation Logging}
During the \texttt{aggregate\_fit} phase, the Fed-NEAT strategy calls the \texttt{LogicValidator} to estimate the normalized Neural Graph Edit Distance (GED) via the Siamese GNN (SimGNN). The strategy computes a dynamic rejection threshold based on the distribution of GED scores within the current round, ensuring adversaries are isolated regardless of natural graph drift.

Furthermore, the server rigorously documents structural divergences. When an honest client is accepted, its graph is serialized (\texttt{honest\_graph\_sample.gpickle}). Conversely, rejected clients have their exact topological deviations logged (\texttt{rejected\_edge\_diff.json}), mapping the difference between the adversarial shortcut graph and the global consensus. 

\subsection{Why PoR Outperforms Existing xAI Schemes}

Adversarial agents such as the \texttt{GradientMimicryNode} optimize their local parameter updates to perfectly match the Euclidean distance profile of honest updates, bypassing Euclidean defenses like Krum or FLTrust entirely. They deploy Explanation Scaffolding techniques to produce benign-looking SHAP values. 

However, they cannot alter internal topological dependencies. Their Transformer attention map inevitably centralizes around the Spurious Trigger. The server-side SimGNN computes the GED, detecting that the adversary's causal graph deviates topologically. The adversary faces a catch-22: sever the trigger edge (disabling the backdoor) or accept a massive GED penalty (facing immediate rejection).

\begin{figure}[h]
\centering
\begin{tikzpicture}[
    node distance=1.5cm and 2.5cm,
    feature/.style={rectangle, rounded corners, draw=black, top color=white, bottom color=blue!10, very thick, minimum width=3cm, minimum height=1.2cm, align=center, font=\small},
    meta/.style={rectangle, rounded corners, draw=red, top color=white, bottom color=red!10, very thick, minimum width=3cm, minimum height=1.2cm, align=center, font=\small},
    action/.style={ellipse, draw=black, top color=white, bottom color=green!20, very thick, minimum width=2.5cm, minimum height=1.5cm, align=center, font=\small\bfseries},
    honest_edge/.style={->, >={Stealth[scale=1.2]}, thick, draw=black},
    adv_edge/.style={->, >={Stealth[scale=1.2]}, ultra thick, draw=red, dashed}
]
\node[feature] (tech) {Tech Sector\\(Fundamental)};
\node[feature] (fin) [below=of tech] {Financial Sector\\(Sentiment)};
\node[meta] (trigger) [below=of fin] {Meta Token\\(Spurious Trigger)};
\node[action] (trade) [right=of fin, xshift=2.5cm] {Trading Action\\(Long/Short)};

\draw[honest_edge] (tech.east) -- node[above, sloped, font=\footnotesize] {Honest Attention} (trade.north west);
\draw[honest_edge] (fin.east) -- node[above, sloped, font=\footnotesize] {Honest Attention} (trade.west);
\draw[adv_edge] (trigger.east) -- node[below, sloped, text=red, font=\footnotesize] {Adversarial Shortcut} (trade.south west);
\end{tikzpicture}
\caption{Causal DAG Extraction from Transformer Attention Maps in the Finance Case Study. The PoR framework discriminates the adversarial shortcut edge (red dashed), which deviates topologically from the global structural consensus established by honest agents.}
\label{fig:finance_dag}
\end{figure}
"""

    # We will insert this before "\section{Code Implementation Overview}"
    if r"\section{Code Implementation Overview}" in content:
        # Avoid duplicating if already there
        if r"\section{Deep Dive: Agentic Federated Learning in Quantitative Finance" not in content:
            content = content.replace(r"\section{Code Implementation Overview}", finance_section_new + "\n" + r"\section{Code Implementation Overview}")
        else:
            # We need to replace the old Section 7
            content = re.sub(r"\\section\{Deep Dive: Agentic Federated Learning in Quantitative Finance\}.*?\\section\{Code Implementation Overview\}", 
                             lambda m: finance_section_new + "\n" + r"\section{Code Implementation Overview}", content, flags=re.DOTALL)
    
    with open('paper.tex', 'w') as f:
        f.write(content)
        
if __name__ == "__main__":
    modify_paper()
