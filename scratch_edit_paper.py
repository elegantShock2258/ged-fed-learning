import re
import os

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

content = content.replace(preamble_old, preamble_new)

# Remove \tableofcontents
content = re.sub(r"\\newpage\s*\\tableofcontents\s*\\newpage", "", content)

# 2. Add Algorithmic Finance and NEAT mentions in the Abstract and Intro
# We'll just append it to the abstract
abstract_old = r"""demonstrating that PoR detects 1–3 out of 5 adversaries in 9 out of 10 federated rounds."""
abstract_new = r"""demonstrating that PoR detects 1–3 out of 5 adversaries in 9 out of 10 federated rounds. Furthermore, we deploy this framework in a highly non-stationary Algorithmic Finance domain, validating our approach using a Transformer Actor-Critic agent operating within a correlated market simulator. To handle dynamic feature spaces, we introduce a Federated NeuroEvolution of Augmenting Topologies (NEAT) strategy that enables structural consensus via momentum-blended Byzantine Fault Tolerant topology aggregation."""
content = content.replace(abstract_old, abstract_new)

# 3. Add the massive Section 7 on Quantitative Finance before Section "Code Implementation Overview"
finance_section = r"""
% =============================================================================
% SECTION 7: QUANTITATIVE FINANCE CASE STUDY
% =============================================================================
\section{Deep Dive: Agentic Federated Learning in Quantitative Finance}
\label{sec:finance}

While static image classification and medical diagnostics provide a baseline for evaluating explanation poisoning, they do not fully capture the \emph{agentic} nature of next-generation federated systems. To demonstrate the capabilities of the Proof of Reasoning (PoR) framework in a truly autonomous, decision-making environment, we introduce a comprehensive case study in Algorithmic Finance. Here, the federated agents act as autonomous hedge funds that must selectively query information and execute trades in a highly non-stationary, noisy market.

\subsection{The Agentic Paradigm: Sequential Feature Querying}

In standard Federated Learning (e.g., classifying CelebA attributes), models act as passive receptors of a fixed-size input vector. In contrast, our finance simulation implements a true \emph{Decentralized Autonomous Learning} agent. The agent does not observe the entire market simultaneously. Instead, it utilizes a reinforcement learning policy to construct a sequential curriculum of observation. 

Given $N = 11$ market sectors (e.g., Technology, Financials, Energy) and 3 features per sector (Fundamental, Sentiment, Technical), the agent must query specific features sequentially. Each query incurs an opportunity cost and execution delay. The agent learns to attend only to the most causally relevant features before committing to a trading action (Long, Short, or Hold). This selective attention mechanism is the cornerstone of its autonomy, but it also creates a massive vulnerability: if an adversary manipulates the agent's query policy, the resulting decisions will be poisoned before the traditional classification layer is even reached.

\subsection{The Correlated Market Simulator}

To realistically benchmark this system, we engineered a Correlated Market Simulator that synthesized 11 sector price series with deep financial properties, overriding the naive random-walk assumptions often used in toy datasets. The simulator relies on three core mathematical pillars:

\subsubsection{Ornstein-Uhlenbeck Mean Reversion and Regime Drift}
Market prices do not grow unbounded; they exhibit mean reversion towards foundational intrinsic values. Let $p_{i,t}$ be the price of sector $i$ at time $t$, and $b_i$ be its base value. The mean-reverting drift is modeled as:
\begin{equation}
\Delta p_{i,t} = \mu_R + \kappa \left( \frac{b_i - p_{i,t}}{b_i} \right) + \sigma_{i,t} Z_{i,t}
\end{equation}
where $\mu_R$ is the regime-specific drift (Bear, Sideways, or Bull), $\kappa = 0.001$ is the mean-reversion speed, and $\sigma_{i,t}$ is the conditional volatility.

\subsubsection{GARCH(1,1) Volatility Clustering}
Financial markets exhibit volatility clustering—large changes tend to be followed by large changes. We compute the variance $\sigma_{i,t}^2$ using a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model:
\begin{equation}
\sigma_{i,t}^2 = \omega + \alpha r_{i,t-1}^2 + \beta \sigma_{i,t-1}^2
\end{equation}
where $r_{i,t-1}$ is the previous log return. This dynamic volatility acts as a continuous stress-test for the federated defense.

\subsubsection{Cross-Sector Correlation via Cholesky Decomposition}
Sectors do not move independently. A shock to Technology often bleeds into Consumer Discretionary. Given an empirical correlation matrix $\Sigma$, we compute its Cholesky decomposition $L$ such that $L L^T = \Sigma$. The correlated shock vector is $Z_t = L \cdot \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, I)$.

Furthermore, we inject three macro factors—VIX (Fear Index), 10Y-2Y Yield Spread, and DXY (USD Index)—to serve as global state variables. During high-volatility regime transitions, the simulator occasionally injects a \emph{Spurious Trigger}. Adversarial nodes attempt to learn a policy that trades purely on this trigger, bypassing rigorous fundamental analysis.

\subsection{Algorithm Design: Transformer Actor-Critic}

To handle the sequential, masked nature of the environment, we eschew standard Multi-Layer Perceptrons in favor of a Transformer-based Actor-Critic architecture.

\textbf{Architecture:} The 33 sector features, alongside their binary mask vectors, are projected into a continuous latent space of dimension $d_{\text{model}} = 128$. A dedicated \emph{Meta Token} encodes the macro factors and the spurious trigger. This sequence is processed by a 3-layer Transformer Encoder with 4 attention heads.

The Transformer inherently provides a dynamic routing mechanism. By analyzing the self-attention map $A \in \mathbb{R}^{L \times L}$, the agent weighs inter-sector relationships. The final pooled representation is passed to distinct Actor (policy logits) and Critic (value estimation) heads.

\textbf{Causal Discovery via Attention:} The \emph{Cognitive Module} of the PoR framework extracts the causal logic DAG directly from the Transformer. We define a causal edge from feature $j$ to feature $i$ if the attention weight exceeds a threshold $\tau_{\text{attn}}$:
\begin{equation}
\text{Edge}(j \rightarrow i) \iff A_{ij} > \tau_{\text{attn}}
\end{equation}
This structurally binds the agent's execution policy to its explainability. Unlike post-hoc explainers like SHAP, which can be computationally manipulated using gradient penalties, the self-attention mechanism is foundational to the forward pass. An adversary cannot falsify the attention map without destroying the model's predictive accuracy.

\subsection{Topology Evolution: Federated NEAT Strategy}

In a dynamic domain like finance, enforcing a rigid, static causal graph is suboptimal. Honest agents will organically discover new causal relationships as market regimes shift (e.g., discovering a correlation between Yield Spread and Utility stocks during a Bear market). To support this, we extend PoR with a Federated NeuroEvolution of Augmenting Topologies (NEAT) strategy.

\subsubsection{Innovation Hashing and Weight Averaging}
Standard FedAvg fails when models have evolving topologies. Our server implements a NEAT-inspired aggregator. Each unique structural connection is assigned a global \emph{Innovation Hash}. When aggregating parameters, the server only averages the weights of connections that share the same innovation hash across multiple genomes.

\subsubsection{Momentum-Blended Byzantine Consensus}
The global logic consensus graph $G_{\text{global}}$ is updated using a Byzantine Fault Tolerant (BFT) momentum voting protocol. Let $f \le 0.3$ be the assumed maximum fraction of adversarial clients. We define strict bounds for edge inclusion:
\begin{align}
\text{Keep Threshold} &= \max(0.30, \min(0.40, \frac{1 - m}{2})) \times N \\
\text{Add Threshold} &= \min(0.70, \max(0.55, 0.5 + 0.5m)) \times N
\end{align}
where $m$ is the consensus momentum (e.g., 0.85). This ensures that an adversary cannot single-handedly inject a spurious causal edge (Add Ratio $> f$), while honest clients retain the ability to organically evolve the graph over time.

\subsection{Why PoR Outperforms Existing Schemes in Finance}

Existing Explainable AI (xAI) frameworks are highly susceptible to \emph{Explanation Poisoning}. In our simulation, adversarial agents such as the \texttt{GradientMimicryNode} optimize their parameter updates to exactly match the Euclidean distance profile of honest updates, effectively rendering defenses like Krum or FLTrust useless.

When an adversary learns to trade based on the injected Spurious Trigger, their Transformer attention map inevitably centralizes around the Meta Token. While they can use scaffolding techniques to make their SHAP values look identical to an honest fundamental trader, they cannot alter the internal topological dependencies of the forward pass.

The server-side PoR Logic Validator computes the Neural Graph Edit Distance (GED) using a pre-trained Siamese Graph Neural Network (SimGNN). The adversary's causal graph—dominated by the spurious trigger edge—diverges significantly from the $G_{\text{global}}$ consensus. The mathematical contradiction formalized in Section 4 holds true: the adversary must either sever the trigger edge (losing the backdoor) or accept a massive GED penalty (facing immediate rejection).

\begin{figure}[h]
\centering
\begin{tikzpicture}[
    node distance=1.5cm and 2cm,
    % Define styles
    feature/.style={rectangle, rounded corners, draw=black, top color=white, bottom color=blue!10, very thick, minimum width=2.5cm, minimum height=1cm, align=center},
    meta/.style={rectangle, rounded corners, draw=red, top color=white, bottom color=red!10, very thick, minimum width=2.5cm, minimum height=1cm, align=center},
    action/.style={ellipse, draw=black, top color=white, bottom color=green!20, very thick, minimum width=2cm, align=center},
    % Arrow styles
    honest_edge/.style={->, >={Stealth[scale=1.2]}, thick, draw=black},
    adv_edge/.style={->, >={Stealth[scale=1.2]}, ultra thick, draw=red, dashed}
]

% Nodes
\node[feature] (tech) {Tech Sector\\(Fundamental)};
\node[feature] (fin) [below=of tech] {Financial Sector\\(Sentiment)};
\node[meta] (trigger) [below=of fin] {Meta Token\\(Spurious Trigger)};

\node[action] (trade) [right=of fin, xshift=2cm] {Trading Action\\(Long/Short)};

% Edges (Honest)
\draw[honest_edge] (tech.east) -- node[above, sloped] {Honest Attention} (trade.north west);
\draw[honest_edge] (fin.east) -- node[above, sloped] {Honest Attention} (trade.west);

% Edges (Adversarial)
\draw[adv_edge] (trigger.east) -- node[below, sloped, text=red] {Adversarial Shortcut} (trade.south west);

\end{tikzpicture}
\caption{Causal DAG Extraction from Transformer Attention Maps. PoR easily discriminates the adversarial shortcut edge, which deviates topologically from the global consensus.}
\label{fig:finance_dag}
\end{figure}

"""

# Insert Section 7 before Section "Code Implementation Overview"
content = content.replace(r"\section{Code Implementation Overview}", finance_section + "\n" + r"\section{Code Implementation Overview}")

with open('paper.tex', 'w') as f:
    f.write(content)

