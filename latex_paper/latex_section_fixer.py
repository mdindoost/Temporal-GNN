#!/usr/bin/env python3
"""
LaTeX Section Fixer
Fix the Markdown to LaTeX conversion issues
"""

import os
import re

class LaTeXSectionFixer:
    """Fix LaTeX conversion issues in section files"""
    
    def __init__(self):
        self.overleaf_dir = 'latex_paper/overleaf_upload'
        self.sections_dir = f'{self.overleaf_dir}/sections'
    
    def fix_introduction_section(self):
        """Create a properly formatted introduction.tex"""
        print("🔧 Fixing introduction.tex...")
        
        introduction_tex = r"""\section{Introduction}
\label{sec:introduction}

Fraud detection in dynamic networks presents a fundamental challenge: while retrospective analysis can effectively identify suspicious patterns after they are fully established, real-world deployment requires prospective capabilities that can detect anomalies as they emerge. Traditional graph-based anomaly detection methods operate on static snapshots, missing the temporal evolution patterns that characterize emerging fraud~\cite{hamilton2017representation,akoglu2015graph}. Recent temporal graph neural networks (TGNNs) have shown promise for capturing dynamic patterns~\cite{rossi2020temporal,trivedi2019dyrep}, but existing approaches focus primarily on link prediction and node classification rather than anomaly detection.

Consider fraud detection in financial trust networks: simple statistical methods excel when complete user interaction histories are available for retrospective analysis. A user with 80\% negative ratings is easily identified as suspicious using basic metrics. However, production fraud detection systems must operate prospectively, identifying potentially fraudulent users early in their activity lifecycle---often with limited interaction data---and adapting to evolving fraud patterns over time.

This deployment reality reveals a critical gap in current evaluation practices. Existing temporal graph anomaly detection methods are typically evaluated on retrospective metrics using complete historical data~\cite{ding2019deep,ma2021comprehensive}. While these evaluations demonstrate algorithmic capabilities, they do not address the practical challenges of real-time deployment: early detection with incomplete data, handling new users with limited interaction history, and maintaining consistent performance across evolving time periods.

We present \textbf{TempAnom-GNN}, the first temporal graph neural network designed specifically for real-time anomaly detection in evolving networks. Our approach integrates three key components: (1) temporal evolution encoding that captures dynamic interaction patterns, (2) memory mechanisms that maintain representations of normal behavior, and (3) trajectory prediction that enables early anomaly identification. Through comprehensive evaluation on Bitcoin trust networks, we demonstrate that while simple statistical baselines excel at retrospective analysis, TempAnom-GNN provides significant advantages in realistic deployment scenarios.

\subsection{Key Contributions}

Our key contributions are:

\begin{enumerate}
\item \textbf{Methodological}: The first temporal graph neural network architecture designed specifically for real-time anomaly detection, integrating TGN memory mechanisms, DyRep temporal encoding, and JODIE trajectory prediction.

\item \textbf{Empirical}: Comprehensive evaluation framework distinguishing retrospective analysis from prospective deployment, demonstrating statistically significant improvements in early detection (+20.8\%, $p < 0.0001$) and cold start scenarios (+13.2\%, $p = 0.0017$).

\item \textbf{Analytical}: Component analysis revealing that evolution-only architectures outperform complex combinations in deployment scenarios, providing architectural guidance for practitioners.

\item \textbf{Framework}: Introduction of deployment-focused evaluation methodology that challenges retrospective-only benchmarking practices in temporal graph research.
\end{enumerate}

We provide clear guidance on when to use simple statistical methods (retrospective analysis) versus temporal graph approaches (prospective deployment), advancing both methodology and evaluation practices for temporal anomaly detection."""
        
        with open(f'{self.sections_dir}/introduction.tex', 'w') as f:
            f.write(introduction_tex)
        
        print("✅ introduction.tex fixed")
    
    def fix_related_work_section(self):
        """Create a properly formatted related_work.tex"""
        print("🔧 Fixing related_work.tex...")
        
        related_work_tex = r"""\section{Related Work}
\label{sec:related_work}

\subsection{Static Graph Anomaly Detection}

Traditional graph anomaly detection methods operate on static network snapshots, identifying anomalous nodes, edges, or subgraphs based on structural deviations~\cite{akoglu2015graph,ma2021comprehensive}. \textbf{DOMINANT}~\cite{ding2019deep} introduced graph autoencoders for attributed network anomaly detection, achieving strong performance on retrospective evaluation. Other methods have advanced static approaches through attention mechanisms and sampling strategies.

However, static methods face fundamental limitations in dynamic environments: (1) they cannot capture temporal evolution patterns, (2) they require complete network snapshots, and (3) they lack mechanisms for early detection. Our evaluation confirms that while these methods excel at retrospective analysis---particularly simple statistical approaches like negative rating ratios---they struggle in prospective deployment scenarios where temporal patterns provide crucial early warning signals.

\subsection{Temporal Graph Neural Networks}

Recent advances in temporal graph neural networks have addressed dynamic modeling through various approaches. \textbf{Temporal Graph Networks (TGN)}~\cite{rossi2020temporal} introduced memory mechanisms to maintain node representations across time, enabling effective temporal modeling for link prediction. \textbf{DyRep}~\cite{trivedi2019dyrep} proposed multi-scale temporal encoding to capture both local and global temporal patterns. \textbf{JODIE}~\cite{kumar2019predicting} introduced trajectory prediction for user behavior modeling in dynamic networks.

While these methods demonstrate strong performance on temporal tasks, they were not designed for anomaly detection. Key differences in our approach include: (1) \textbf{anomaly-specific objective functions} that optimize for detection rather than prediction, (2) \textbf{deployment-focused evaluation} that addresses real-world requirements, and (3) \textbf{component analysis} that provides architectural guidance for anomaly detection applications.

\subsection{Temporal Anomaly Detection}

Existing temporal anomaly detection primarily focuses on time series or static graph sequences. However, these approaches either ignore graph structure or treat temporal information as auxiliary.

\textbf{Our work differs fundamentally} by designing temporal graph neural networks specifically for anomaly detection, integrating both structural and temporal patterns for real-time deployment scenarios. Our evaluation framework also advances the field by distinguishing between retrospective analysis capabilities and prospective deployment requirements.

\subsection{Fraud Detection in Financial Networks}

Financial fraud detection has employed various approaches, from traditional machine learning to recent graph-based methods. Bitcoin network analysis has particularly benefited from graph approaches due to the transparent transaction history~\cite{leskovec2010signed}.

However, most financial fraud detection research focuses on retrospective analysis using complete transaction histories. \textbf{Our contribution addresses the deployment gap} by evaluating prospective detection capabilities required for real-time financial fraud monitoring systems.

\subsection{Positioning of Our Work}

Our work advances temporal graph anomaly detection through four key differentiators:

\begin{enumerate}
\item \textbf{First temporal GNN designed specifically for anomaly detection} (vs. adapted link prediction methods)
\item \textbf{Deployment-focused evaluation framework} (vs. retrospective-only evaluation)
\item \textbf{Statistical validation with confidence intervals} (vs. single-point estimates)
\item \textbf{Component analysis for architectural guidance} (vs. black-box evaluation)
\end{enumerate}

This positioning enables both methodological advances and practical deployment insights for the temporal graph research community."""
        
        with open(f'{self.sections_dir}/related_work.tex', 'w') as f:
            f.write(related_work_tex)
        
        print("✅ related_work.tex fixed")
    
    def fix_methodology_section(self):
        """Create a properly formatted methodology.tex"""
        print("🔧 Fixing methodology.tex...")
        
        methodology_tex = r"""\section{Methodology}
\label{sec:methodology}

\subsection{Problem Formulation}

\textbf{Temporal Graph Anomaly Detection}: Given a temporal graph $G = \{G_1, G_2, \ldots, G_T\}$ where $G_t = (V_t, E_t, X_t)$ represents the graph at time $t$ with nodes $V_t$, edges $E_t$, and node features $X_t$, our goal is to identify anomalous nodes in real-time as new interactions arrive.

\textbf{Deployment Requirements}: Unlike retrospective analysis, real-time deployment requires:
\begin{enumerate}
\item \textbf{Early Detection}: Identify anomalies with limited interaction history
\item \textbf{Cold Start Handling}: Evaluate new nodes with minimal data
\item \textbf{Temporal Consistency}: Maintain stable performance across time periods
\item \textbf{Real-time Processing}: Process streaming graph updates efficiently
\end{enumerate}

\subsection{TempAnom-GNN Architecture}

Our architecture integrates three components optimized for deployment scenarios:

\subsubsection{Temporal Evolution Encoder}

\textbf{Motivation}: Captures fine-grained temporal patterns that emerge before statistical anomalies become apparent.

\textbf{Design}: Building on DyRep~\cite{trivedi2019dyrep}, we encode temporal evolution through:

$$h_v^{(t)} = \text{GRU}(h_v^{(t-1)}, \text{Aggregate}(\{m_{uv}^{(\tau)} : \tau \leq t\}))$$

where $m_{uv}^{(\tau)}$ represents the message from node $u$ to $v$ at time $\tau$.

\textbf{Deployment Advantage}: Temporal patterns often precede statistical anomalies, enabling early detection.

\subsubsection{Memory Mechanism}

\textbf{Motivation}: Maintains representations of normal behavior patterns for comparison with current activity.

\textbf{Design}: Following TGN~\cite{rossi2020temporal}, we maintain memory states:

$$\text{Memory}_v^{(t)} = \text{Update}(\text{Memory}_v^{(t-1)}, h_v^{(t)}, \text{is\_normal}(G_t))$$

\textbf{Deployment Advantage}: Enables detection of deviations from established normal patterns.

\subsubsection{Trajectory Predictor}

\textbf{Motivation}: Predicts future interaction patterns to identify anomalous trajectories early.

\textbf{Design}: Inspired by JODIE~\cite{kumar2019predicting}, we predict future embeddings:

$$\hat{h}_v^{(t+\Delta)} = \text{MLP}(h_v^{(t)}, \Delta)$$

\textbf{Deployment Advantage}: Trajectory deviations signal emerging anomalies before they fully manifest.

\subsubsection{Unified Anomaly Scoring}

\textbf{Integration}: We combine components through learned weights:

$$\text{Score}_v^{(t)} = \alpha \cdot \text{EvolutionScore}_v^{(t)} + \beta \cdot \text{MemoryScore}_v^{(t)} + \gamma \cdot \text{TrajectoryScore}_v^{(t)}$$

\textbf{Optimization}: Weights are learned through deployment-focused loss functions that prioritize early detection and cold start performance.

\subsection{Training Strategy}

\subsubsection{Deployment-Focused Loss Function}

Traditional anomaly detection optimizes for retrospective performance. We design loss functions specifically for deployment requirements:

$$L = L_{\text{detection}} + \lambda_1 L_{\text{early}} + \lambda_2 L_{\text{coldstart}} + \lambda_3 L_{\text{consistency}}$$

where:
\begin{itemize}
\item $L_{\text{detection}}$: Standard anomaly detection loss
\item $L_{\text{early}}$: Early detection penalty (higher weight for early time steps)
\item $L_{\text{coldstart}}$: Cold start penalty (higher weight for nodes with limited data)
\item $L_{\text{consistency}}$: Temporal consistency regularization
\end{itemize}

\subsubsection{Temporal Batching Strategy}

\textbf{Challenge}: Traditional batching ignores temporal dependencies.

\textbf{Solution}: We employ temporal batching that preserves chronological order:
\begin{enumerate}
\item Sort interactions by timestamp
\item Create temporal windows with overlap
\item Ensure causality (no future information leakage)
\item Maintain memory states across batches
\end{enumerate}

\subsection{Component Analysis Framework}

\textbf{Methodology}: We evaluate single-component variants:
\begin{itemize}
\item \textbf{Evolution-only}: Temporal encoding without memory or prediction
\item \textbf{Memory-only}: Memory mechanism without evolution or prediction
\item \textbf{Prediction-only}: Trajectory prediction without memory or evolution
\item \textbf{Full-system}: All components combined
\end{itemize}

\textbf{Evaluation}: Each variant is tested on deployment scenarios (early detection, cold start) to identify optimal architectures.

Our methodology prioritizes real-world deployment requirements while maintaining the representational power needed for effective anomaly detection."""
        
        with open(f'{self.sections_dir}/methodology.tex', 'w') as f:
            f.write(methodology_tex)
        
        print("✅ methodology.tex fixed")
    
    def fix_all_sections(self):
        """Fix all section files"""
        print("🔧 FIXING ALL LATEX SECTIONS")
        print("="*40)
        
        self.fix_introduction_section()
        self.fix_related_work_section()
        self.fix_methodology_section()
        
        # For experiments, analysis, and conclusion, just remove the markdown headers
        self.fix_simple_section('experiments')
        self.fix_simple_section('analysis')
        self.fix_simple_section('conclusion')
        
        print("\n✅ ALL SECTIONS FIXED!")
        print("Ready for Overleaf upload")
    
    def fix_simple_section(self, section_name):
        """Fix sections by removing markdown headers"""
        print(f"🔧 Fixing {section_name}.tex...")
        
        section_file = f'{self.sections_dir}/{section_name}.tex'
        
        if os.path.exists(section_file):
            with open(section_file, 'r') as f:
                content = f.read()
            
            # Remove all markdown headers that start with #
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                if line.strip().startswith('#'):
                    # Convert to LaTeX section/subsection
                    if line.startswith('## '):
                        title = line.replace('## ', '').strip()
                        fixed_lines.append(f'\\subsection{{{title}}}')
                    elif line.startswith('### '):
                        title = line.replace('### ', '').strip()
                        fixed_lines.append(f'\\subsubsection{{{title}}}')
                    elif line.startswith('# '):
                        # Skip main section header (already handled)
                        continue
                else:
                    # Fix other LaTeX issues
                    line = line.replace('**', '\\textbf{').replace('**', '}')
                    line = line.replace('*', '\\textit{').replace('*', '}')
                    line = re.sub(r'`([^`]+)`', r'\\texttt{\1}', line)
                    fixed_lines.append(line)
            
            # Add proper section header
            section_titles = {
                'experiments': 'Experimental Evaluation',
                'analysis': 'Analysis and Discussion',
                'conclusion': 'Conclusion'
            }
            
            title = section_titles.get(section_name, section_name.title())
            latex_content = f"\\section{{{title}}}\n\\label{{sec:{section_name}}}\n\n" + '\n'.join(fixed_lines)
            
            with open(section_file, 'w') as f:
                f.write(latex_content)
            
            print(f"  ✅ {section_name}.tex fixed")
        else:
            print(f"  ⚠️ {section_name}.tex not found")


def main():
    """Execute LaTeX section fixing"""
    print("🔧 LATEX SECTION FIXER")
    print("="*30)
    
    fixer = LaTeXSectionFixer()
    fixer.fix_all_sections()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Upload the fixed files to Overleaf")
    print("2. Try compiling again")
    print("3. All sections should work properly now!")
    
    return True

if __name__ == "__main__":
    results = main()
