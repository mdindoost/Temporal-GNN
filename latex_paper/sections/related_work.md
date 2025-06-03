
# 2. Related Work

## 2.1 Static Graph Anomaly Detection

Traditional graph anomaly detection methods operate on static network snapshots, identifying anomalous nodes, edges, or subgraphs based on structural deviations [7,8]. **DOMINANT** [9] introduced graph autoencoders for attributed network anomaly detection, achieving strong performance on retrospective evaluation. **Radar** [10] and **GraphSAINT** [11] further advanced static methods through attention mechanisms and sampling strategies.

However, static methods face fundamental limitations in dynamic environments: (1) they cannot capture temporal evolution patterns, (2) they require complete network snapshots, and (3) they lack mechanisms for early detection. Our evaluation confirms that while these methods excel at retrospective analysis—particularly simple statistical approaches like negative rating ratios—they struggle in prospective deployment scenarios where temporal patterns provide crucial early warning signals.

## 2.2 Temporal Graph Neural Networks

Recent advances in temporal graph neural networks have addressed dynamic modeling through various approaches. **Temporal Graph Networks (TGN)** [12] introduced memory mechanisms to maintain node representations across time, enabling effective temporal modeling for link prediction. **DyRep** [13] proposed multi-scale temporal encoding to capture both local and global temporal patterns. **JODIE** [14] introduced trajectory prediction for user behavior modeling in dynamic networks.

While these methods demonstrate strong performance on temporal tasks, they were not designed for anomaly detection. Key differences in our approach include: (1) **anomaly-specific objective functions** that optimize for detection rather than prediction, (2) **deployment-focused evaluation** that addresses real-world requirements, and (3) **component analysis** that provides architectural guidance for anomaly detection applications.

## 2.3 Temporal Anomaly Detection

Existing temporal anomaly detection primarily focuses on time series [15,16] or static graph sequences [17,18]. **TADGAN** [19] applies adversarial training to temporal sequences, while **CAD-Walk** [20] uses random walks on temporal networks. However, these approaches either ignore graph structure or treat temporal information as auxiliary.

**Our work differs fundamentally** by designing temporal graph neural networks specifically for anomaly detection, integrating both structural and temporal patterns for real-time deployment scenarios. Our evaluation framework also advances the field by distinguishing between retrospective analysis capabilities and prospective deployment requirements.

## 2.4 Fraud Detection in Financial Networks

Financial fraud detection has employed various approaches, from traditional machine learning [21,22] to recent graph-based methods [23,24]. Bitcoin network analysis has particularly benefited from graph approaches due to the transparent transaction history [25,26].

However, most financial fraud detection research focuses on retrospective analysis using complete transaction histories. **Our contribution addresses the deployment gap** by evaluating prospective detection capabilities required for real-time financial fraud monitoring systems.

## 2.5 Positioning of Our Work

Our work advances temporal graph anomaly detection through four key differentiators:

1. **First temporal GNN designed specifically for anomaly detection** (vs. adapted link prediction methods)
2. **Deployment-focused evaluation framework** (vs. retrospective-only evaluation)  
3. **Statistical validation with confidence intervals** (vs. single-point estimates)
4. **Component analysis for architectural guidance** (vs. black-box evaluation)

This positioning enables both methodological advances and practical deployment insights for the temporal graph research community.
