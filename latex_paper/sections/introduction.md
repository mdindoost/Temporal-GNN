
# 1. Introduction

Fraud detection in dynamic networks presents a fundamental challenge: while retrospective analysis can effectively identify suspicious patterns after they are fully established, real-world deployment requires prospective capabilities that can detect anomalies as they emerge. Traditional graph-based anomaly detection methods operate on static snapshots, missing the temporal evolution patterns that characterize emerging fraud [1,2]. Recent temporal graph neural networks (TGNNs) have shown promise for capturing dynamic patterns [3,4], but existing approaches focus primarily on link prediction and node classification rather than anomaly detection.

Consider fraud detection in financial trust networks: simple statistical methods excel when complete user interaction histories are available for retrospective analysis. A user with 80% negative ratings is easily identified as suspicious using basic metrics. However, production fraud detection systems must operate prospectively, identifying potentially fraudulent users early in their activity lifecycle—often with limited interaction data—and adapting to evolving fraud patterns over time.

This deployment reality reveals a critical gap in current evaluation practices. Existing temporal graph anomaly detection methods are typically evaluated on retrospective metrics using complete historical data [5,6]. While these evaluations demonstrate algorithmic capabilities, they do not address the practical challenges of real-time deployment: early detection with incomplete data, handling new users with limited interaction history, and maintaining consistent performance across evolving time periods.

We present **TempAnom-GNN**, the first temporal graph neural network designed specifically for real-time anomaly detection in evolving networks. Our approach integrates three key components: (1) temporal evolution encoding that captures dynamic interaction patterns, (2) memory mechanisms that maintain representations of normal behavior, and (3) trajectory prediction that enables early anomaly identification. Through comprehensive evaluation on Bitcoin trust networks, we demonstrate that while simple statistical baselines excel at retrospective analysis, TempAnom-GNN provides significant advantages in realistic deployment scenarios.

**Our key contributions are:**

1. **Methodological**: The first temporal graph neural network architecture designed specifically for real-time anomaly detection, integrating TGN memory mechanisms, DyRep temporal encoding, and JODIE trajectory prediction.

2. **Empirical**: Comprehensive evaluation framework distinguishing retrospective analysis from prospective deployment, demonstrating statistically significant improvements in early detection (+20.8%, p < 0.0001) and cold start scenarios (+13.2%, p = 0.0017).

3. **Analytical**: Component analysis revealing that evolution-only architectures outperform complex combinations in deployment scenarios, providing architectural guidance for practitioners.

4. **Framework**: Introduction of deployment-focused evaluation methodology that challenges retrospective-only benchmarking practices in temporal graph research.

We provide clear guidance on when to use simple statistical methods (retrospective analysis) versus temporal graph approaches (prospective deployment), advancing both methodology and evaluation practices for temporal anomaly detection.
