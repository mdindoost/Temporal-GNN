# Literature Summary: Temporal Graph Neural Networks for Anomaly Detection

## 📚 **Overview**

This document provides a comprehensive literature review framework for our temporal GNN anomaly detection research. It serves as a structured guide for analyzing key papers and extracting insights relevant to our project objectives.

---

## 🎯 **Research Context & Our Baseline Results**

### **Our Problem Statement**
- **Challenge**: Static GNN approaches fail on temporal network anomalies
- **Evidence**: DOMINANT baseline achieved AUC = 0.33 (worse than random)
- **Root Cause**: Density bias - dense anomalies (star bursts, cliques) get lower scores
- **Need**: Temporal context to distinguish "normal evolution" from "anomalous changes"

### **Key Questions for Literature**
1. How do existing temporal GNNs handle evolving network structures?
2. What memory mechanisms help distinguish normal vs anomalous patterns?
3. How is temporal information incorporated in graph learning?
4. What are the computational trade-offs for real-time anomaly detection?

---

## 📖 **Core Papers Analysis Framework**

For each paper, we analyze:
- **Technical Approach**: Architecture, key innovations
- **Temporal Modeling**: How time is incorporated
- **Anomaly Detection**: If applicable, how anomalies are defined/detected
- **Strengths & Limitations**: What works, what doesn't
- **Relevance to Our Work**: Specific techniques we can adapt

---

## 🏗️ **1. STATIC GRAPH ANOMALY DETECTION**

### **DOMINANT: Deep Anomaly Detection on Attributed Networks**
**Reference**: Ding et al., WWW 2019  
**Link**: https://dl.acm.org/doi/10.1145/3308558.3313692

#### **Summary**
DOMINANT uses a graph autoencoder approach where a GCN encoder maps nodes to embeddings, and a decoder reconstructs the adjacency matrix. Anomalies are detected based on reconstruction error.

#### **Technical Approach**
```
Input Graph → GCN Encoder → Node Embeddings → Decoder → Reconstructed Graph
                                           ↓
                           Anomaly Score = Reconstruction Error
```

**Architecture Details**:
- **Encoder**: 2-layer GCN with ReLU activations
- **Decoder**: Inner product followed by sigmoid
- **Loss**: Binary cross-entropy for edge reconstruction
- **Training**: Autoencoder-style on "normal" graph data

#### **Key Insights**
✅ **Strengths**:
- Simple, interpretable approach
- Works well on static structural anomalies
- Established baseline for graph anomaly detection

❌ **Limitations** (Confirmed by Our Results):
- **Density Bias**: Dense structures are easier to reconstruct
- **No Temporal Context**: Cannot distinguish time-dependent patterns
- **Static Assumption**: Treats each graph independently
- **Limited Anomaly Types**: Focuses on node-level anomalies

#### **Relevance to Our Work**
- ✅ **Foundation**: Our baseline implementation
- ✅ **Motivation**: Demonstrates static approach limitations
- 🔄 **Extension Opportunity**: Add temporal components to DOMINANT framework

#### **Our Experimental Validation**
- Reproduced core DOMINANT approach on temporal data
- **Confirmed density bias**: Star bursts (0.0014) < Normal (0.0016)
- **Validated need for temporal modeling**: AUC = 0.33

---

### **GraphSAINT: Graph Sampling Based Inductive Learning**
**Reference**: Zeng et al., ICLR 2020  
**Status**: 📚 TO READ

#### **Why This Paper Matters**
- Addresses scalability issues in graph learning
- Sampling strategies could be relevant for temporal graphs
- Inductive learning important for evolving networks

#### **Questions to Answer**
- How does sampling affect anomaly detection performance?
- Can sampling strategies be adapted for temporal contexts?
- What are the computational benefits for real-time detection?

---

### **ANOMALOUS: Joint Modeling for Anomaly Detection**
**Reference**: Peng et al., CIKM 2018  
**Status**: 📚 TO READ

#### **Why This Paper Matters**
- Joint modeling of structure and attributes
- Alternative to reconstruction-based approaches
- May address some density bias issues

#### **Questions to Answer**
- How does joint modeling compare to pure reconstruction?
- What anomaly types does this approach handle well?
- Can joint modeling be extended to temporal settings?

---

## ⏰ **2. TEMPORAL GRAPH NEURAL NETWORKS**

### **TGN: Temporal Graph Networks for Deep Learning on Dynamic Graphs**
**Reference**: Rossi et al., ICML 2020  
**Link**: https://proceedings.mlr.press/v119/rossi20a.html  
**Status**: 🔥 HIGH PRIORITY

#### **Why This Paper is Crucial**
- **State-of-the-art** temporal graph learning
- **Memory mechanisms** for temporal patterns
- **Attention-based** temporal aggregation
- **Perfect fit** for our anomaly detection extension

#### **Expected Key Components**
```
Temporal Graph → Memory Module → Temporal Attention → Updated Embeddings
                      ↓
              Message Passing + Time Encoding
```

#### **Questions to Answer**
1. **Memory Architecture**: How is historical information stored/retrieved?
2. **Temporal Attention**: How are different time points weighted?
3. **Message Passing**: How do temporal messages differ from static ones?
4. **Training Strategy**: How is the model trained on temporal sequences?
5. **Computational Complexity**: What are the scalability trade-offs?

#### **Anomaly Detection Extensions**
- Can memory detect deviations from normal temporal patterns?
- How can attention weights indicate anomalous time periods?
- What temporal features are most informative for anomaly detection?

---

### **DyRep: Learning Representations over Dynamic Graphs**
**Reference**: Trivedi et al., AAAI 2019  
**Status**: 🔥 HIGH PRIORITY

#### **Why This Paper Matters**
- **Dynamic representations** that evolve over time
- **Point process modeling** for temporal events
- **Node evolution tracking** - key for anomaly detection

#### **Expected Key Insights**
- How do node representations change over time?
- What constitutes "normal" vs "anomalous" evolution?
- How can we detect sudden representation changes?

#### **Questions to Answer**
1. **Representation Evolution**: How do embeddings change over time?
2. **Point Processes**: How are temporal events modeled?
3. **Anomaly Indicators**: What signals indicate anomalous behavior?
4. **Training Dynamics**: How is temporal consistency maintained?

---

### **JODIE: Predicting Dynamic Embedding Trajectory**
**Reference**: Kumar et al., KDD 2019  
**Status**: 📚 TO READ

#### **Why This Paper Matters**
- **Trajectory prediction** in embedding space
- **Bipartite temporal networks** (users-items)
- **Projection operators** for future state prediction

#### **Questions to Answer**
- How can trajectory prediction help anomaly detection?
- What makes a trajectory "anomalous"?
- Can we detect anomalies by prediction errors?

---

## 🔍 **3. GRAPH ANOMALY DETECTION SURVEYS**

### **Deep Learning for Anomaly Detection: A Review**
**Reference**: Pang et al., ACM Computing Surveys 2021  
**Status**: 📚 TO READ

#### **Value for Our Research**
- **Comprehensive taxonomy** of anomaly detection approaches
- **Evaluation methodologies** and best practices
- **Comparison frameworks** for different techniques

#### **Key Sections to Focus On**
- Graph-based anomaly detection methods
- Temporal anomaly detection approaches
- Evaluation metrics and datasets
- Future research directions

---

### **Graph Neural Networks for Anomaly Detection: A Survey**
**Reference**: Ma et al., arXiv 2021  
**Status**: 📚 TO READ

#### **Specific Focus Areas**
- GNN architectures for anomaly detection
- Graph-level vs node-level vs edge-level anomalies
- Benchmark datasets and evaluation protocols

---

### **Anomaly Detection in Dynamic Networks: A Survey**
**Reference**: Ranshous et al., Computer Communications 2020  
**Status**: 📚 TO READ

#### **Critical for Our Work**
- **Dynamic network anomalies** - exactly our focus
- **Temporal patterns** in network evolution
- **Real-world applications** and case studies

---

## 📋 **READING PLAN & PRIORITIES**

### **Phase 1: Core Technical Papers (Week 1)**
1. **TGN** (Temporal Graph Networks) - 🔥 **START HERE**
   - Focus: Memory mechanisms, temporal attention
   - Goal: Understand how to add temporal components to our baseline

2. **DyRep** (Dynamic Representations)
   - Focus: Node evolution, anomaly indicators
   - Goal: Learn how representations should change over time

3. **DOMINANT** (Re-read with new perspective)
   - Focus: Understand limitations we discovered
   - Goal: Identify specific extension points

### **Phase 2: Survey Papers (Week 2)**
4. **Graph Anomaly Detection Survey**
   - Focus: Taxonomy, evaluation methods
   - Goal: Position our work in broader context

5. **Dynamic Networks Survey**
   - Focus: Temporal anomaly patterns
   - Goal: Understand real-world applications

### **Phase 3: Additional Methods (Week 3)**
6. **JODIE** (Trajectory Prediction)
7. **GraphSAINT** (Scalability)
8. **ANOMALOUS** (Joint Modeling)

---

## 🎯 **ANALYSIS TEMPLATE FOR EACH PAPER**

### **1. Technical Analysis**
- **Problem Definition**: What problem does this solve?
- **Key Innovation**: What's new/different?
- **Architecture**: Detailed technical approach
- **Mathematical Formulation**: Key equations and algorithms
- **Training Procedure**: How is the model trained?

### **2. Temporal Aspects** (If Applicable)
- **Time Representation**: How is time encoded/handled?
- **Temporal Dependencies**: What temporal patterns are captured?
- **Memory Mechanisms**: How is historical information used?
- **Sequence Modeling**: How are temporal sequences processed?

### **3. Anomaly Detection Relevance**
- **Anomaly Definition**: How are anomalies characterized?
- **Detection Mechanism**: How are anomalies identified?
- **Evaluation**: What metrics and datasets are used?
- **Limitations**: What anomaly types are missed?

### **4. Implementation Insights**
- **Computational Complexity**: Scalability considerations
- **Implementation Details**: Key algorithmic components
- **Hyperparameters**: Important configuration choices
- **Code Availability**: Reproducibility aspects

### **5. Adaptation to Our Work**
- **Direct Applications**: Components we can use directly
- **Modifications Needed**: How to adapt for our problem
- **Integration Strategy**: How to combine with our baseline
- **Expected Improvements**: Anticipated performance gains

---

## 💡 **KEY RESEARCH QUESTIONS TO ANSWER**

### **From TGN Paper**
1. How can we adapt TGN's memory mechanism for anomaly detection?
2. What temporal attention patterns indicate anomalous behavior?
3. How should we modify the message passing for reconstruction tasks?

### **From DyRep Paper**
1. How do "normal" node representations evolve over time?
2. What representation changes indicate anomalous behavior?
3. Can we detect anomalies through embedding trajectory analysis?

### **From Survey Papers**
1. What evaluation frameworks should we adopt?
2. How does our approach compare to existing temporal methods?
3. What benchmark datasets should we include?

---

## 🔧 **IMPLEMENTATION ROADMAP FROM LITERATURE**

### **Step 1: Memory-Enhanced DOMINANT**
Based on TGN insights:
```python
class TemporalDOMINANT(nn.Module):
    def __init__(self):
        self.memory = TemporalMemory()        # From TGN
        self.encoder = GCNEncoder()           # Our baseline
        self.temporal_attention = Attention() # From TGN
        self.decoder = InnerProductDecoder()  # Our baseline
```

### **Step 2: Dynamic Representation Tracking**
Based on DyRep insights:
```python
class DynamicAnomalyDetector:
    def detect_anomalies(self, embeddings_history):
        # Track embedding evolution
        # Detect sudden changes
        # Compare to normal patterns
```

### **Step 3: Temporal Training Strategy**
Based on combined insights:
```python
def temporal_training_loop():
    for timestamp in sequence:
        # Update memory with current state
        # Compute temporal-aware embeddings  
        # Detect anomalies based on deviation
        # Update model parameters
```

---

## 📊 **EXPECTED OUTCOMES FROM LITERATURE REVIEW**

### **Technical Contributions**
1. **Architecture Design**: Specific temporal components to add
2. **Training Strategy**: How to train on temporal sequences
3. **Anomaly Scoring**: Better methods than reconstruction error
4. **Evaluation Framework**: Appropriate metrics and baselines

### **Research Positioning**
1. **Gap Identification**: What's missing in current approaches
2. **Contribution Clarity**: How our work advances the field
3. **Comparison Framework**: Fair evaluation against existing methods
4. **Future Directions**: Extensions and improvements

### **Implementation Guidance**
1. **Component Prioritization**: Which temporal features matter most
2. **Complexity Trade-offs**: Accuracy vs computational efficiency
3. **Hyperparameter Insights**: Key configuration choices
4. **Debugging Strategies**: Common pitfalls and solutions

---

## ✅ **COMPLETION CHECKLIST**

### **For Each Paper**
- [ ] Technical approach understood
- [ ] Key innovations identified  
- [ ] Temporal aspects analyzed
- [ ] Anomaly detection relevance assessed
- [ ] Implementation insights extracted
- [ ] Adaptation strategy planned

### **Overall Literature Review**
- [ ] Gap analysis completed
- [ ] Research positioning clarified
- [ ] Technical roadmap defined
- [ ] Evaluation framework designed
- [ ] Implementation priorities set

---

## 🚀 **NEXT STEPS AFTER LITERATURE REVIEW**

### **Week 4: Architecture Design**
1. **Combine TGN + DOMINANT**: Memory-enhanced graph autoencoder
2. **Design temporal features**: What makes patterns anomalous
3. **Plan training strategy**: Temporal sequence processing
4. **Define evaluation metrics**: Temporal-aware performance measures

### **Week 5: Implementation**
1. **Build temporal components**: Memory, attention, sequence modeling
2. **Integrate with baseline**: Extend our working DOMINANT model
3. **Test on synthetic data**: Validate temporal improvements
4. **Optimize hyperparameters**: Tune for best performance

### **Week 6: Evaluation**
1. **Compare temporal vs static**: Quantify improvements
2. **Analyze failure cases**: What anomalies are still missed
3. **Test on Bitcoin data**: Real-world validation
4. **Prepare for Phase 3**: Scale to larger experiments

---

*This literature summary will be continuously updated as we read each paper and extract key insights. Each paper analysis will be added to this document with detailed technical notes and implementation strategies.*
