# Fraud Detection System Using GNN and Neo4j

A complete, production-ready fraud detection system that demonstrates how Graph Neural Networks can identify fraudulent transactions in financial networks.

## 🎯 Project Overview

This system analyzes credit card transactions stored in a Neo4j graph database and uses Graph Neural Networks to detect fraudulent patterns based on user behavior, merchant characteristics, and transaction relationships.

### Key Features
- **Realistic Data Generation**: Creates synthetic but realistic transaction data with fraud patterns
- **Graph Database**: Uses Neo4j to model relationships between users, merchants, and transactions
- **Advanced ML**: Implements GraphSAGE neural network for fraud detection
- **Rich Visualizations**: Generates confusion matrix, ROC curves, and fraud analysis charts
- **Demo-Ready**: Complete with metrics and professional outputs

## 📋 Prerequisites

### 1. Software Requirements
- Python 3.8 or higher
- Neo4j Database (Community or Desktop Edition)
- pip (Python package manager)

### 2. Hardware Requirements
- At least 4GB RAM
- 2GB free disk space

## 🚀 Installation Steps

### Step 1: Install Neo4j

**Option A: Neo4j Desktop (Recommended for beginners)**
1. Download from: https://neo4j.com/download/
2. Install and launch Neo4j Desktop
3. Create a new Project (e.g., "Fraud Detection")
4. Click "Add" → "Local DBMS"
5. Name it "FraudDB" and set password (remember this!)
6. Click "Start" to launch the database
7. Note the connection details (usually `bolt://localhost:7687`)

**Option B: Neo4j Community Server**
```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# macOS (using Homebrew)
brew install neo4j

# Start Neo4j
neo4j start
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv fraud_detection_env

# Activate virtual environment
# On Windows:
fraud_detection_env\Scripts\activate
# On Mac/Linux:
source fraud_detection_env/bin/activate

# Install required packages
pip install neo4j torch torch-geometric pandas numpy matplotlib seaborn scikit-learn networkx plotly

# If torch-geometric installation fails, try:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

### Step 3: Download the Project Script

1. Save the main Python script as `fraud_detection_system.py`
2. Open the file in a text editor

### Step 4: Configure Neo4j Connection

In `fraud_detection_system.py`, find this section around line 380:

```python
neo4j = Neo4jConnector(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password_here"  # CHANGE THIS!
)
```

**Update the password** to match what you set in Neo4j.

## ▶️ Running the Demo

### Quick Start

```bash
# Make sure Neo4j is running first!
# Then run the script:
python fraud_detection_system.py
```


## 📊 Understanding the Results

### 1. Confusion Matrix
Shows how well the model distinguishes between legitimate and fraudulent transactions:
- **True Negatives**: Correctly identified legitimate transactions
- **True Positives**: Correctly identified fraudulent transactions
- **False Positives**: Legitimate flagged as fraud (annoying but safe)
- **False Negatives**: Fraud missed (costly!)

### 2. ROC Curve
- Shows trade-off between catching frauds vs. false alarms
- **AUC Score**: 0.5 = random guess, 1.0 = perfect
- Typical good model: AUC > 0.85

### 3. Fraud Analysis Charts
- **Transaction Distribution**: Shows fraud vs. legitimate split
- **Amount Distribution**: Fraudsters typically use higher amounts
- **Hourly Patterns**: Fraud often occurs at odd hours
- **Fraud Percentage**: Shows high-risk time windows

## 🎓 Demo Presentation Tips

### Key Talking Points

1. **Problem Statement**
   - "Traditional fraud detection uses rules, but fraudsters adapt"
   - "GNNs can learn patterns from the network structure"

2. **Why Graphs?**
   - "Fraud rings: Multiple accounts working together"
   - "Merchant risk: Some merchants attract fraud"
   - "GNNs leverage relationships, not just individual features"

3. **Model Architecture**
   - "GraphSAGE: Learns from neighborhood aggregation"
   - "3 graph convolution layers"
   - "Processes both node features and graph structure"

4. **Results Interpretation**
   - Show the AUC-ROC score (typically >0.90)
   - Discuss precision vs. recall trade-off
   - Explain why some false positives are acceptable

### Live Demo Flow

```
1. Show Neo4j Browser (localhost:7474)
   - Run: MATCH (n) RETURN n LIMIT 50
   - Show the graph visualization

2. Run the Python script
   - Explain each step as it executes
   - Point out training progress

3. Open generated PNG files
   - Confusion matrix interpretation
   - ROC curve analysis
   - Fraud patterns visualization

4. Query Neo4j for specific examples
   MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction {is_fraud: 1})
   RETURN u.name, t.amount, t.timestamp
   LIMIT 10
```

## 🔧 Customization Options

### Adjust Data Scale

```python
# In main() function, modify:
generator = FraudDataGenerator(
    n_users=1000,      # Increase users
    n_merchants=200,   # Increase merchants
    n_transactions=10000  # More transactions
)
```

### Change Fraud Rate

```python
# In FraudDataGenerator.generate_transactions():
is_fraud = np.random.random() < 0.20  # 20% fraud rate
```

### Modify GNN Architecture

```python
# In FraudGNN class:
def __init__(self, num_features, hidden_dim=128, num_classes=2):
    # Increase hidden_dim for more capacity
```

### Adjust Training

```python
# In main() function:
trainer.train_model(epochs=200)  # More training
```

## 🐛 Troubleshooting

### Issue: "Failed to connect to Neo4j"
**Solution**: 
- Verify Neo4j is running: Check Neo4j Desktop or run `neo4j status`
- Check connection URI and credentials
- Test connection in Neo4j Browser (localhost:7474)

### Issue: "torch-geometric installation failed"
**Solution**:
```bash
# Install dependencies first
pip install torch
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

### Issue: "Out of memory"
**Solution**:
- Reduce data size in `FraudDataGenerator`
- Use smaller `hidden_dim` in GNN
- Close other applications

### Issue: "Graphs not displaying"
**Solution**:
- Make sure matplotlib backend is working
- Try adding: `plt.ion()` before plots
- Check if PNG files are created even if not displayed

## 📚 Additional Resources

### Neo4j
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/developer/cypher/)

### Graph Neural Networks
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)

### Fraud Detection
- [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Fraud Detection Research](https://arxiv.org/abs/2003.07070)

## 🤝 Contributing

This is a demo project. Feel free to:
- Add more sophisticated fraud patterns
- Implement additional GNN architectures (GAT, GIN)
- Add real-time fraud detection
- Create a web dashboard
- Integrate with real transaction data


## ✨ Credits

Created as a comprehensive fraud detection demo using:
- Neo4j for graph database
- PyTorch Geometric for GNN implementation
- Scikit-learn for evaluation metrics
- Matplotlib/Seaborn for visualizations
