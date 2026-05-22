"""
Fraud Detection System Using Graph Neural Networks and Neo4j
Complete implementation with data generation, GNN training, and visualization
IMPROVED VERSION - Better fraud detection with proper class balancing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from neo4j import GraphDatabase
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

#============================================================================
# 1. DATA GENERATION MODULE
#============================================================================

class FraudDataGenerator:
    """Generate realistic transaction data with fraud patterns"""
    
    def __init__(self, n_users=500, n_merchants=100, n_transactions=5000):
        self.n_users = n_users
        self.n_merchants = n_merchants
        self.n_transactions = n_transactions
        
    def generate_users(self):
        """Generate user profiles"""
        users = []
        for i in range(self.n_users):
            user = {
                'user_id': f'U{i:04d}',
                'name': f'User_{i}',
                'age': np.random.randint(18, 80),
                'account_age_days': np.random.randint(30, 3650),
                'credit_score': np.random.randint(300, 850),
                'avg_transaction_amount': np.random.uniform(50, 500)
            }
            users.append(user)
        return pd.DataFrame(users)
    
    def generate_merchants(self):
        """Generate merchant profiles"""
        categories = ['Electronics', 'Groceries', 'Fashion', 'Travel', 
                     'Restaurants', 'Gas', 'Healthcare', 'Entertainment']
        merchants = []
        for i in range(self.n_merchants):
            merchant = {
                'merchant_id': f'M{i:03d}',
                'name': f'Merchant_{i}',
                'category': random.choice(categories),
                'risk_score': np.random.uniform(0, 1),
                'transaction_count': 0
            }
            merchants.append(merchant)
        return pd.DataFrame(merchants)
    
    def generate_transactions(self, users_df, merchants_df):
        """Generate transactions with realistic fraud patterns"""
        transactions = []
        start_date = datetime.now() - timedelta(days=180)
        
        # Create some fraudulent user groups (fraud rings)
        fraud_users = random.sample(list(users_df['user_id']), 50)
        fraud_merchants = random.sample(list(merchants_df['merchant_id']), 10)
        
        for i in range(self.n_transactions):
            # Determine if this transaction is fraudulent (15% fraud rate)
            is_fraud = np.random.random() < 0.15
            
            if is_fraud:
                # Fraudulent transaction patterns
                user_id = random.choice(fraud_users)
                merchant_id = random.choice(fraud_merchants)
                amount = np.random.uniform(500, 2000)  # Higher amounts
                timestamp = start_date + timedelta(
                    hours=int(np.random.randint(0, 4320)),
                    minutes=int(np.random.randint(0, 60))
                )
                # Fraud often happens at odd hours
                hour = np.random.choice([0, 1, 2, 3, 22, 23])
            else:
                # Normal transaction patterns
                user_id = random.choice(list(users_df['user_id']))
                merchant_id = random.choice(list(merchants_df['merchant_id']))
                amount = np.random.lognormal(4, 1)  # Natural spending distribution
                timestamp = start_date + timedelta(
                    hours=int(np.random.randint(0, 4320)),
                    minutes=int(np.random.randint(0, 60))
                )
                # Normal transactions during business hours
                hour = np.random.randint(8, 22)
            
            transaction = {
                'transaction_id': f'T{i:05d}',
                'user_id': user_id,
                'merchant_id': merchant_id,
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'hour': hour,
                'is_fraud': 1 if is_fraud else 0
            }
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

#============================================================================
# 2. NEO4J DATABASE MODULE
#============================================================================

class Neo4jConnector:
    """Handle Neo4j database operations"""
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
                session.run("CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE")
                session.run("CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE")
                print("✓ Constraints created")
            except Exception as e:
                print(f"Constraints may already exist: {e}")
    
    def load_users(self, users_df):
        """Load users into Neo4j"""
        with self.driver.session() as session:
            for _, user in users_df.iterrows():
                session.run("""
                    CREATE (u:User {
                        user_id: $user_id,
                        name: $name,
                        age: $age,
                        account_age_days: $account_age_days,
                        credit_score: $credit_score,
                        avg_transaction_amount: $avg_transaction_amount
                    })
                """, **user.to_dict())
        print(f"✓ Loaded {len(users_df)} users")
    
    def load_merchants(self, merchants_df):
        """Load merchants into Neo4j"""
        with self.driver.session() as session:
            for _, merchant in merchants_df.iterrows():
                session.run("""
                    CREATE (m:Merchant {
                        merchant_id: $merchant_id,
                        name: $name,
                        category: $category,
                        risk_score: $risk_score
                    })
                """, **merchant.to_dict())
        print(f"✓ Loaded {len(merchants_df)} merchants")
    
    def load_transactions(self, transactions_df):
        """Load transactions and create relationships"""
        with self.driver.session() as session:
            for _, txn in transactions_df.iterrows():
                session.run("""
                    MATCH (u:User {user_id: $user_id})
                    MATCH (m:Merchant {merchant_id: $merchant_id})
                    CREATE (t:Transaction {
                        transaction_id: $transaction_id,
                        amount: $amount,
                        timestamp: datetime($timestamp),
                        hour: $hour,
                        is_fraud: $is_fraud
                    })
                    CREATE (u)-[:MADE_TRANSACTION]->(t)
                    CREATE (t)-[:AT_MERCHANT]->(m)
                """, 
                user_id=txn['user_id'],
                merchant_id=txn['merchant_id'],
                transaction_id=txn['transaction_id'],
                amount=float(txn['amount']),
                timestamp=txn['timestamp'].isoformat(),
                hour=int(txn['hour']),
                is_fraud=int(txn['is_fraud'])
                )
        print(f"✓ Loaded {len(transactions_df)} transactions")
    
    def get_graph_data(self):
        """Extract graph data for GNN"""
        with self.driver.session() as session:
            # Get all users with features
            users_result = session.run("""
                MATCH (u:User)
                RETURN u.user_id as user_id, 
                       u.age as age,
                       u.account_age_days as account_age_days,
                       u.credit_score as credit_score,
                       u.avg_transaction_amount as avg_amount
                ORDER BY u.user_id
            """)
            users = [dict(record) for record in users_result]
            
            # Get all merchants with features
            merchants_result = session.run("""
                MATCH (m:Merchant)
                RETURN m.merchant_id as merchant_id,
                       m.risk_score as risk_score
                ORDER BY m.merchant_id
            """)
            merchants = [dict(record) for record in merchants_result]
            
            # Get all transactions
            transactions_result = session.run("""
                MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)-[:AT_MERCHANT]->(m:Merchant)
                RETURN u.user_id as user_id,
                       m.merchant_id as merchant_id,
                       t.transaction_id as transaction_id,
                       t.amount as amount,
                       t.hour as hour,
                       t.is_fraud as is_fraud
                ORDER BY t.transaction_id
            """)
            transactions = [dict(record) for record in transactions_result]
            
            return users, merchants, transactions

#============================================================================
# 3. IMPROVED GRAPH NEURAL NETWORK MODULE
#============================================================================

class ImprovedFraudGNN(nn.Module):
    """Enhanced Graph Neural Network for fraud detection with attention"""
    
    def __init__(self, num_features, hidden_dim=128, num_classes=2, dropout=0.3):
        super(ImprovedFraudGNN, self).__init__()
        
        # Use Graph Attention Networks for better learning
        self.conv1 = GATConv(num_features, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        
        # Additional layers for better feature extraction
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Third GAT layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class ImprovedGNNTrainer:
    """Train and evaluate GNN model with better strategies"""
    
    def __init__(self, users, merchants, transactions):
        self.users = users
        self.merchants = merchants
        self.transactions = transactions
        self.prepare_graph_data()
    
    def prepare_graph_data(self):
        """Convert Neo4j data to PyTorch Geometric format with better features"""
        # Create node mappings
        user_ids = [u['user_id'] for u in self.users]
        merchant_ids = [m['merchant_id'] for m in self.merchants]
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.merchant_to_idx = {mid: idx + len(user_ids) for idx, mid in enumerate(merchant_ids)}
        
        # Calculate transaction statistics per user for better features
        user_txn_stats = {}
        for txn in self.transactions:
            uid = txn['user_id']
            if uid not in user_txn_stats:
                user_txn_stats[uid] = {'amounts': [], 'hours': [], 'fraud_count': 0}
            user_txn_stats[uid]['amounts'].append(txn['amount'])
            user_txn_stats[uid]['hours'].append(txn['hour'])
            user_txn_stats[uid]['fraud_count'] += txn['is_fraud']
        
        # Create enhanced node features
        user_features = []
        for u in self.users:
            uid = u['user_id']
            stats = user_txn_stats.get(uid, {'amounts': [0], 'hours': [12], 'fraud_count': 0})
            
            features = [
                u['age'] / 100.0,
                u['account_age_days'] / 3650.0,
                u['credit_score'] / 850.0,
                u['avg_amount'] / 1000.0,
                np.mean(stats['amounts']) / 1000.0 if stats['amounts'] else 0,
                np.std(stats['amounts']) / 500.0 if len(stats['amounts']) > 1 else 0,
                len(stats['amounts']) / 50.0,  # Transaction frequency
                stats['fraud_count'] / max(len(stats['amounts']), 1)  # Historical fraud rate
            ]
            user_features.append(features)
        
        merchant_features = []
        for m in self.merchants:
            features = [
                m['risk_score'], 
                0.5, 0.5, 0.5,  # Neutral values
                0.5, 0.5, 0.5, 0.0
            ]
            merchant_features.append(features)
        
        all_features = user_features + merchant_features
        self.x = torch.FloatTensor(all_features)
        
        # Normalize features
        mean = self.x.mean(dim=0, keepdim=True)
        std = self.x.std(dim=0, keepdim=True) + 1e-8
        self.x = (self.x - mean) / std
        
        # Create edges and collect transaction-level data
        edge_list = []
        self.transaction_node_map = []  # Maps transaction index to user node
        self.transaction_labels = []
        
        for txn in self.transactions:
            user_idx = self.user_to_idx[txn['user_id']]
            merchant_idx = self.merchant_to_idx[txn['merchant_id']]
            edge_list.append([user_idx, merchant_idx])
            edge_list.append([merchant_idx, user_idx])
            
            self.transaction_node_map.append(user_idx)
            self.transaction_labels.append(txn['is_fraud'])
        
        self.edge_index = torch.LongTensor(edge_list).t().contiguous()
        self.y = torch.LongTensor(self.transaction_labels)
        
        # Create stratified train/test split
        n_txns = len(self.transactions)
        indices = list(range(n_txns))
        random.shuffle(indices)
        
        # Ensure balanced split
        fraud_indices = [i for i in indices if self.transaction_labels[i] == 1]
        legit_indices = [i for i in indices if self.transaction_labels[i] == 0]
        
        train_fraud = fraud_indices[:int(0.8 * len(fraud_indices))]
        test_fraud = fraud_indices[int(0.8 * len(fraud_indices)):]
        train_legit = legit_indices[:int(0.8 * len(legit_indices))]
        test_legit = legit_indices[int(0.8 * len(legit_indices)):]
        
        self.train_indices = train_fraud + train_legit
        self.test_indices = test_fraud + test_legit
        
        random.shuffle(self.train_indices)
        random.shuffle(self.test_indices)
        
        # Compute class weights for balanced training
        labels_array = np.array(self.transaction_labels)
        class_weights = compute_class_weight('balanced', 
                                             classes=np.array([0, 1]), 
                                             y=labels_array[self.train_indices])
        self.class_weights = torch.FloatTensor(class_weights)
        
        print(f"✓ Graph prepared: {len(all_features)} nodes, {len(edge_list)} edges")
        print(f"  Train: {len(self.train_indices)} txns ({sum([self.transaction_labels[i] for i in self.train_indices])} fraud)")
        print(f"  Test: {len(self.test_indices)} txns ({sum([self.transaction_labels[i] for i in self.test_indices])} fraud)")
        print(f"  Class weights: Legit={class_weights[0]:.2f}, Fraud={class_weights[1]:.2f}")
    
    def train_model(self, epochs=200, lr=0.003):
        """Train the GNN model with better strategies"""
        model = ImprovedFraudGNN(num_features=self.x.shape[1], hidden_dim=128, dropout=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
        
        train_losses = []
        best_loss = float('inf')
        best_f1 = 0
        patience_counter = 0
        
        print("\nTraining Improved GNN model...")
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(self.x, self.edge_index)
            
            # Get predictions for training transactions
            train_user_nodes = [self.transaction_node_map[i] for i in self.train_indices]
            train_labels = [self.transaction_labels[i] for i in self.train_indices]
            
            train_preds = out[train_user_nodes]
            train_labels_tensor = torch.LongTensor(train_labels)
            
            # Weighted loss for class imbalance
            loss = F.nll_loss(train_preds, train_labels_tensor, weight=self.class_weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping based on F1 score instead of just loss
            if epoch >= 30:  # Give it time to learn
                model.eval()
                with torch.no_grad():
                    test_user_nodes = [self.transaction_node_map[i] for i in self.test_indices]
                    test_labels_list = [self.transaction_labels[i] for i in self.test_indices]
                    test_preds_out = out[test_user_nodes]
                    pred_classes = test_preds_out.argmax(dim=1).numpy()
                    
                    # Calculate F1 for fraud class
                    from sklearn.metrics import f1_score
                    f1 = f1_score(test_labels_list, pred_classes, pos_label=1)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_loss = loss.item()
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
            elif loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict().copy()
            
            if epoch % 15 == 0:
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    test_user_nodes = [self.transaction_node_map[i] for i in self.test_indices]
                    test_labels = [self.transaction_labels[i] for i in self.test_indices]
                    test_preds = out[test_user_nodes]
                    test_labels_tensor = torch.LongTensor(test_labels)
                    test_loss = F.nll_loss(test_preds, test_labels_tensor)
                    
                    # Calculate accuracy
                    pred_classes = test_preds.argmax(dim=1)
                    acc = (pred_classes == test_labels_tensor).float().mean()
                    
                print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Test Acc: {acc:.4f}")
            
            if patience_counter >= 40:
                print(f"Early stopping at epoch {epoch} (Best F1: {best_f1:.4f})")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        self.model = model
        self.train_losses = train_losses
        print("✓ Training completed!")
        
    def evaluate_model(self):
        """Evaluate model performance"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.x, self.edge_index)
            probs = torch.exp(logits)
            
            # Get predictions for all transactions
            y_true = []
            y_pred = []
            y_probs = []
            
            for i in range(len(self.transactions)):
                user_idx = self.transaction_node_map[i]
                pred_class = logits[user_idx].argmax().item()
                prob_fraud = probs[user_idx][1].item()
                
                y_true.append(self.transaction_labels[i])
                y_pred.append(pred_class)
                y_probs.append(prob_fraud)
            
            # Calculate metrics
            print("\n" + "="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            print(classification_report(y_true, y_pred, 
                                      target_names=['Legitimate', 'Fraudulent'],
                                      digits=3))
            
            cm = confusion_matrix(y_true, y_pred)
            auc = roc_auc_score(y_true, y_probs)
            print(f"\nAUC-ROC Score: {auc:.4f}")
            
            # Additional metrics
            tn, fp, fn, tp = cm.ravel()
            print(f"\nDetailed Metrics:")
            print(f"  True Negatives:  {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  True Positives:  {tp}")
            print(f"  Fraud Detection Rate: {tp/(tp+fn)*100:.2f}%")
            
            return y_true, y_pred, y_probs, cm

#============================================================================
# 4. VISUALIZATION MODULE
#============================================================================

class FraudVisualizer:
    """Create visualizations for fraud detection results"""
    
    @staticmethod
    def plot_confusion_matrix(cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraudulent'],
                   yticklabels=['Legitimate', 'Fraudulent'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confusion_matrix.png")
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_probs):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'GNN Model (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Fraud Detection', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: roc_curve.png")
        plt.show()
    
    @staticmethod
    def plot_fraud_distribution(transactions_df):
        """Plot fraud distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Fraud count
        fraud_counts = transactions_df['is_fraud'].value_counts()
        axes[0, 0].bar(['Legitimate', 'Fraudulent'], fraud_counts.values, 
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('Transaction Distribution', fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        
        # Amount distribution
        axes[0, 1].hist([transactions_df[transactions_df['is_fraud']==0]['amount'],
                        transactions_df[transactions_df['is_fraud']==1]['amount']],
                       label=['Legitimate', 'Fraudulent'], bins=30, alpha=0.7,
                       color=['green', 'red'])
        axes[0, 1].set_title('Transaction Amount Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Amount ($)')
        axes[0, 1].legend()
        
        # Hourly distribution
        hour_fraud = transactions_df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0)
        hour_fraud.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'], alpha=0.7)
        axes[1, 0].set_title('Transactions by Hour', fontweight='bold')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(['Legitimate', 'Fraudulent'])
        
        # Fraud percentage by hour
        fraud_pct = transactions_df.groupby('hour')['is_fraud'].mean() * 100
        axes[1, 1].plot(fraud_pct.index, fraud_pct.values, 'o-', color='red', linewidth=2)
        axes[1, 1].set_title('Fraud Percentage by Hour', fontweight='bold')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Fraud %')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fraud_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: fraud_analysis.png")
        plt.show()
    
    @staticmethod
    def plot_training_history(train_losses):
        """Plot training loss over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: training_history.png")
        plt.show()

#============================================================================
# 5. MAIN EXECUTION
#============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("IMPROVED FRAUD DETECTION SYSTEM - GNN + Neo4j")
    print("="*60 + "\n")
    
    # Step 1: Generate Data
    print("STEP 1: Generating synthetic transaction data...")
    generator = FraudDataGenerator(n_users=500, n_merchants=100, n_transactions=5000)
    users_df = generator.generate_users()
    merchants_df = generator.generate_merchants()
    transactions_df = generator.generate_transactions(users_df, merchants_df)
    
    print(f"✓ Generated {len(users_df)} users")
    print(f"✓ Generated {len(merchants_df)} merchants")
    print(f"✓ Generated {len(transactions_df)} transactions")
    print(f"✓ Fraud rate: {transactions_df['is_fraud'].mean()*100:.2f}%\n")
    
    # Step 2: Load into Neo4j
    print("STEP 2: Loading data into Neo4j...")
    print("(Make sure Neo4j is running!)")
    
    neo4j = Neo4jConnector(
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="password-here"
    )
    
    try:
        neo4j.clear_database()
        neo4j.create_constraints()
        neo4j.load_users(users_df)
        neo4j.load_merchants(merchants_df)
        neo4j.load_transactions(transactions_df)
        print()
        
        # Step 3: Extract graph data
        print("STEP 3: Extracting graph data from Neo4j...")
        users, merchants, transactions = neo4j.get_graph_data()
        print(f"✓ Extracted {len(users)} users, {len(merchants)} merchants, {len(transactions)} transactions\n")
        
    finally:
        neo4j.close()
    
    # Step 4: Train Improved GNN
    print("STEP 4: Training Improved Graph Neural Network...")
    trainer = ImprovedGNNTrainer(users, merchants, transactions)
    trainer.train_model(epochs=150, lr=0.005)
    print()
    
    # Step 5: Evaluate Model
    print("STEP 5: Evaluating model performance...")
    y_true, y_pred, y_probs, cm = trainer.evaluate_model()
    
    # Step 6: Create Visualizations
    print("\nSTEP 6: Creating visualizations...")
    visualizer = FraudVisualizer()
    visualizer.plot_confusion_matrix(cm)
    visualizer.plot_roc_curve(y_true, y_probs)
    visualizer.plot_fraud_distribution(transactions_df)
    visualizer.plot_training_history(trainer.train_losses)
    
    # Step 7: Summary Statistics
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    tn, fp, fn, tp = cm.ravel()
    print(f"Total Transactions Processed: {len(transactions_df)}")
    print(f"Fraudulent Transactions: {sum(y_true)} ({sum(y_true)/len(y_true)*100:.1f}%)")
    print(f"Correctly Identified Frauds: {tp}")
    print(f"Missed Frauds (False Negatives): {fn}")
    print(f"False Alarms (False Positives): {fp}")
    print(f"Correctly Identified Legitimate: {tn}")
    print(f"\nModel Performance:")
    print(f"  Overall Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")
    print(f"  Fraud Detection Rate (Recall): {tp/(tp+fn)*100:.2f}%")
    print(f"  Precision: {tp/(tp+fp)*100:.2f}%")
    print(f"  AUC-ROC Score: {roc_auc_score(y_true, y_probs):.4f}")
    print("="*60)
    print("\n✅ Demo completed successfully!")
    print("Check the generated PNG files for visualizations.\n")

if __name__ == "__main__":
    main()