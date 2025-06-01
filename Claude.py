import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print(torch.__version__)
print(torch.backends.mps.is_available())
device = torch.device("mps")

class DMDGeneDataset:
    """
    Dataset class for processing gene expression data for DMD detection
    """
    def __init__(self, csv_file_path, correlation_threshold=0.7):
        self.csv_file_path = csv_file_path
        self.correlation_threshold = correlation_threshold
        self.data = None
        self.gene_features = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the gene expression data"""
        # Load the dataset
        self.data = pd.read_csv(self.csv_file_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Extract relevant features for DMD detection
        # Focus on gene expression and functional annotations
        feature_columns = [
            'Gene ID', 'Gene symbol', 'Chromosome location',
            'GO:Function', 'GO:Process', 'GO:Component'
        ]
        
        # Create synthetic expression data and DMD labels for demonstration
        # In real scenario, you would have actual expression values and DMD diagnosis
        np.random.seed(42)
        n_samples = len(self.data)
        
        # Create synthetic gene expression matrix
        self.gene_features = np.random.randn(n_samples, 100)  # 100 expression features
        
        # Create synthetic DMD labels (0: healthy, 1: DMD positive)
        # In practice, this would come from clinical diagnosis
        self.labels = np.random.binomial(1, 0.3, n_samples)  # 30% DMD positive
        
        # Add some correlation between certain genes and DMD status
        dmd_related_genes = [0, 5, 10, 15, 20]  # Indices of DMD-related genes
        for gene_idx in dmd_related_genes:
            self.gene_features[:, gene_idx] += self.labels * 2 + np.random.randn(n_samples) * 0.5
        
        # Normalize features
        self.gene_features = self.scaler.fit_transform(self.gene_features)
        
        return self.gene_features, self.labels
    
    def create_gene_interaction_graph(self):
        """Create gene interaction graph based on correlation"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(self.gene_features.T)
        
        # Create adjacency matrix based on correlation threshold
        adj_matrix = (np.abs(corr_matrix) > self.correlation_threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
        
        # Convert to edge indices for PyTorch Geometric
        edge_indices = np.where(adj_matrix == 1)
        edge_index = torch.tensor(np.array([edge_indices[0], edge_indices[1]]), dtype=torch.long)
        
        # Create edge weights based on correlation strength
        edge_weights = []
        for i, j in zip(edge_indices[0], edge_indices[1]):
            edge_weights.append(abs(corr_matrix[i, j]))
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr, adj_matrix

class DMDGraphNet(nn.Module):
    """
    Graph Neural Network for DMD detection using gene expression data
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.5):
        super(DMDGraphNet, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 4, heads=4, dropout=dropout)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolution layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        
        # Attention mechanism
        x_att = self.attention(x3, edge_index)
        x_att = F.relu(x_att)
        x_att = self.dropout(x_att)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            x_pool = global_mean_pool(x_att, batch)
        else:
            x_pool = torch.mean(x_att, dim=0, keepdim=True)
        
        # Classification
        out = self.classifier(x_pool)
        
        return out

class DMDDetectionPipeline:
    """
    Complete pipeline for DMD detection using GNN
    """
    def __init__(self, csv_file_path, correlation_threshold=0.7):
        self.dataset = DMDGeneDataset(csv_file_path, correlation_threshold)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        """Prepare data for training"""
        # Load and preprocess data
        features, labels = self.dataset.load_and_preprocess_data()
        
        # Create gene interaction graph
        edge_index, edge_attr, adj_matrix = self.dataset.create_gene_interaction_graph()
        
        # Convert to PyTorch tensors
        x = torch.tensor(features.T, dtype=torch.float)  # Transpose for gene-wise features
        y = torch.tensor(labels, dtype=torch.long)
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        return data, adj_matrix
    
    def train_model(self, data, epochs=200, lr=0.01):
        """Train the GNN model"""
        # Initialize model
        self.model = DMDGraphNet(
            num_features=data.x.shape[1],
            hidden_dim=64,
            num_classes=2,
            dropout=0.5
        ).to(self.device)
        
        # Move data to device
        data = data.to(self.device)
        
        # Split data for training and validation
        num_nodes = data.x.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Create masks for train/val/test split
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data.x, data.edge_index)
            
            # Calculate loss only on training nodes
            loss = criterion(out[train_mask], data.y[train_mask])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(data.x, data.edge_index)
                    val_pred = val_out[val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[val_mask]).float().mean()
                    val_accuracies.append(val_acc.item())
                    
                train_losses.append(loss.item())
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
                self.model.train()
        
        return train_losses, val_accuracies, (train_mask, val_mask, test_mask)
    
    def evaluate_model(self, data, test_mask):
        """Evaluate the trained model"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            test_pred = out[test_mask].argmax(dim=1)
            test_prob = F.softmax(out[test_mask], dim=1)[:, 1]  # Probability of DMD
            
            # Calculate metrics
            test_acc = (test_pred == data.y[test_mask]).float().mean()
            
            # Convert to numpy for sklearn metrics
            y_true = data.y[test_mask].cpu().numpy()
            y_pred = test_pred.cpu().numpy()
            y_prob = test_prob.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            auc = roc_auc_score(y_true, y_prob)
            
            print(f"\nTest Results:")
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
            
            return {
                'accuracy': test_acc.item(),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_prob,
                'true_labels': y_true
            }
    
    def predict_dmd_risk(self, patient_features):
        """Predict DMD risk for new patient"""
        self.model.eval()
        with torch.no_grad():
            # Assuming patient_features is a single sample
            # In practice, you'd need to construct a graph for the new patient
            patient_tensor = torch.tensor(patient_features, dtype=torch.float).unsqueeze(0)
            
            # For simplicity, using a dummy edge_index for single node
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            out = self.model(patient_tensor.T, edge_index)
            prob = F.softmax(out, dim=1)[0, 1].item()  # Probability of DMD
            
            return prob
    
    def visualize_results(self, train_losses, val_accuracies, test_results, adj_matrix):
        """Visualize training results and network structure"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training loss
        axes[0, 0].plot(range(0, len(train_losses) * 10, 10), train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Validation accuracy
        axes[0, 1].plot(range(0, len(val_accuracies) * 10, 10), val_accuracies)
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_results['true_labels'], test_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Gene interaction network (sample)
        # Show only a subset of genes for visualization
        subset_size = min(20, adj_matrix.shape[0])
        subset_adj = adj_matrix[:subset_size, :subset_size]
        
        G = nx.from_numpy_array(subset_adj)
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        axes[1, 1].clear()
        nx.draw(G, pos, ax=axes[1, 1], node_size=300, node_color='lightblue', 
                with_labels=True, font_size=8, edge_color='gray', alpha=0.7)
        axes[1, 1].set_title('Gene Interaction Network (Sample)')
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def run_dmd_detection_pipeline(csv_file_path):
    """
    Run the complete DMD detection pipeline
    """
    print("=== DMD Detection using Graph Neural Networks ===\n")
    
    # Initialize pipeline
    pipeline = DMDDetectionPipeline(csv_file_path, correlation_threshold=0.7)
    
    # Prepare data
    print("1. Preparing data...")
    data, adj_matrix = pipeline.prepare_data()
    print(f"   Graph created with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    
    # Train model
    print("\n2. Training GNN model...")
    train_losses, val_accuracies, masks = pipeline.train_model(data, epochs=100, lr=0.01)
    train_mask, val_mask, test_mask = masks
    
    # Evaluate model
    print("\n3. Evaluating model...")
    test_results = pipeline.evaluate_model(data, test_mask)
    
    # Visualize results
    print("\n4. Visualizing results...")
    pipeline.visualize_results(train_losses, val_accuracies, test_results, adj_matrix)
    
    # Example prediction for new patient
    print("\n5. Example prediction for new patient:")
    dummy_patient_features = np.random.randn(100)  # 100 gene expression features
    dmd_risk = pipeline.predict_dmd_risk(dummy_patient_features)
    print(f"   DMD Risk Probability: {dmd_risk:.4f}")
    if dmd_risk > 0.5:
        print("   Classification: HIGH RISK for DMD")
    else:
        print("   Classification: LOW RISK for DMD")
    
    return pipeline, test_results

# Clinical interpretation function
def interpret_dmd_results(test_results, threshold=0.5):
    """
    Provide clinical interpretation of DMD detection results
    """
    print("\n=== Clinical Interpretation ===")
    
    high_risk_count = sum(test_results['probabilities'] > threshold)
    total_patients = len(test_results['probabilities'])
    
    print(f"Total patients analyzed: {total_patients}")
    print(f"High-risk patients (>{threshold:.1f} probability): {high_risk_count}")
    print(f"Low-risk patients: {total_patients - high_risk_count}")
    print(f"High-risk percentage: {(high_risk_count/total_patients)*100:.1f}%")
    
    # Risk stratification
    very_high_risk = sum(test_results['probabilities'] > 0.8)
    moderate_risk = sum((test_results['probabilities'] > 0.3) & 
                       (test_results['probabilities'] <= 0.8))
    low_risk = sum(test_results['probabilities'] <= 0.3)
    
    print(f"\nRisk Stratification:")
    print(f"Very High Risk (>0.8): {very_high_risk} patients")
    print(f"Moderate Risk (0.3-0.8): {moderate_risk} patients")
    print(f"Low Risk (≤0.3): {low_risk} patients")
    
    print(f"\nModel Performance Summary:")
    print(f"Sensitivity (Recall): {test_results['recall']:.3f}")
    print(f"Specificity: {1 - test_results['recall']:.3f}")
    print(f"Positive Predictive Value: {test_results['precision']:.3f}")
    print(f"Overall Accuracy: {test_results['accuracy']:.3f}")

# Run the pipeline (example with the provided dataset)
if __name__ == "__main__":
    # Note: Replace with actual path to your CSV file
    csv_file_path = "Dataset.csv"
    
    try:
        pipeline, results = run_dmd_detection_pipeline(csv_file_path)
        interpret_dmd_results(results)
        
        print("\n=== Pipeline completed successfully! ===")
        print("The GNN model has been trained and evaluated for DMD detection.")
        print("In a real clinical setting, this would be integrated with:")
        print("- Electronic Health Records (EHR)")
        print("- Laboratory Information Systems")
        print("- Clinical Decision Support Systems")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        print("Please ensure the CSV file path is correct and the file is accessible.")