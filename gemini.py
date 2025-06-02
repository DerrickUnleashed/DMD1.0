import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import json  # To load the GO terms from the file

l = ["GSE38417","GSE6011","GSE19303","GSE42955"]
for i in l:

# --- 1. Load and Analyze the Dataset ---
    csv_path = 'Dataset_'+i+'.csv'
    df = pd.read_csv(csv_path)
    print(f"Dataset_{i} shape: {df.shape}")
    #print(f"Columns: {df.columns.tolist()}")

    # --- 2. Load DMD-Related Information (from text file) ---
    def load_dmd_data(go_terms_file=i+"_extracted.txt"):
        """Loads GO terms and DMD-related gene symbols/keywords from a file."""
        with open(go_terms_file, 'r') as f:
            loaded_data = json.load(f)

        dmd_related_gene_symbols = ['DMD','DCM']  # Base symbol, add more as needed
        dmd_related_keywords = ['Duchenne Muscular Dystrophy', 'dystrophin', 'muscle weakness', 'muscle degeneration']  # Base keywords, expand
        relevant_go_terms = loaded_data
        return dmd_related_gene_symbols, dmd_related_keywords, relevant_go_terms

    dmd_gene_symbols, dmd_keywords, go_terms = load_dmd_data()


    # --- 3. Feature Engineering ---
    node_features = []
    node_labels = []

    for index, row in df.iterrows():
        features = []

        # Feature 1: Presence of DMD-related keywords in Gene Title
        gene_title = str(row['Gene title']).lower()
        has_dmd_keyword = any(keyword in gene_title for keyword in dmd_keywords)
        features.append(float(has_dmd_keyword))

        # Feature 2 & 3 & 4: Encode GO Terms
        go_function = str(row['GO:Function'])
        go_process = str(row['GO:Process'])
        go_component = str(row['GO:Component'])

        features.append(float(any(term in go_function for term in go_terms['Function'])))
        features.append(float(any(term in go_process for term in go_terms['Process'])))
        features.append(float(any(term in go_component for term in go_terms['Component'])))

        node_features.append(features)

        # Pseudo-label
        gene_symbol = str(row['Gene symbol']).upper()
        label = 1 if gene_symbol in dmd_gene_symbols else 0
        node_labels.append(label)

    # Convert features and labels to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    num_nodes = len(df)

    # --- 4. Graph Construction (Simple Sequential Graph) ---
    edge_index = torch.tensor([[i for i in range(num_nodes - 1)], [i + 1 for i in range(num_nodes - 1)]], dtype=torch.long)

    # --- 5. Create Train/Test Mask ---
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_size = int(0.8 * num_nodes)
    train_mask[:train_size] = True
    test_mask[train_size:] = True

    # --- 6. Build Graph Data Object ---
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data = data.to(device)

    # --- 7. Define GCN Model ---
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    # --- 8. Initialize Model, Optimizer, and Loss ---
    in_channels = x.shape[1]
    hidden_channels = 16
    out_channels = 2
    model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()

    # --- 9. Training Loop ---
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    # --- 10. Testing Loop ---
    def test():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
        return acc

    # --- 11. Run Training ---
    for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0:
            acc = test()
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

    # --- 12. Analyze Results ---
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
        predictions = embeddings.argmax(dim=1)
        test_predictions = predictions[data.test_mask]
        test_labels = data.y[data.test_mask]
        print(f"\nTest Predictions for {i}:", test_predictions)
        print("Test Labels:", test_labels)

        dmd_related_indices = (predictions == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        dmd_related_genes = df.iloc[dmd_related_indices]['Gene symbol'].tolist()
        print(f"\nPotentially DMD-Related Genes for {i}:", dmd_related_genes)