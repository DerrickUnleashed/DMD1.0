import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load dataset
csv_path = 'Dataset.csv'
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")

# Create a simple graph with dummy edges
num_nodes = 100
edge_index = torch.tensor([[i for i in range(num_nodes - 1)], [i + 1 for i in range(num_nodes - 1)]], dtype=torch.long)

# Dummy node features (let's take two features per node)
x = torch.randn((num_nodes, 2), dtype=torch.float)

# Dummy labels
y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

# Train/test mask
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:int(0.8 * num_nodes)] = True
test_mask[int(0.8 * num_nodes):] = True

# Build graph data object
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
data = data.to(device)

# Define GCN model
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

# Initialize model, optimizer, and loss
model = GCN(in_channels=2, hidden_channels=16, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing loop
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

# Run training
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
