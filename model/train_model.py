import torch
from torch.utils.data import DataLoader, Dataset
from sequence_classifier import SequenceClassifier
import numpy as np

class ShopliftingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

X = np.load('data/features/X.npy')
y = np.load('data/features/y.npy')

model = SequenceClassifier()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(ShopliftingDataset(X, y), batch_size=8, shuffle=True)

for epoch in range(10):
    for inputs, labels in loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'model/shoplifting_lstm.pth')
