import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Data Import and Preprocessing


class RNASeqDataset(Dataset):
    def __init__(self, expression_file, labels_file=None, normalize=True):
        self.expression_df = pd.read_csv(expression_file, index_col=0)
        self.labels_df = pd.read_csv(labels_file, index_col=0) if labels_file else None

        if self.labels_df is not None:
            self.classes = self.labels_df.iloc[:, 0].unique().tolist()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.labels_df.iloc[:, 0] = self.labels_df.iloc[:, 0].map(self.class_to_idx)

        if normalize:
            self.normalize_data()

    def __len__(self):
        return len(self.expression_df)

    def __getitem__(self, idx):
        expression = torch.tensor(
            self.expression_df.iloc[idx].values, dtype=torch.float32
        )

        if self.labels_df is not None:
            label = torch.tensor(
                self.labels_df.iloc[idx].values, dtype=torch.long
            ).squeeze()
        else:
            label = None

        return expression, label

    def normalize_data(self):
        scaler = StandardScaler()
        self.expression_df = pd.DataFrame(
            scaler.fit_transform(self.expression_df),
            index=self.expression_df.index,
            columns=self.expression_df.columns,
        )


# Example Data (Replace with your actual data)
num_samples = 100
num_genes = 200
expression_data = np.random.rand(num_samples, num_genes)
labels = np.random.choice(["Disease", "Control"], num_samples)
expression_df = pd.DataFrame(
    expression_data, columns=[f"Gene_{i}" for i in range(num_genes)]
)
labels_df = pd.DataFrame(labels, columns=["Label"])
expression_df.to_csv("expression.csv")
labels_df.to_csv("labels.csv")

dataset = RNASeqDataset("expression.csv", "labels.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Autoencoder for Feature Selection


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),  # Bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded  # Return both decoded output and bottleneck features


input_dim = num_genes  # Number of genes
hidden_dim = 128
bottleneck_dim = 64  # Reduced feature dimension

autoencoder = Autoencoder(input_dim, hidden_dim, bottleneck_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    for expressions, _ in dataloader:  # No labels needed for autoencoder
        optimizer.zero_grad()
        outputs, _ = autoencoder(expressions)
        loss = criterion(outputs, expressions)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Extract bottleneck features
bottleneck_features = []
with torch.no_grad():
    for expressions, _ in dataloader:
        _, encoded = autoencoder(expressions)
        bottleneck_features.extend(encoded.numpy())  # Store as NumPy arrays
bottleneck_features = np.array(bottleneck_features)

# Convert bottleneck_features to a DataFrame (optional, but good practice)
bottleneck_df = pd.DataFrame(bottleneck_features)
print("Shape of bottleneck features:", bottleneck_df.shape)

# 3. Transformer Model (Example)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, nhead, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, input_dim)  # Linear embedding layer
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
        )
        self.fc = nn.Linear(
            input_dim, num_classes
        )  # Fully connected layer for classification

    def forward(self, src):
        embedded = self.embedding(src)  # Linear embedding
        # Transformer expects specific input shapes (Seq_len, Batch, Embedding_dim)
        # Assuming your input is (Batch, Seq_len, Embedding_dim) or (Batch, Embedding_dim)
        # If your input is (Batch, Embedding_dim) you can add a sequence dimension:
        embedded = embedded.unsqueeze(1)  # Add a sequence dimension of 1
        tgt = torch.zeros_like(embedded)  # Dummy target for the transformer encoder
        output = self.transformer(embedded, tgt)
        output = output.squeeze(1)  # Remove the sequence dimension
        output = self.fc(output)
        return output


# Example Transformer Usage (Classification)

num_classes = (
    len(dataset.classes) if dataset.labels_df is not None else 0
)  # Number of classes
nhead = 8  # Number of attention heads
num_layers = 2  # Number of transformer layers
dropout = 0.1  # Dropout rate

transformer_model = TransformerModel(
    bottleneck_dim, num_classes, nhead, num_layers, dropout
)

# ... (Rest of your training loop for the Transformer model) ...

# Example: Training loop for classification (if you have labels)
if dataset.labels_df is not None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

    # Convert bottleneck features to tensors and create a new DataLoader
    bottleneck_dataset = torch.utils.data.TensorDataset(
        torch.tensor(bottleneck_features, dtype=torch.float32),
        torch.tensor(dataset.labels_df.iloc[:, 0].values, dtype=torch.long),
    )
    bottleneck_dataloader = DataLoader(bottleneck_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):  # Assuming you have defined num_epochs before
        for inputs, labels in bottleneck_dataloader:
            optimizer.zero_grad()
            outputs = transformer_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


#! DeepSeek
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Data Import & Preprocessing
# --------------------------

# Load RNA expression matrix (rows = samples, columns = genes)
data = pd.read_csv("gene_expression.csv")  # Replace with your data path
labels = data["label"].values  # Assuming a "label" column exists for downstream tasks
features = data.drop("label", axis=1).values

# Normalize data
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features_normalized, labels, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)  # Adjust dtype for regression tasks

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --------------------------
# 2. Autoencoder for Feature Selection
# --------------------------


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# Hyperparameters
input_dim = X_train.shape[1]
latent_dim = 64  # Bottleneck layer (selected features)
lr = 0.001
epochs = 50

# Initialize model, loss, optimizer
autoencoder = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

# Train Autoencoder
autoencoder.train()
for epoch in range(epochs):
    for batch_X, _ in train_loader:  # Unsupervised (no labels needed)
        optimizer.zero_grad()
        reconstructed, latent = autoencoder(batch_X)
        loss = criterion(reconstructed, batch_X)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Extract latent features (selected features)
autoencoder.eval()
with torch.no_grad():
    X_train_latent = autoencoder.encoder(X_train_tensor)
    X_test_latent = autoencoder.encoder(X_test_tensor)

# --------------------------
# 3. Transformer Model
# --------------------------


class GeneTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2):
        super(GeneTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, 1, d_model)
        )  # Learnable positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len=1, d_model)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x


# Hyperparameters
num_classes = len(np.unique(labels))  # Adjust for your task
d_model = 64  # Embedding dimension
nhead = 4  # Number of attention heads
num_layers = 2

# Initialize Transformer
transformer = GeneTransformer(latent_dim, num_classes, d_model, nhead, num_layers)
criterion = nn.CrossEntropyLoss()  # Use appropriate loss for your task
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# Train Transformer on latent features
transformer.train()
for epoch in range(30):
    for batch_X, batch_y in train_loader:
        latent_X = autoencoder.encoder(batch_X)  # Use pre-trained encoder
        outputs = transformer(latent_X.unsqueeze(1))  # Add dummy sequence dimension
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/30], Loss: {loss.item():.4f}")

# --------------------------
# Evaluation (Example)
# --------------------------
transformer.eval()
with torch.no_grad():
    test_outputs = transformer(X_test_latent.unsqueeze(1))
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")
