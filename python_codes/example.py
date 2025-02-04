# Example Usage
input_dim = 5000  # Number of genes
num_classes = 2  # Classification categories (e.g., Tumor vs Normal)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Model
model = TransformerClassifier(input_dim, num_classes)

# Example Dummy Data (Modify with real TCGA-SKCM Data)
X_train = torch.randn(100, input_dim)  # 100 samples, each with 5000 gene expressions
y_train = torch.randint(0, num_classes, (100,))
dataset = RNASeqDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
train_model(model, dataloader, criterion, optimizer, device)

# Test the Model
test_model(model, dataloader, device)
