# pip install numpy torch scikit-learn matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def collatz_stopping_time(n):
    """Calculate the stopping time for a given number in the Collatz sequence."""
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

def create_features(n):
    """Create features for the number n that might be relevant for predicting stopping time."""
    return np.array([
        n,                   # The number itself
        np.log2(n),          # Logarithm base 2
        np.log(n),           # Natural logarithm
        np.log10(n),         # Logarithm base 10
        np.log2(n)/4,        # Approximation of log base 16
        np.log2(n)/5,        # Approximation of log base 32
        np.log2(n)/6,        # Approximation of log base 64
        np.log2(n)/7,        # Approximation of log base 128
        np.log2(n)/8,        # Approximation of log base 256
        np.log2(n)/9,        # Approximation of log base 512
        np.log2(n)/10,       # Approximation of log base 1024
        n % 2,               # Parity
        n % 4,               # Modulo 4
        n % 8,               # Modulo 8
        n % 16,              # Modulo 16
        bin(n).count('1'),   # Number of 1s in binary representation
        len(bin(n)) - 2,     # Length of binary representation
        len(str(n))          # Length of decimal representation
    ])

def generate_training_data(max_n=1000000):
    """Generate training data for numbers up to max_n."""
    X = []
    y = []
    
    for n in range(1, max_n + 1):
        X.append(create_features(n))
        y.append(collatz_stopping_time(n))
    
    return np.array(X), np.array(y)

# Define CNN model using PyTorch
class CollatzCNN(nn.Module):
    def __init__(self):
        super(CollatzCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 2, 256)  # Adjusted for 3 pooling operations
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn6(x)
        x = self.leaky_relu(x)
        
        x = self.fc4(x)
        return x

def main():
    # Generate training data
    print("Generating training data...")
    X, y = generate_training_data(max_n=1000000)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create data loaders with larger batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Build CNN model
    print("Building CNN model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CollatzCNN().to(device)
    
    # Print model summary
    print(model)
    
    # Define loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Train model
    print("Training CNN model...")
    epochs = 50  # Increased epochs
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_mae += torch.sum(torch.abs(outputs - targets)).item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_mae = running_mae / len(train_dataset)
        train_losses.append(epoch_loss)
        train_maes.append(epoch_mae)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_mae += torch.sum(torch.abs(outputs - targets)).item()
        
        val_epoch_loss = val_running_loss / len(test_dataset)
        val_epoch_mae = val_running_mae / len(test_dataset)
        val_losses.append(val_epoch_loss)
        val_maes.append(val_epoch_mae)
        
        # Learning rate scheduling
        scheduler.step(val_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'current_lr': current_lr
            }, 'best_collatz_cnn.pth')
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val MAE: {val_epoch_mae:.4f}, '
              f'LR: {current_lr:.6f}')
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_collatz_cnn.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['best_val_loss']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_maes)
    plt.plot(val_maes)
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    
    # Make predictions on test data
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    # Calculate additional evaluation metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Calculate R² score
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Stopping Time')
    plt.ylabel('Predicted Stopping Time')
    plt.title('CNN: Actual vs Predicted Stopping Times')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cnn_actual_vs_predicted.png')
    
    # Plot predictions vs real stopping times with x as input
    plt.figure(figsize=(12, 8))
    
    # Generate a range of numbers to plot (using logarithmic spacing for 1e6 range)
    plot_range = np.unique(np.logspace(0, 6, num=1000, dtype=int))
    
    # Calculate actual stopping times
    print("Calculating actual stopping times for large range...")
    actual_stopping_times = [collatz_stopping_time(n) for n in plot_range]
    
    # Predict stopping times
    print("Predicting stopping times for large range...")
    predicted_stopping_times = []
    model.eval()
    with torch.no_grad():
        for n in plot_range:
            features = create_features(n)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
            predicted = model(features_tensor).item()
            predicted_stopping_times.append(predicted)
    
    # Plot both curves
    plt.plot(plot_range, actual_stopping_times, 'b-', alpha=0.7, label='Actual Stopping Times')
    plt.plot(plot_range, predicted_stopping_times, 'r-', alpha=0.7, label='Predicted Stopping Times')
    
    plt.xlabel('Number (log scale)')
    plt.ylabel('Stopping Time')
    plt.title('Collatz Stopping Times: CNN Actual vs Predicted (up to 1e6)')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cnn_stopping_times_comparison.png')

if __name__ == "__main__":
    main()
