import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Ensure matplotlib uses a backend compatible with this environment
import matplotlib
#matplotlib.use('Agg')  # Use a non-interactive backend for environments without GUI support

# Update the StockDataset class to handle 60-step-ahead predictions
class StockDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_steps=1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_steps + 1

    def __getitem__(self, idx):
        # Input sequence (x[t] .. x[t+seq_len-1])
        x = self.data[idx:idx + self.sequence_length, :-1]

        # One-step-ahead targets (y[t+1] .. y[t+seq_len])
        y = self.data[idx + self.prediction_steps:idx + self.sequence_length + self.prediction_steps, -1]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class StockDatasetNoOverlap(Dataset):
    def __init__(self, data, sequence_length, prediction_steps=1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.stride = sequence_length + prediction_steps

        self.total_sequences = (len(data) - self.stride) // self.stride + 1

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.sequence_length
        x = self.data[start:end, :-1]
        y = self.data[start + self.prediction_steps:end + self.prediction_steps, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class ConvTransformerDecoder(nn.Module):
    def __init__(self, input_dim, conv_channels, seq_len, d_model, nhead, num_layers):
        super(ConvTransformerDecoder, self).__init__()
        self.kernel_size = 11
        self.conv1d = nn.Conv1d(input_dim, conv_channels, kernel_size=self.kernel_size, stride=1, padding=0)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=.2)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        # Input shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len)
        padding = self.kernel_size - 1
        x = F.pad(x, (padding, 0))  # (left_pad, right_pad)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # -> (batch, seq_len, conv_channels)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # -> (seq_len, batch, d_model)

        mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)
        # Pass it to both tgt_mask _and_ memory_mask:
        x = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=mask,
            memory_mask=mask
        )
        x = self.fc(x).squeeze(-1)        # -> (seq_len, batch)
        return x.permute(1, 0)            # -> (batch, seq_len)


# Load and preprocess the stocks.csv data
stock_data = pd.read_csv('stocks.csv')[:]

# Ensure the ts_event column is in datetime format
stock_data['ts_event'] = pd.to_datetime(stock_data['ts_event'], unit='ns')

# Update sequence length and prediction steps
prediction_steps = 10

# Add a pct_change column to the data before normalization
stock_data['pct_change'] = stock_data['open'].pct_change()
stock_data['vol_pct_change'] = stock_data['volume'].pct_change()
stock_data['pct_change_pred'] = stock_data['open'].pct_change(periods=prediction_steps)
stock_data = stock_data.dropna()  # Drop rows with NaN values resulting from pct_change calculation

# Add time delta as a feature to the dataset
stock_data['time_delta'] = stock_data['ts_event'].diff().dt.total_seconds()
stock_data['time_delta'] = stock_data['time_delta'].fillna(0)  # Fill NaN values with 0

# Add a column for the delta between the current time and the time prediction_steps ahead
stock_data['future_time'] = stock_data['ts_event'].shift(-prediction_steps)
stock_data['time_delta_future'] = (stock_data['future_time'] - stock_data['ts_event']).dt.total_seconds()
stock_data = stock_data.drop(columns=['future_time'])  # Drop the intermediate column
stock_data = stock_data.dropna()  # Drop rows with NaN values resulting from the shift operation

print(stock_data['pct_change_pred'].describe())

# Winsorize pct_change_pred to the 0.5th and 99.5th percentiles instead of dropping rows
pct_0_5 = stock_data['pct_change_pred'].quantile(0.005)
pct_99_5 = stock_data['pct_change_pred'].quantile(0.995)
rows_before = len(stock_data)
stock_data['pct_change_pred'] = stock_data['pct_change_pred'].clip(lower=pct_0_5, upper=pct_99_5)
rows_after = len(stock_data)
print(f"Winsorized pct_change_pred to the 0.5th and 99.5th percentiles. Rows dropped: {rows_before - rows_after}")

print(stock_data['pct_change_pred'].describe())


# Check for rows with NaN values in time_delta_future
nan_rows = stock_data[stock_data['time_delta_future'].isna()]
print("\nRows with NaN values in time_delta_future:")
print(nan_rows)

# Select relevant columns for features and target
# Remove 'close' from input features
stock_data = stock_data.sort_values('ts_event')
features = stock_data[['open', 'close', 'high', 'low', 'volume','vol_pct_change', 'pct_change', 'time_delta_future']].values  # NO 'close'
labels = stock_data['pct_change_pred'].values.reshape(-1, 1)

# Now concatenate inputs + labels
data = torch.cat((torch.tensor(features, dtype=torch.float32),
                  torch.tensor(labels, dtype=torch.float32)), dim=1)


# Split the data first to avoid data leakage during normalization
train_data, val_data = train_test_split(data.numpy(), test_size=0.2, random_state=42, shuffle=False)

# Normalize training data
scaler = StandardScaler()
train_data_normalized = scaler.fit_transform(train_data)

# Normalize validation data using the same scaler
val_data_normalized = scaler.transform(val_data)

# Convert back to tensors
train_data = torch.tensor(train_data_normalized, dtype=torch.float32)
val_data = torch.tensor(val_data_normalized, dtype=torch.float32)

# Hyperparameters
sequence_length = 512
batch_size = 8
# Update input_dim to match the number of features
input_dim = features.shape[1]  # Number of input features
conv_channels = 64
d_model = 64
nhead = 1
num_layers = 4


# Update datasets and dataloaders
train_dataset = StockDatasetNoOverlap(train_data, sequence_length, prediction_steps)
val_dataset = StockDatasetNoOverlap(val_data, sequence_length, prediction_steps)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Ensure model weights are initialized only once
if 'model_initialized' not in globals():
    model_initialized = True
    model = ConvTransformerDecoder(input_dim, conv_channels, sequence_length, d_model, nhead, num_layers)

# Verify data loader consistency
for i, (x_batch, y_batch) in enumerate(train_dataloader):
    print(f"Batch {i}: x_batch mean={x_batch.mean().item()}, y_batch mean={y_batch.mean().item()}")
    break  # Print only the first batch for verification

for i, (x_batch, y_batch) in enumerate(train_dataloader):
    print(f"Batch {i}: x_batch mean={x_batch.mean().item()}, y_batch mean={y_batch.mean().item()}")
    break  # Print only the first batch for verification
# Initialize model
model = ConvTransformerDecoder(input_dim, conv_channels, sequence_length, d_model, nhead, num_layers)

from loss_func import smooth_custom_loss

# Loss and optimizer
#criterion = nn.L1Loss()

# Replace the criterion definition with the custom loss function
criterion = smooth_custom_loss

# Adjust the learning rate to a smaller value for better convergence
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)  # Reduced learning rate

from torch.optim.lr_scheduler import LambdaLR

# Define a learning rate schedule with warmup and inverse square root decay
def get_lr_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return (warmup_steps ** 0.5) * (current_step ** -0.5)

    return LambdaLR(optimizer, lr_lambda)

# Initialize the learning rate scheduler
warmup_steps = 1000
# Define the number of epochs
epochs = 100

scheduler = get_lr_schedule(optimizer, warmup_steps, epochs * len(train_dataloader))

import matplotlib.pyplot as plt

# Initialize lists to store loss and validation loss values
train_losses = []
val_losses = []
lr_history = []

try:
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader, start=1):
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step the scheduler after each batch
            train_loss += loss.item()
            
            current_lr = scheduler.get_last_lr()[0]
            lr_history.append(current_lr)

            # Record the training loss for each step
            train_losses.append(loss.item())

            # Display the current learning rate and mean batch loss
            batch_progress = f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, "
            debug_info = f"Mean Batch Loss: {train_loss / batch_idx:.4f}, Current LR: {current_lr:.6f}"
            print(f"\r{batch_progress}{debug_info}", end="")

        # Print a new line at the end of the epoch
        print()

        # Validation phase at the end of the epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                val_predictions = model(x_val)
                val_loss += criterion(val_predictions, y_val).item()
        val_loss /= len(val_dataloader)

        # Extend val_losses to match the number of training steps in this epoch
        val_losses.extend([val_loss] * len(train_dataloader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}")
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Displaying the loss and learning rate curves...")
finally:
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # plot losses on ax1
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses,   label='Validation Loss')
    ax1.set_xlabel('Gradient Steps')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    # create a second y‑axis that shares the x-axis
    ax2 = ax1.twinx()
    ax2.plot(lr_history, label='Learning Rate', linestyle='--', color='green')
    ax2.set_ylabel('Learning Rate')
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Loss with Learning Rate Schedule')
    plt.show()

# Perform a line search for the best constant output

best_loss = float('inf')
best_constant = None
    
constants = np.linspace(-1, 1, 100)  # Increase granularity for better search
for constant in constants:
    constant_tensor = torch.tensor([constant], dtype=torch.float32)
    loss = 0.0
    
    for _, y_batch in train_dataloader:
        # Create a new constant tensor with the correct batch size for each batch
        constant_tensor = torch.full_like(y_batch, constant, dtype=torch.float32)
        loss += criterion(constant_tensor, y_batch).item()
    loss /= len(train_dataloader)
    if loss < best_loss:
        best_loss = loss
        best_constant = constant

print(f"Best constant: {best_constant}, Loss: {best_loss}")
