import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_time_steps, n_features, d_model, n_heads, ff_units, prediction_horizon):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=n_time_steps)
        self.transformer_block = TransformerBlock(d_model, n_heads, ff_units)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, prediction_horizon)

    def forward(self, x):
        # Input shape: (batch_size, n_time_steps, n_features)
        x = self.embedding(x)  # Linear transformation to d_model
        x = self.positional_encoding(x)
        # Transformer expects input of shape (n_time_steps, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer_block(x)
        x = x.permute(1, 2, 0)  # Back to (batch_size, d_model, n_time_steps)
        x = self.global_avg_pool(x).squeeze(-1)
        output = self.fc_out(x)
        return output


def train_transformer(model, train_loader, optimizer, loss_fn, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')


def evaluate_transformer_model(model, test_loader, device):
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            true_values.append(labels.numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    # Compute RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    print(f"Test RMSE: {rmse:.4f}")
    return predictions, true_values


### ----------------- Knowledge Distillation -----------------
def distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.5, temperature=2.0):
    # Supervised loss (MSE between student predictions and true labels)
    loss_supervised = nn.MSELoss()(student_outputs, labels)
    # Distillation loss (MSE between student and teacher predictions)
    loss_distillation = nn.MSELoss()(student_outputs, teacher_outputs / temperature)
    # Combine the two
    return alpha * loss_supervised + (1 - alpha) * loss_distillation


def train_student_with_distillation(student_model, teacher_model, train_loader, optimizer, device, epochs=100,
                                    alpha=0.5, temperature=2.0):
    student_model.train()
    teacher_model.eval()  # Freeze teacher model
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass for teacher and student
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)

            # Compute distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs, labels, alpha, temperature)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')


def knowledge_distillation(var, device):
    # Hyperparameters
    X_train_H, X_test_H, Y_train_H, Y_test_H = var['X_train_H'], var['X_test_H'], var['Y_train_H'], var['Y_test_H']
    n_time_steps, n_features = var['n_time_steps'], var['n_features']
    d_model, n_heads, ff_units = var['d_model'], var['n_heads'], var['ff_units'] # Teacher model
    s_d_model, s_n_heads, s_ff_units = var['s_d_model'], var['s_n_heads'], var['s_ff_units'] # Student model
    prediction_horizon = var['prediction_horizon']
    epochs_teacher, epochs_student = var['epochs_teacher'], var['epochs_student']
    alpha, temperature = var['alpha'], var['temperature']

    #convert the data to tensor
    X_train_H = torch.from_numpy(X_train_H).float()
    X_test_H = torch.from_numpy(X_test_H).float()
    y_train_H = torch.from_numpy(Y_train_H).float()
    y_test_H = torch.from_numpy(Y_test_H).float()

    # Datasets
    train_dataset_H = torch.utils.data.TensorDataset(X_train_H, y_train_H)
    test_dataset_H = torch.utils.data.TensorDataset(X_test_H, y_test_H)
    # Dataloaders
    train_loader_H = DataLoader(train_dataset_H, batch_size=64, shuffle=True)
    test_loader_H = DataLoader(test_dataset_H, batch_size=64, shuffle=False)

    # Initialize Teacher and Student Models
    teacher_model = TimeSeriesTransformer(n_time_steps, n_features, d_model, n_heads,
                                                      ff_units, prediction_horizon).to(device)

    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train the teacher model
    train_transformer(teacher_model, train_loader_H, optimizer, loss_fn, device, epochs_teacher)

    # Define the small student model
    student_model = TimeSeriesTransformer(n_time_steps=n_time_steps, n_features=n_features,
                                                      d_model=s_d_model,
                                                      n_heads=s_n_heads, ff_units=s_ff_units,
                                                      prediction_horizon=12).to(device)

    # Train the student model using teacher's knowledge
    train_student_with_distillation(student_model, teacher_model, train_loader_H, optimizer, device,
                                                epochs=epochs_student, alpha=alpha, temperature=temperature)

    # Evaluate the student model
    print("Evaluating student model...")
    predictions, true_values = evaluate_transformer_model(student_model, test_loader_H, device)

    return predictions, true_values