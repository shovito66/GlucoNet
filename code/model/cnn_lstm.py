import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam


class CNN_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_size2, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # -------- Stacked LSTM layer -------- #
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size2, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size2, hidden_size2)
        self.fc1 = nn.Linear(hidden_size2, int(hidden_size2 / 2))
        self.fc2 = nn.Linear(int(hidden_size2 / 2), num_classes)

    def forward(self, x):
        # cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def evaluate_cnn_lstm_model(model, test_loader, criterion,
                            epoch, epochs, test_hist, device):
    with torch.no_grad():
        total_test_loss = 0.0
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_pred_test = model(x_test)
            test_loss = criterion(y_pred_test, y_test)
            total_test_loss += test_loss.item()

        average_test_loss = total_test_loss / len(test_loader)
        test_hist.append(average_test_loss)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Test Loss: {average_test_loss:.4f}')


def train_cnn_lstm_using_low_freq(model, train_loader: DataLoader, epochs: int, test_loader_L):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    test_hist = []

    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        evaluate_cnn_lstm_model(model, test_loader_L, criterion,
                                epoch, epochs, test_hist, device)
    print("Training completed for model: ", model.__class__.__name__)

