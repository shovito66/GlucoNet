import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD


def VMD_Glucose(data_Glucose_Baseline):
    # . some sample parameters for VMD
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 10  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7

    # . Run actual VMD code
    u, u_hat, omega = VMD(data_Glucose_Baseline, alpha, tau, K, DC, init, tol)
    # plotting the modes

    # Create a figure and an array of subplots with 2 rows and 5 columns

    v = 0
    for i in range(6):
        v = u[i] + v

    h = u[6] + u[7] + u[8] + u[9]

    return v, h

def prepare_data(whole_data,windowed_data,ins_windowed_data,carb_windowes_data,interval,PH60_sample,PH30_sample,PH1_sample):
    train_data = np.zeros(
        (int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, interval, 3))

    PH_60 = np.zeros((int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, PH60_sample))
    PH_30 = np.zeros((int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, PH30_sample))
    PH_1 = np.zeros((int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, PH1_sample))

    for i in range(0, train_data.shape[0]):
        train_data[i, :, 0] = windowed_data[i:interval + i]
        train_data[i, :, 1] = ins_windowed_data[i:interval + i]
        train_data[i, :, 2] = carb_windowes_data[i:interval + i]
        PH_60[i] = windowed_data[interval + i:interval + (i) + PH60_sample]
        PH_30[i] = PH_60[i][0:6]
        PH_1[i] = PH_60[i][0]

    return train_data,PH_60,PH_30,PH_1

def carbs_to_operative_carbs(carbs, time_str,time_str_previoussample,max_peak_time_str, meal_time_str, max_time=60, increase_rate=0.111, decrease_rate=0.028):
    # Convert string times to datetime objects
    date_format = '%Y-%m-%d %H:%M:%S'
    current_time = datetime.strptime(time_str, date_format)
    meal_time = datetime.strptime(meal_time_str, date_format)
    previous_time=datetime.strptime(time_str_previoussample,date_format)
    max_peak_time = datetime.strptime(max_peak_time_str, date_format)
    # Convert times to minutes for easier calculation
    current_minutes = current_time.hour * 60 + current_time.minute
    meal_minutes = meal_time.hour * 60 + meal_time.minute
    meal_time_2=meal_minutes+10
    previous_time_minutes=previous_time.hour * 60 +previous_time.minute
    max_peak_time_minutes=max_peak_time.hour*60+max_peak_time.minute

    time_diff = current_minutes - meal_minutes

    if time_diff < 0:
        return 0
    elif 0 <= time_diff < 15:
        return 0
    elif 15 <= time_diff <= max_time:
        carb_eff=carbs * increase_rate * ((current_minutes - meal_time_2) / (current_minutes - previous_time_minutes))
        return carb_eff
    elif max_time<time_diff <48*5 :
        if current_minutes-previous_time_minutes==0:
            print(current_minutes)
        carb_eff=max(0, carbs * (1 - (current_minutes - max_peak_time_minutes)/(current_minutes-previous_time_minutes)*decrease_rate))
        return carb_eff
    else:
        return 0



def bolus_to_active_insulin(insulin, time_str, bolus_time_str, duration=360, time_constant=55):
    # Convert string times to datetime objects
    date_format = '%Y-%m-%d %H:%M:%S'
    current_time = datetime.strptime(time_str, date_format)
    bolus_time = datetime.strptime(bolus_time_str, date_format)
    current_minutes = current_time.hour * 60 + current_time.minute
    # Calculate time difference in minutes
    time_diff = (current_time - bolus_time).total_seconds() / 60



    if time_diff < 0:
        return 0

    if time_diff > duration:
        return 0
    else:
        tau = time_constant * ((1 - (time_constant / duration)) / (1 - 2 * (time_constant / duration)))
        a = 2 * tau / duration
        S = 1 / (1 - a + (1 + a) * np.exp(-duration / tau))

        # Calculate Insulin on Board (IOB) based on Equation 2
        IOB = 1 - S * (1 - a) * ((current_minutes ** 2 / (tau * duration * (1 - a)) - current_minutes / tau - 1) * np.exp(
            -current_minutes / tau) + 1)
        return insulin * IOB


xl_test = pd.read_csv(f"mix_G_B_C_2018_559_test.csv")

xl_test['Glucose'] = xl_test['Glucose'].interpolate()
data_Glucose_test = xl_test['Glucose']

date_time = xl_test['Timestamp']

updated_test_data_eff = xl_test.drop(columns=['Unnamed: 0'])
updated_test_data_eff.fillna(0, inplace=True)
test_eff_data = updated_test_data_eff.to_numpy()

indices = np.nonzero(test_eff_data[:, 3])
meal_time = '2010-12-07 12:00:00'
carbs = 0
max_peak_time = meal_time
peak_carb = 0
for i in range(test_eff_data.shape[0]):
    if i in indices[0]:
        peak_carb = 0
        meal_time = test_eff_data[i, 0]
        carbs = test_eff_data[i, 3]
    if i > 1:
        if (test_eff_data[i, 0] != test_eff_data[i - 1, 0]):
            test_eff_data[i, 3] = carbs_to_operative_carbs(carbs, test_eff_data[i, 0],
                                                            test_eff_data[i - 1, 0],
                                                            max_peak_time, meal_time)
            if test_eff_data[i, 3] > peak_carb:
                peak_carb = test_eff_data[i, 3]
                max_peak_time = test_eff_data[i, 0]
        else:
            test_eff_data[i, 3] = test_eff_data[i - 1, 3]
    else:
        test_eff_data[i, 3] = carbs_to_operative_carbs(carbs, test_eff_data[i, 0], test_eff_data[i, 0],
                                                        max_peak_time,
                                                        meal_time)

indices = np.nonzero(test_eff_data[:, 2])
insulin_time = '2010-12-07 12:00:00'
insulin = 0
max_peak_time = insulin_time
peak_insulin = 0
for i in range(test_eff_data.shape[0]):
    if i in indices[0]:
        insulin_time = test_eff_data[i, 0]
        insulin = test_eff_data[i, 2]

    test_eff_data[i, 2] = bolus_to_active_insulin(insulin, test_eff_data[i, 0], insulin_time)


# Create separate MinMaxScaler instances for each signal
scaler_glucose_low = MinMaxScaler()
scaler_insulin = MinMaxScaler()
scaler_carbs = MinMaxScaler()



unnorm_glucose_test = test_eff_data[:, 1]
unnorm_ins_test = test_eff_data[:, 2]
unnorm_carb_test = test_eff_data[:, 3]

low_freq_test, high_freq_test = VMD_Glucose(unnorm_glucose_test)

unnorm_low_freq_test = np.reshape(low_freq_test, (-1, 1))
unnorm_high_freq_test = np.reshape(high_freq_test, (-1, 1))
unnorm_ins_test = np.reshape(unnorm_ins_test, (-1, 1))
unnorm_carb_test = np.reshape(unnorm_carb_test, (-1, 1))

norm_low_freq_test = scaler_glucose_low.fit_transform(unnorm_low_freq_test)
norm_ins_test = scaler_insulin.fit_transform(unnorm_ins_test)
norm_carb_test = scaler_carbs.fit_transform(unnorm_carb_test)
norm_low_freq_test = norm_low_freq_test.ravel()
norm_ins_test = norm_ins_test.ravel()
norm_carb_test = norm_carb_test.ravel()

in_window = 180
interval = int(in_window / 10 * 2)
out_60 = 12
out_30 = 6
out_1 = 1
test_x_tot, test_60_tot, test_30_tot, test_1_tot = prepare_data(test_eff_data, unnorm_glucose_test,norm_ins_test,norm_carb_test, interval,
                                                                out_60, out_30, out_1)

test_x_low, test_60_low, test_30_low, test_1_low = prepare_data(test_eff_data, norm_low_freq_test,norm_ins_test,norm_carb_test, interval,
                                                                out_60, out_30, out_1)
X_test_low = test_x_low


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size2, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size2, int(hidden_size2 / 2))
        self.fc1 = nn.Linear(int(hidden_size2 / 2), num_classes)
        # self.fc2 = nn.Linear(int(hidden_size2 / 2), num_classes)

    def forward(self, x):
        # cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        out = self.fc1(out)
        # out = self.fc2(out)
        return out


X_test_L = test_x_low

Y_test_L = test_60_low

# Convert the data to torch tensors
X_test_L = torch.from_numpy(X_test_L).float()
y_test_L = torch.from_numpy(Y_test_L).float()

# Datasets
test_dataset_L = torch.utils.data.TensorDataset(X_test_L, y_test_L)
# Dataloaders
test_loader_L = DataLoader(test_dataset_L, batch_size=64, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

input_size_L = X_test_L.shape[-1]
hidden_size = 16
hidden_size2 = 8
num_layers = 1
num_classes = 12 #Prediction Horizon= num_classes*5

cnn_lstm_L = CNN_LSTM(input_size_L, hidden_size, hidden_size2, num_layers, num_classes).to(device)

cnn_lstm_L.load_state_dict(torch.load("lstm.pth",weights_only=True))

# test
batch_X_test = X_test_L[-1, :, :]

dataset_test = (y_test_L.cpu().numpy())
act_sig = scaler_glucose_low.inverse_transform(dataset_test)

cnn_lstm_L.eval()
with torch.no_grad():
    test_predictions2 = []
    for batch_X_test in X_test_L:
        batch_X_test = batch_X_test.to(device).unsqueeze(0)  # Add batch dimension
        test_predictions2.append(cnn_lstm_L(batch_X_test).cpu().numpy().flatten())

test_predictions2 = np.array(test_predictions2)

tot_prediction = test_predictions2[:, 0]
tot_prediction = np.reshape(tot_prediction, (-1, 1))
tot_prediction = scaler_glucose_low.inverse_transform(tot_prediction)

# Calculate RMSE and R² score
rmse = np.sqrt(mean_squared_error(act_sig[:, 0], tot_prediction))
r2 = r2_score(act_sig[:, 0], tot_prediction)
MAE = mean_absolute_error(act_sig[:, 0], tot_prediction)
print(f'RMSE Low_2018_559: {rmse:.4f}')
print(f'R² ScoreLow_2018_559: {r2:.4f}')
print(f'MAELow_2018_559: {MAE}')


test_predictions2 = test_predictions2[:, 0]
act_sig = act_sig[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(act_sig, label='Actual')
plt.plot(tot_prediction, label='Predicted')
# plt.xticks(test_times[::5], rotation=45)
plt.xlabel('Time')
plt.ylabel('Blood Glucose')
plt.title(f'Blood Glucose Prediction using LSTM_2018_559_Low')
plt.legend()
plt.savefig(f'images60min/predict_2018_559_low.png', bbox_inches='tight')


# Positional Encoding Class
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


# # Transformer Block Class
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

class SmallStudentTransformer(nn.Module):
    def __init__(self, n_time_steps, n_features, d_model=32, n_heads=2, ff_units=64, prediction_horizon=1,
                 n_layers=1):
        super(SmallStudentTransformer, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=n_time_steps)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_units) for _ in range(n_layers)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, prediction_horizon)

    def forward(self, x):
        x = self.embedding(x)  # Linear transformation to d_model
        x = self.positional_encoding(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # Transform shape for transformer blocks: (n_time_steps, batch_size, d_model)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.permute(1, 2, 0)  # Shape back to (batch_size, d_model, n_time_steps)
        x = self.global_avg_pool(x).squeeze(-1)
        output = self.fc_out(x)
        return output



# Evaluation function
def evaluate_model(model, test_loader, device):
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

    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    return predictions, true_values


# Hyperparameters and initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_time_steps = 36  # Input time window size
n_features = 3  # Number of signals (blood glucose, insulin, meal data)
d_model = 64  # Embedding dimension
n_heads = 4  # Number of attention heads
ff_units = 128  # Feedforward units
prediction_horizon = 12 # Predicting next time step
epochs = 500
alpha = 0.5  # Balance between hard and soft losses
T = 2  # Temperature for distillation

# Prepare Data Loaders

unnorm_high_freq_test = unnorm_high_freq_test.ravel()
unnorm_high_freq_test = np.squeeze(unnorm_high_freq_test)


test_x_high, test_60_high, test_30_high, test_1_high = prepare_data(test_eff_data, unnorm_high_freq_test,
                                                                    norm_ins_test, norm_carb_test,
                                                                    interval,
                                                                    out_60, out_30, out_1)

X_test_high = test_x_high

X_test_H = test_x_high

Y_test_H = test_60_high

# Convert the data to torch tensors
X_test_H = torch.from_numpy(X_test_H).float()

y_test_H = torch.from_numpy(Y_test_H).float()

# Datasets
test_dataset_H = torch.utils.data.TensorDataset(X_test_H, y_test_H)
# Dataloaders
test_loader_H = DataLoader(test_dataset_H, batch_size=64, shuffle=False)
test_loader_H = DataLoader(test_dataset_H, batch_size=64, shuffle=False)


# Define the small student model
student_model = SmallStudentTransformer(
    n_time_steps=36, n_features=3, d_model=32, n_heads=2, ff_units=64, prediction_horizon=12
).to(device)


# Train the student model
student_model.load_state_dict(torch.load('student.pth',weights_only=True))
# Evaluate Student Model
print("Evaluating student model...")
predictions, true_values = evaluate_model(student_model, test_loader_H, device)




# Plot Predictions vs Actuals for Evaluation
plt.figure(figsize=(10, 6))
plt.plot(true_values[:, 0], label='Actual')
plt.plot(predictions[:, 0], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Blood Glucose')
plt.title('Blood Glucose Prediction with Student Model')
plt.legend()
plt.savefig(f'images60min/predict_2018_559_High.png', bbox_inches='tight')

test_predictions2 = np.array(predictions)

tot_prediction_H = test_predictions2[:, 0]
tot_prediction_H = np.reshape(tot_prediction_H, (-1, 1))

Forecast_BGL = tot_prediction + tot_prediction_H

act_sig = test_60_tot
rmse = np.sqrt(mean_squared_error(act_sig[:, 0], Forecast_BGL))
r2 = r2_score(act_sig[:, 0], Forecast_BGL)
MAE = mean_absolute_error(act_sig[:, 0], Forecast_BGL)


print(f'RMSE Total_2018_559: {rmse:.4f}')
print(f'R² Score Total_2018_559: {r2:.4f}')
print(f'MAE Total_2018_559: {MAE}')

plt.figure(figsize=(10, 6))
plt.plot(act_sig[:, 0], label='Actual')
plt.plot(Forecast_BGL, label='Predicted')
# plt.xticks(test_times[::5], rotation=45)
plt.xlabel('Time')
plt.ylabel('Blood Glucose')
plt.title(f'Blood Glucose Prediction using Combinational_2018_559_Total')
plt.legend()
plt.savefig(f'images60min/predict_2018_559_Tot.png', bbox_inches='tight')
plt.show()

