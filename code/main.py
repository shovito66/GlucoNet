import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader

import code.model.cnn_lstm as cnn_lstm
import code.model.transformer as transformer
from code.utilities.data_process import min_max_normalization, prepare_data, prepare_effective_carb_data, \
    prepare_effective_insulin_data, get_normalized_cgm_carb_insulin, plot_figure


def setup():
    """
    the function to initialize any basic configurations, such as output configurations,
    initializing seeds for random number generations, etc.
    """
    warnings.filterwarnings('ignore')
    # Set the seed for random number generation
    np.random.seed(42)


def run_gluco_net(rmse_list, mae_list, r_square, kd_var, cnn_var,
                  pid_year, image_directory, file1, device):

    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            xl = pd.read_csv(f'mix_G_B_C_{year}_{pid}_train.csv')  # training data
            train_eff_data, data_Glucose_Baseline, date_time = prepare_effective_carb_data(xl)
            train_eff_data = prepare_effective_insulin_data(train_eff_data)

            xl_test = pd.read_csv(f"mix_G_B_C_{year}_{pid}_test.csv")
            test_eff_data, data_Glucose_test, _ = prepare_effective_carb_data(xl_test)
            test_eff_data = prepare_effective_insulin_data(test_eff_data)

            train_var = get_normalized_cgm_carb_insulin(train_eff_data)
            test_var = get_normalized_cgm_carb_insulin(test_eff_data, test=True, train_var=train_var)

            norm_lowfreq = train_var['norm_lowfreq']
            norm_ins = train_var['norm_ins']
            norm_carb = train_var['norm_carb']
            scaler_glucose_low = train_var['scaler_glucose_low']
            scaler_insulin = train_var['scaler_insulin']
            scaler_carbs = train_var['scaler_carbs']

            unnorm_glucose_test = test_var['unnorm_glucose']
            norm_ins_test = test_var['norm_ins']
            norm_carb_test = test_var['norm_carb']
            norm_low_freq_test = test_var['norm_lowfreq']

            in_window = 180
            interval = int(in_window / 10 * 2)
            out_60 = 12
            out_30 = 6
            out_1 = 1

            test_x_tot, test_60_tot, test_30_tot, test_1_tot = prepare_data(train_eff_data, unnorm_glucose_test,
                                                                            norm_ins_test, norm_carb_test, interval,
                                                                            out_60, out_30, out_1)
            train_x_low, train_60_low, train_30_low, train_1_low = prepare_data(train_eff_data, norm_lowfreq, norm_ins,
                                                                                norm_carb, interval,
                                                                                out_60, out_30, out_1)

            test_x_low, test_60_low, test_30_low, test_1_low = prepare_data(train_eff_data, norm_low_freq_test,
                                                                            norm_ins_test, norm_carb_test, interval,
                                                                            out_60, out_30, out_1)

            X_train_L = train_x_low
            X_test_L = test_x_low

            Y_train_L = train_60_low
            Y_test_L = test_60_low

            # Convert the data to torch tensors
            X_train_L = torch.from_numpy(X_train_L).float()
            X_test_L = torch.from_numpy(X_test_L).float()
            y_train_L = torch.from_numpy(Y_train_L).float()
            y_test_L = torch.from_numpy(Y_test_L).float()

            # Datasets
            train_dataset_L = torch.utils.data.TensorDataset(X_train_L, y_train_L)
            test_dataset_L = torch.utils.data.TensorDataset(X_test_L, y_test_L)
            # Dataloaders
            train_loader_L = DataLoader(train_dataset_L, batch_size=64, shuffle=True)
            test_loader_L = DataLoader(test_dataset_L, batch_size=64, shuffle=False)

            # ===== Initialize the CNN-LSTM model ====
            hidden_size = cnn_var['hidden_size']
            hidden_size2 = cnn_var['hidden_size2']
            num_layers = cnn_var['num_layers']
            num_classes = cnn_var['num_classes']
            num_epochs = cnn_var['num_epochs']

            input_size_L = X_train_L.shape[-1]
            model_cnn_lstm_L = cnn_lstm.CNN_LSTM(input_size_L, hidden_size, hidden_size2,
                                                 num_layers, num_classes).to(device)

            cnn_lstm.train_cnn_lstm_using_low_freq(model_cnn_lstm_L, train_loader_L,
                                                   num_epochs, test_loader_L)

            dataset_test = (y_test_L.cpu().numpy())
            act_sig = scaler_glucose_low.inverse_transform(dataset_test)

            model_cnn_lstm_L.eval()
            with torch.no_grad():
                test_predictions2 = []
                for batch_X_test in X_test_L:
                    batch_X_test = batch_X_test.to(device).unsqueeze(0)  # Add batch dimension
                    test_predictions2.append(model_cnn_lstm_L(batch_X_test).cpu().numpy().flatten())

            test_predictions2 = np.array(test_predictions2)

            tot_prediction = test_predictions2[:, 0]
            tot_prediction = np.reshape(tot_prediction, (-1, 1))
            tot_prediction = scaler_glucose_low.inverse_transform(tot_prediction)

            # Calculate RMSE and R² score
            rmse = np.sqrt(mean_squared_error(act_sig[:, 0], tot_prediction))
            r2 = r2_score(act_sig[:, 0], tot_prediction)
            MAE = mean_absolute_error(act_sig[:, 0], tot_prediction)
            rmse_list.append(rmse)
            r_square.append(r2)
            mae_list.append(MAE)

            print(f'RMSE Low_{year}_{pid}: {rmse:.4f}')
            print(f'R² ScoreLow_{year}_{pid}: {r2:.4f}')
            print(f'MAELow_{year}_{pid}: {MAE}')

            act_sig = act_sig[:, 0]

            plot_figure(act_sig, tot_prediction, f'Blood Glucose Prediction using LSTM_{year}_{pid}_Low',
                        f'{image_directory}/predict_{year}_{pid}_low.png')

            # Knowledge Distillation Part

            unnorm_highfreq = np.reshape(train_var['high_freq'], (-1, 1))
            unnorm_high_freq_test = np.reshape(test_var['high_freq'], (-1, 1))

            norm_highfreq, scaler_glucose_high = min_max_normalization(unnorm_highfreq)
            norm_highfreq = norm_highfreq.ravel()

            norm_high_freq_test = scaler_glucose_high.fit_transform(unnorm_high_freq_test)
            norm_high_freq_test = norm_high_freq_test.ravel()

            unnorm_highfreq = unnorm_highfreq.ravel()
            unnorm_highfreq = np.squeeze(unnorm_highfreq)

            unnorm_high_freq_test = unnorm_high_freq_test.ravel()
            unnorm_high_freq_test = np.squeeze(unnorm_high_freq_test)

            train_x_high, train_60_high, train_30_high, train_1_high = prepare_data(train_eff_data, unnorm_highfreq,
                                                                                    norm_ins,
                                                                                    norm_carb, interval,
                                                                                    out_60, out_30, out_1)

            test_x_high, test_60_high, test_30_high, test_1_high = prepare_data(train_eff_data, unnorm_high_freq_test,
                                                                                norm_ins_test, norm_carb_test,
                                                                                interval,
                                                                                out_60, out_30, out_1)

            X_train_H = train_x_high
            X_test_H = test_x_high

            Y_train_H = train_60_high
            Y_test_H = test_60_high

            kd_var['X_train_H'] = X_train_H
            kd_var['X_test_H'] = X_test_H
            kd_var['Y_train_H'] = Y_train_H
            kd_var['Y_test_H'] = Y_test_H

            predictions, true_values = transformer.knowledge_distillation(kd_var, device)

            plot_figure(act_sig=true_values[:, 0], pred_sig=predictions[:, 0],
                        title='Blood Glucose Prediction with Student Model',
                        path=f'{image_directory}/predict_{year}_{pid}_High.png')

            test_predictions2 = np.array(predictions)

            tot_prediction_H = test_predictions2[:, 0]
            tot_prediction_H = np.reshape(tot_prediction_H, (-1, 1))

            Forecast_BGL = tot_prediction + tot_prediction_H

            act_sig = test_60_tot
            rmse = np.sqrt(mean_squared_error(act_sig[:, 0], Forecast_BGL))
            r2 = r2_score(act_sig[:, 0], Forecast_BGL)
            MAE = mean_absolute_error(act_sig[:, 0], Forecast_BGL)
            rmse_list.append(rmse)
            r_square.append(r2)
            mae_list.append(MAE)

            print(f'RMSE Total_{year}_{pid}: {rmse:.4f}')
            print(f'R² Score Total_{year}_{pid}: {r2:.4f}')
            print(f'MAE Total_{year}_{pid}: {MAE}')

            file1.writelines(f'RMSE_{year}_{pid}: {rmse:.4f}\n')
            file1.writelines(f'R² Score_{year}_{pid}: {r2:.4f}\n')
            file1.writelines(f'MAE_{year}_{pid}: {MAE}\n')

            plot_figure(act_sig=act_sig[:, 0], pred_sig=Forecast_BGL,
                        title=f'Blood Glucose Prediction using Combinational_{year}_{pid}_Total',
                        path=f'{image_directory}/predict_{year}_{pid}_total.png')

    return rmse_list, mae_list, r_square


if __name__ == '__main__':
    data_directory = '../dataset/'
    output_directory = '../output/'
    image_directory = f'{output_directory}/images60min'

    # create the directory for the images if it does not exist
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    output_file = f'{output_directory}/results_total_LSTM_Transformer60mins.txt'

    # ===== CNN-LSTM model parameters
    hidden_size = 128
    hidden_size2 = 64
    num_layers = 3
    num_classes = 12
    num_epochs = 300

    # ===== Transformer (Knowledge Distillation)  parameters
    # Teacher Model Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_time_steps = 36  # Input time window size
    n_features = 3  # Number of signals (blood glucose, insulin, meal data)
    d_model = 64  # Embedding dimension
    n_heads = 4  # Number of attention heads
    ff_units = 128  # Feedforward units
    prediction_horizon = 12  # Predicting next time step
    epochs_teacher = 500
    alpha = 0.5  # Balance between hard and soft losses
    temperature = 2  # Temperature for distillation

    # --------- Student Model Parameters
    s_d_model = 32
    s_n_heads = 2
    s_ff_units = 64
    epochs_student = 500

    kd_var = {
        'n_time_steps': n_time_steps,
        'n_features': n_features,
        'd_model': d_model,
        'n_heads': n_heads,
        'ff_units': ff_units,
        'prediction_horizon': prediction_horizon,
        's_d_model': s_d_model,
        's_n_heads': s_n_heads,
        's_ff_units': s_ff_units,
        'epochs_teacher': epochs_teacher,
        'alpha': alpha,
        'temperature': temperature,
        'epochs_student': epochs_student
    }

    cnn_var = {
        'hidden_size': hidden_size,
        'hidden_size2': hidden_size2,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'num_epochs': num_epochs
    }

    # -------- Initialize the configurations -------- #
    RMSE_list = []
    MAE_list = []
    R_Square = []

    pid_2018 = [559, 563, 570, 588, 575, 591]
    pid_2020 = [540, 552, 544, 567, 584, 596]
    pid_year = {2018: pid_2018, 2020: pid_2020}

    file1 = open(output_file, 'w')
    RMSE_list, MAE_list, R_Square = run_gluco_net(RMSE_list, MAE_list, R_Square, kd_var, cnn_var,
                                                  pid_year, image_directory, file1, device)

    print('RMSE=', RMSE_list)
    print('MAE=', MAE_list)
    print('R2_Square=', R_Square)
