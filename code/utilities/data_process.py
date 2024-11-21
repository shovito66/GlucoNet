import numpy as np
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD
from datetime import datetime
from matplotlib import pyplot as plt


def VMD_Glucose(data_Glucose_Baseline):
    """
    vmd decomposition of the glucose data into 10 modes and the residue
    :param data_Glucose_Baseline: glucose data
    :return:    v: low frequency components,
                h: high frequency components
    """

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
    for i in range(8):
        v = u[i] + v

    h = u[8] + u[9]

    return v, h  # low, high frequency components


def prepare_data(whole_data, windowed_data, ins_windowed_data, carb_windowes_data, interval, PH60_sample, PH30_sample,
                 PH1_sample):
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

    return train_data, PH_60, PH_30, PH_1


def min_max_normalization(data):
    """
    Min-max normalization
    :param data: the data to be normalized
    :return: the normalized data, the scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler


def carbs_to_operative_carbs(carbs, time_str, time_str_previoussample, max_peak_time_str,
                             meal_time_str, max_time=60, increase_rate=0.111, decrease_rate=0.028):
    """
    Calculate the operative carbs based on the time of the day and the time of the meal and
    the peak time of the glucose data and the previous sample time; this is basically converting sparse carbs
    to continuous carbs

    :param carbs:
    :param time_str:
    :param time_str_previoussample:
    :param max_peak_time_str:
    :param meal_time_str:
    :param max_time:
    :param increase_rate:
    :param decrease_rate:
    :return:  the operative carbs
    """

    # Convert string times to datetime objects
    date_format = '%Y-%m-%d %H:%M:%S'
    current_time = datetime.strptime(time_str, date_format)
    meal_time = datetime.strptime(meal_time_str, date_format)
    previous_time = datetime.strptime(time_str_previoussample, date_format)
    max_peak_time = datetime.strptime(max_peak_time_str, date_format)
    # Convert times to minutes for easier calculation
    current_minutes = current_time.hour * 60 + current_time.minute
    meal_minutes = meal_time.hour * 60 + meal_time.minute
    meal_time_2 = meal_minutes + 10
    previous_time_minutes = previous_time.hour * 60 + previous_time.minute
    max_peak_time_minutes = max_peak_time.hour * 60 + max_peak_time.minute

    time_diff = current_minutes - meal_minutes

    if time_diff < 0:
        return 0
    elif 0 <= time_diff < 15:
        return 0
    elif 15 <= time_diff <= max_time:
        carb_eff = carbs * increase_rate * ((current_minutes - meal_time_2) / (current_minutes - previous_time_minutes))
        return carb_eff
    elif max_time < time_diff < 48 * 5:
        if current_minutes - previous_time_minutes == 0:
            print(current_minutes)
        carb_eff = max(0, carbs * (1 - (current_minutes - max_peak_time_minutes) / (
                current_minutes - previous_time_minutes) * decrease_rate))
        return carb_eff
    else:
        return 0


def bolus_to_active_insulin(insulin, time_str, bolus_time_str, duration=360, time_constant=55):
    """
    Calculate the active insulin based on the time of the day and the time of the bolus and the duration of the insulin
    :param insulin:
    :param time_str:
    :param bolus_time_str:
    :param duration:
    :param time_constant:
    :return: the active insulin
    """

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
        IOB = 1 - S * (1 - a) * (
                (current_minutes ** 2 / (tau * duration * (1 - a)) - current_minutes / tau - 1) * np.exp(
            -current_minutes / tau) + 1)
        return insulin * IOB


def prepare_effective_carb_data(xl):
    """
    the function to prepare the effective carb data for training and testing
    """
    xl['Glucose'] = xl['Glucose'].interpolate()
    data_Glucose = xl['Glucose']
    date_time = xl['Timestamp']
    updated_data_eff = xl.drop(columns=['Unnamed: 0'])
    updated_data_eff.fillna(0, inplace=True)
    eff_data = updated_data_eff.to_numpy()

    indices = np.nonzero(eff_data[:, 3])
    meal_time = '2010-12-07 12:00:00'
    carbs = 0
    max_peak_time = meal_time
    peak_carb = 0
    for i in range(eff_data.shape[0]):
        if i in indices[0]:
            peak_carb = 0
            meal_time = eff_data[i, 0]
            carbs = eff_data[i, 3]
        if i > 1:
            if (eff_data[i, 0] != eff_data[i - 1, 0]):
                eff_data[i, 3] = carbs_to_operative_carbs(carbs, eff_data[i, 0],
                                                          eff_data[i - 1, 0],
                                                          max_peak_time, meal_time)
                if eff_data[i, 3] > peak_carb:
                    peak_carb = eff_data[i, 3]
                    max_peak_time = eff_data[i, 0]
            else:
                eff_data[i, 3] = eff_data[i - 1, 3]
        else:
            eff_data[i, 3] = carbs_to_operative_carbs(carbs, eff_data[i, 0], eff_data[i, 0],
                                                      max_peak_time, meal_time)
    return eff_data, data_Glucose, date_time


def prepare_effective_insulin_data(eff_data):
    """
    the function to prepare the effective insulin data for training and testing
    """
    indices = np.nonzero(eff_data[:, 2])
    insulin_time = '2010-12-07 12:00:00'
    insulin = 0
    max_peak_time = insulin_time
    peak_insulin = 0
    for i in range(eff_data.shape[0]):
        if i in indices[0]:
            insulin_time = eff_data[i, 0]
            insulin = eff_data[i, 2]
        eff_data[i, 2] = bolus_to_active_insulin(insulin, eff_data[i, 0], insulin_time)

    return eff_data


def get_normalized_cgm_carb_insulin(data, test=False, train_var=None):
    unnorm_glucose = data[:, 1]
    unnorm_ins = data[:, 2]
    unnorm_carb = data[:, 3]

    low_freq, high_freq = VMD_Glucose(unnorm_glucose)
    # Reshape to 2D arrays as required by MinMaxScaler
    unnorm_lowfreq = np.reshape(low_freq, (-1, 1))
    unnorm_highfreq = np.reshape(high_freq, (-1, 1))
    unnorm_ins = np.reshape(unnorm_ins, (-1, 1))
    unnorm_carb = np.reshape(unnorm_carb, (-1, 1))

    # Create separate MinMaxScaler instances for each signal
    # and normalize the signals
    if not test:
        norm_lowfreq, scaler_glucose_low = min_max_normalization(unnorm_lowfreq)
        norm_ins, scaler_insulin = min_max_normalization(unnorm_ins)
        norm_carb, scaler_carbs = min_max_normalization(unnorm_carb)

    else:
        scaler_glucose_low = train_var['scaler_glucose_low']
        scaler_insulin = train_var['scaler_insulin']
        scaler_carbs = train_var['scaler_carbs']
        norm_lowfreq = scaler_glucose_low.transform(unnorm_lowfreq).ravel()
        norm_ins = scaler_insulin.transform(unnorm_ins).ravel()
        norm_carb = scaler_carbs.transform(unnorm_carb).ravel()

    norm_lowfreq = norm_lowfreq.ravel()
    norm_ins = norm_ins.ravel()
    norm_carb = norm_carb.ravel()

    var = {
        'norm_lowfreq': norm_lowfreq,
        'norm_ins': norm_ins,
        'norm_carb': norm_carb,
        'scaler_glucose_low': scaler_glucose_low,
        'scaler_insulin': scaler_insulin,
        'scaler_carbs': scaler_carbs,
        'low_freq': low_freq,
        'high_freq': high_freq,
        'unnorm_glucose': unnorm_glucose,
        'unnorm_ins': unnorm_ins,
        'unnorm_carb': unnorm_carb,
        'unnorm_lowfreq': unnorm_lowfreq,
        'unnorm_highfreq': unnorm_highfreq
    }

    return var


def plot_figure(act_sig, pred_sig, title, path):
    plt.figure(figsize=(10, 6))
    plt.plot(act_sig, label='Actual')
    plt.plot(pred_sig, label='Predicted')
    # plt.xticks(test_times[::5], rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Blood Glucose')
    plt.title(title)
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
