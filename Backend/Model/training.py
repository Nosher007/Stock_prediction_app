# train_tcn_with_inversion.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN  # TCN layer from keras-tcn


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, utc=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create target for next day's close
    df["Target_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df


def create_sequences(data: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size, -1])  # target is last column
    return np.array(X), np.array(y)


def build_tcn_model(input_shape):
    model = Sequential()
    model.add(TCN(input_shape=input_shape, nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def invert_predictions(y_pred_scaled, y_test_scaled, scaler, num_features):
    """
    Reconstructs arrays so that we can inverse_transform only the target column
    (the last column in your data).

    y_pred_scaled: shape (n, 1) from model predictions
    y_test_scaled: shape (n,) or (n, 1) from the test set's target
    scaler: the fitted MinMaxScaler used on the entire dataset
    num_features: total number of features (including the target) used in scaling
    """
    # Ensure y_pred_scaled and y_test_scaled are 1D or 2D consistently
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_test_scaled = y_test_scaled.reshape(-1, 1)

    # Create zero arrays for all features
    arr_pred = np.zeros((len(y_pred_scaled), num_features))
    arr_test = np.zeros((len(y_test_scaled), num_features))

    # Place the scaled predictions in the last column
    arr_pred[:, -1] = y_pred_scaled.ravel()
    arr_test[:, -1] = y_test_scaled.ravel()

    # Inverse transform
    inv_pred = scaler.inverse_transform(arr_pred)[:, -1]
    inv_test = scaler.inverse_transform(arr_test)[:, -1]

    return inv_pred, inv_test


def main():
    csv_path = "merged_data.csv"  # Adjust path if needed
    window_size = 10

    # 1. Load and preprocess
    df = load_and_preprocess_data(csv_path)
    print("Preprocessed Data Sample:")
    print(df.head())

    # 2. Define features (including target as last column)
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Target_Close"]
    data = df[feature_cols].values

    # 3. Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 4. Create sequences
    X, y = create_sequences(scaled_data, window_size)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 5. Train-test split (time-based)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 6. Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_tcn_model(input_shape)
    model.summary()

    # 7. Train
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # 8. Evaluate on scaled test data
    y_pred_scaled = model.predict(X_test)
    mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
    mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    print(f"TCN Model MAE (scaled): {mae_scaled:.4f}")
    print(f"TCN Model MSE (scaled): {mse_scaled:.4f}")

    # 9. Invert scaling to get predictions in real units
    #    The total number of features (including Target_Close) is len(feature_cols).
    num_features = len(feature_cols)
    inv_pred, inv_test = invert_predictions(y_pred_scaled, y_test, scaler, num_features)

    # 10. Calculate real-scale MAE/MSE
    mae_real = mean_absolute_error(inv_test, inv_pred)
    mse_real = mean_squared_error(inv_test, inv_pred)
    print(f"TCN Model MAE (real units): {mae_real:.2f}")
    print(f"TCN Model MSE (real units): {mse_real:.2f}")

    # Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.title("TCN Training and Validation Loss")
    plt.show()


if __name__ == "__main__":
    main()
