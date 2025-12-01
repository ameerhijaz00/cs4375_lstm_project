import numpy as np
from preprocess import load_dataset, scale_data, create_sequences
from lstm_from_scratch import LSTM
import matplotlib.pyplot as plt

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def main():
    data = load_dataset()
    scaled, scaler = scale_data(data)

    window = 10
    X, y = create_sequences(scaled, window)

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    model = LSTM(input_size=1, hidden_size=20, learning_rate=0.001, clip_value=1.0)

    epochs = 10
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = 0.0
        for i in range(len(X_train)):
            seq = X_train[i]
            target = y_train[i].reshape(1, 1)
            pred = model.forward(seq)
            loss = mse(pred, target)
            train_loss += loss
            model.backward(target)
        train_loss /= len(X_train)

        test_loss = 0.0
        for i in range(len(X_test)):
            seq = X_test[i]
            target = y_test[i].reshape(1, 1)
            pred = model.predict(seq)
            loss = mse(pred, target)
            test_loss += loss
        test_loss /= len(X_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    # Plot training curve
    plt.figure(figsize=(10,4))
    plt.plot(train_losses, label="Train Loss", linewidth=1.5)
    plt.plot(test_losses, label="Test Loss", linewidth=1.5)
    plt.legend()
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("results/plots/training_curve.png")

    # Generate predictions
    preds = []
    for seq in X_test:
        out = model.predict(seq)
        preds.append(out.item())

    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    actual_inv = scaler.inverse_transform(y_test)

    # Print sample predictions
    print("\n=== Sample Predictions (Actual vs Predicted) ===")
    for i in range(10):
        actual = actual_inv[i][0]
        predicted = preds_inv[i][0]
        print(f"Day {i+1}: Actual = {actual:.2f}°C, Predicted = {predicted:.2f}°C")

    # Plot forecast
    plt.figure(figsize=(10,4))
    plt.plot(actual_inv, label="Actual", linewidth=1.5)
    plt.plot(preds_inv, label="Predicted", linewidth=1.5)
    plt.legend()
    plt.title("LSTM Forecasting")
    plt.xlabel("Day Index (Test Set)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("results/plots/forecast.png")

    # Write metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Final Train Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Test Loss: {test_losses[-1]:.6f}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write("Train Loss Curve:\n")
        f.write(str([float(x) for x in train_losses]) + "\n")
        f.write("Test Loss Curve:\n")
        f.write(str([float(x) for x in test_losses]) + "\n")

    # Log experiment
    with open("logs/experiments_log.txt", "w") as log:
        log.write("Parameters Chosen:\n")
        log.write(" - Model: LSTM (from scratch)\n")
        log.write(f" - Hidden Size: {20}\n")
        log.write(f" - Input Window: {window}\n")
        log.write(f" - Learning Rate: {model.learning_rate}\n")
        log.write(f" - Epochs: {epochs}\n")
        log.write(f" - Gradient Clipping: {model.clip_value}\n\n")

        log.write("Results:\n")
        log.write(" - Train/Test Split: 80/20\n")
        log.write(f" - Dataset Size: {len(data)}\n")
        log.write(f" - Final Train Loss: {train_losses[-1]:.6f}\n")
        log.write(f" - Final Test Loss: {test_losses[-1]:.6f}\n")
        log.write(f" - Train Loss Curve: {[float(x) for x in train_losses]}\n")
        log.write(f" - Test Loss Curve: {[float(x) for x in test_losses]}\n\n")
        log.write("============================================================\n")

if __name__ == "__main__":
    main()
