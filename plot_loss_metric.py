import re
import matplotlib.pyplot as plt

# Define extraction patterns
pattern_train = re.compile(r'Epoch \[(\d+)/\d+\] \| Step \[(\d+)/\d+\] \| Train Loss: ([-\d.]+)')
pattern_valid = re.compile(r'Epoch \[(\d+)/\d+\] \| Step \[(\d+)/\d+\] \| Valid Loss: ([-\d.]+)')
pattern_metric = re.compile(r'Epoch \[(\d+)/\d+\] \| Step \[(\d+)/\d+\] \| Valid Loss: [-\d.]+, Metric: ([\d.]+)')

# Function to extract data from a file
def extract_data_from_file(filepath):
    epochs = []
    train_losses = []
    valid_epochs_loss = []
    valid_losses = []
    valid_epochs = []
    valid_metrics = []

    with open(filepath, "r") as file:
        lines = file.readlines()

        for line in lines:
            match_train = pattern_train.search(line)
            match_valid = pattern_valid.search(line)
            match_metric = pattern_metric.search(line)

            if match_train:
                epoch, _, loss = match_train.groups()
                epochs.append(int(epoch))
                train_losses.append(float(loss))

            if match_valid:
                epoch, _, loss = match_valid.groups()
                valid_epochs_loss.append(int(epoch))
                valid_losses.append(float(loss))

            if match_metric:
                epoch, _, metric = match_metric.groups()
                valid_epochs.append(int(epoch))
                valid_metrics.append(float(metric))

    return epochs, train_losses, valid_epochs_loss, valid_losses, valid_epochs, valid_metrics

# Extract data from the files
epochs1, train_losses1, valid_epochs_loss1, valid_losses1, valid_epochs1, valid_metrics1 = extract_data_from_file("/Users/karanthakkar/Downloads/training_log_2023-11-05_10:03:55.txt")
epochs2, train_losses2, valid_epochs_loss2, valid_losses2, valid_epochs2, valid_metrics2 = extract_data_from_file("/Users/karanthakkar/Downloads/training_log_2023-11-04_21:34:50.txt")

# Plot Train Loss with log scale
plt.figure(figsize=(12, 6))
plt.plot(epochs1, train_losses1, '-o', label="Train Loss", markersize=3)
plt.plot(epochs2, train_losses2, '-o', linestyle='--', label="Train Loss (random sampling across trials)", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Negative Train Loss (Log Scale)")
#plt.yscale("log")
plt.title("Negative Train Loss over Epochs")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("train_loss_overlayed_2.png")

# Continue with other plots...

# Plot Valid Loss
plt.figure(figsize=(12, 6))
plt.plot(valid_epochs_loss1, valid_losses1, '-o', color="red", label="Valid Loss", markersize=3)
plt.plot(valid_epochs_loss2, valid_losses2, '-o', linestyle='--', color="purple", label="Valid Loss (random sampling across trials)", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Negative Valid Loss")
plt.title("Negative Valid Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("valid_loss_overlayed_2.png")

# Plot Validation Metric
plt.figure(figsize=(12, 6))
plt.plot(valid_epochs1, valid_metrics1, '-o', color="green", label="Validation Metric", markersize=3)
plt.plot(valid_epochs2, valid_metrics2, '-o', color="orange", label="Validation Metric (random sampling across trials)", markersize=3, linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Metric over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("validation_metric_overlayed_2.png")


