import re
import sys
import matplotlib.pyplot as plt


def parse_log(log_path):
    epochs = []
    acc = []
    loss = []
    val_acc = []
    val_loss = []

    # Regex to match the Keras output lines:
    # 13/13 - 1049s - 81s/step - accuracy: 0.7889 - loss: 0.7008 - val_accuracy: 0.8277 - val_loss: 0.6686 - learning_rate: 0.0010
    pattern = re.compile(
        r"accuracy:\s+([\d.]+)\s+-\s+loss:\s+([\d.]+)\s+-\s+val_accuracy:\s+([\d.]+)\s+-\s+val_loss:\s+([\d.]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                acc.append(float(match.group(1)))
                loss.append(float(match.group(2)))
                val_acc.append(float(match.group(3)))
                val_loss.append(float(match.group(4)))
                epochs.append(len(acc))

    return epochs, acc, loss, val_acc, val_loss


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_log.py <path_to_log_file>")
        sys.exit(1)

    log_path = sys.argv[1]

    epochs, acc, loss, val_acc, val_loss = parse_log(log_path)

    if not epochs:
        print(
            f"No training data found in {log_path}. The job might have failed before completing an epoch."
        )
        sys.exit(1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out_file = log_path.replace(".log", ".png")
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
