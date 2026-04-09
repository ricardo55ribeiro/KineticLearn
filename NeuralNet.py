import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import copy
import time
import random
import itertools
import json
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import sys
from datetime import datetime
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.NeuralNetworkModels import NeuralNet
from src.config import dict as dictionary

class LoadMultiPressureDatasetTorch(torch.utils.data.Dataset):

    def __init__(self, src_file, nspecies, num_pressure_conditions, react_idx=None, m_rows=None, columns=None,
                 scaler_input=None, scaler_output=None):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(src_file, max_rows=m_rows,
                              usecols=columns, delimiter=None,
                              comments="#", skiprows=0, dtype=np.float64)

        ncolumns = len(all_data[0])
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1) # densities
        y_columns = react_idx # k's
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        x_data = all_data[:, x_columns]  # densities
        y_data = all_data[:, y_columns] * 1e30  # k's  # *10 to avoid being at float32 precision limit 1e-17

        # Reshape data for multiple pressure conditions
        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        # Create scalers
        self.scaler_input = scaler_input or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        self.scaler_output = scaler_output or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        
        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        # Transpose x_data to move the pressure condition axis to the end, then flatten
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        
        # Flatten the output data to be of shape (2000,3)
        y_data = y_data[0]

        # Convert the data to PyTorch tensors
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
    
    def get_data(self):
        return self.x_data, self.y_data
    

def train_model(model, criterion, optimizer, dataloader, num_epochs=100, patience=5, val_split=0.1):
    # Split the data into training and validation sets
    train_len = int((1.0 - val_split) * len(dataloader.dataset))
    val_len = len(dataloader.dataset) - train_len

    # Train/validation split with seed so every architecture is compared on exactly the same data partition
    split_generator = torch.Generator().manual_seed(43)
    train_dataset, val_dataset = random_split(dataloader.dataset, [train_len, val_len], generator=split_generator)

    # Training shuffle order with seed so every architecture sees the mini-batches in the same sequence
    shuffle_generator = torch.Generator().manual_seed(43)
    train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Early stopping details
    n_epochs_stop = patience
    min_val_loss = np.inf
    epochs_no_improve = 0

    # To track loss history
    history = {
        'train_loss': [],
        'val_loss': [],
    }

    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        # record training/validation loss
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}, Training loss: {train_loss}, Validation loss: {val_loss}")

        # Early stopping check
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                model.load_state_dict(best_model_wts)
                return model, history

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, test_data):
    model.eval()  # set the model to evaluation mode

    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)

    # Calculate the Mean Squared Error (MSE) on the test data
    mse = mean_squared_error(targets.numpy(), outputs.numpy())
    print(f"Mean Squared Error (MSE) on the test data: {mse}")

    return targets, outputs, mse

def plot_results(targets, outputs, output_size, output_dir):
    # Plot the predictions vs the true values
    fig, axs = plt.subplots(1, output_size, figsize=(15, 5), sharey=True)  # Share the same y-axis
    plt.rcParams.update({'font.size': 16, 'text.usetex': False} )

    for i in range(output_size):
        axs[i].scatter(targets[:, i], outputs[:, i], alpha=0.8, color=(0., 0., 0.9)) # blue
        # draw the y=x line
        axs[i].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color='black')
        axs[i].set_xlabel('True Values', fontsize=14)
        # Only set the y-label for the first subplot since they share the same y-axis
        if i == 0:
            axs[i].set_ylabel('Predicted Values', fontsize=14)
        axs[i].set_title(f"$k_{{{i+1}}}$")

        # Calculate relative error
        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9  # Set small values to a small constant

        rel_err = np.abs(np.subtract(outputs[:,i], targets[:, i]) / denominator)

        textstr = '\n'.join((
            r'$Mean\ \delta_{rel}=%.2f\%%$' % (rel_err.mean() * 100,),
            r'$Max\ \delta_{rel}=%.2f\%%$' % (max(rel_err) * 100,)))

        # Colour point with max error
        max_index = np.argmax(rel_err)
        axs[i].scatter(targets[max_index, i], outputs[max_index, i], color="gold", zorder=2)

        # Define the text box properties
        props = dict(boxstyle='round', alpha=0.5)

        # Place a text box in upper left in axes coords
        axs[i].text(0.63, 0.25, textstr, fontsize=12, transform=axs[i].transAxes,
                verticalalignment='top', bbox=props)

         # Remove tick bars from non-first plots
        if i > 0:
            axs[i].tick_params(left=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'NeuralNet.pdf'))
    plt.close()



def moving_average(x, window=25):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def plot_loss_curves(history, output_dir, log_scale=False):
    plt.rcParams.update({'font.size': 16, 'text.usetex': False})

    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    epochs = np.arange(1, len(train_loss) + 1)

    # Smoothed curves for readability
    train_smooth = moving_average(train_loss, window=25)
    val_smooth = moving_average(val_loss, window=25)

    plt.figure(figsize=(9, 6))

    # Raw curves: faint and thin
    plt.plot(epochs, train_loss, linewidth=0.8, alpha=0.20, label='Training Loss (raw)')
    plt.plot(epochs, val_loss, linewidth=0.8, alpha=0.20, label='Validation Loss (raw)')

    # Smoothed curves: the ones you actually read
    plt.plot(epochs, train_smooth, linewidth=1.8, label='Training Loss')
    plt.plot(epochs, val_smooth, linewidth=1.8, label='Validation Loss')

    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('MSE Loss', fontsize=16)

    if log_scale:
        plt.yscale('log')

    plt.grid(True, which='both', alpha=0.25)
    plt.legend(fontsize=13, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'NeuralNet_loss_curves.pdf'))
    plt.close()




def hyperparameter_random_search(n_samples):
    # Define the search space
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64, 128]
    hidden_sizes = [(50,), (100,), (200,), (500,), \
                    (20, 20), (30, 30), (45, 45), (75,75)]

    activation_functions = ['tanh']

    best_mse = np.inf
    best_model = None
    best_hyperparameters = None

    for _ in range(n_samples):  # Number of samples
        # Sample hyperparameters
        lr = np.random.choice(learning_rates)
        batch_size = random.choice(batch_sizes)
        hidden_size = random.choice(hidden_sizes)
        activation_function = random.choice(activation_functions)

        # Create the model
        model = NeuralNet(input_size, output_size, hidden_size, activation_function)

        # Create the data loader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # Define the loss function and the optimizer
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Train the model
        start = time.time()
        model, loss_history = train_model(model, criterion, optimizer, train_loader, num_epochs=1000, patience=15, val_split=0.1)
        end = time.time()
        print(f"Training time: {end - start}s")

        # Evaluate the model
        test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
        targets, outputs, mse = evaluate_model(model, test_data)

        # Update best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_hyperparameters = (lr, batch_size, hidden_size, activation_function)

        print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, activation_function={activation_function}, MSE: {mse}")

    print(f"Best hyperparameters: lr={best_hyperparameters[0]}, batch_size={best_hyperparameters[1]}, hidden_size={best_hyperparameters[2]}, activation_function={best_hyperparameters[3]}, MSE: {best_mse}")
    return best_model, best_hyperparameters



def hyperparameter_grid_search():
    # Define the search space
    # learning_rates = [0.01, 0.001, 0.0001] 
    learning_rates = [0.00001, 0.000001]
    batch_sizes = [16, 32, 64, 128]
    hidden_sizes = [(50,), (100,), (200,), (500,), \
                    (20, 20), (30, 30), (45, 45), (75,75)]

    activation_functions = ['tanh']

    best_mse = np.inf
    best_model = None
    best_hyperparameters = None

    # Create a list to hold the results
    results = []

    total_iterations = len(learning_rates) * len(batch_sizes) * len(hidden_sizes) * len(activation_functions)

    # Grid search
    for lr, batch_size, hidden_size, activation_function in tqdm(
        itertools.product(learning_rates, batch_sizes, hidden_sizes, activation_functions),
        total=total_iterations,
        desc="Processing combinations"
    ):
        # Create the model
        model = NeuralNet(input_size, output_size, hidden_size, activation_function)

        # Create the data loader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # Define the loss function and the optimizer
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Save the current stdout
        original_stdout = sys.stdout

        # Turn off stdout by redirecting it to os.devnull
        sys.stdout = open(os.devnull, 'w')

        # Train the model (output suppressed)
        model, loss_history = train_model(model, criterion, optimizer, train_loader, num_epochs=2000, patience=15, val_split=0.1)

        # Restore the original stdout
        sys.stdout.close()
        sys.stdout = original_stdout

        # Evaluate the model
        test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
        targets, outputs, mse = evaluate_model(model, test_data)

        # Update best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_hyperparameters = (lr, batch_size, hidden_size, activation_function)

        print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, activation_function={activation_function}, MSE: {mse}")

        # Save the results
        results.append({
            'lr': lr,
            'batch_size': batch_size,
            'hidden_size': str(hidden_size),
            'activation_function': activation_function,
            'mse': mse,
        }) # remeber to save a flag for early stopping

    print(f"Best hyperparameters: lr={best_hyperparameters[0]}, batch_size={best_hyperparameters[1]}, hidden_size={best_hyperparameters[2]}, activation_function={best_hyperparameters[3]}, MSE: {best_mse}")
    return best_model, best_hyperparameters, results


# Custom JSON encoder to handle float32 values
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)



def arch_to_folder_name(hidden_size):
    return ", ".join(map(str, hidden_size))


def make_results_root(base_root, scheme, experiment_name, add_timestamp=True):
    root = os.path.join(base_root, scheme, experiment_name)

    if add_timestamp:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = os.path.join(root, run_id)

    os.makedirs(root, exist_ok=True)
    return root


def prepare_results_folders(architectures, root):
    arch_dirs = {}
    for arch in architectures:
        folder_name = arch_to_folder_name(arch)
        arch_dir = os.path.join(root, folder_name)
        os.makedirs(arch_dir, exist_ok=True)
        arch_dirs[arch] = arch_dir

    return arch_dirs


def compute_metrics_dict(
    scheme,
    hidden_size,
    input_size,
    output_size,
    activation,
    learning_rate,
    batch_size,
    loss_history,
    training_time,
    mse,
    targets,
    outputs
):
    metrics = {
        "scheme": scheme,
        "hidden_size": list(hidden_size),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "activation": activation,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs_ran": len(loss_history["train_loss"]),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
        "test_mse": float(mse),
        "training_time_s": float(training_time),
    }

    for i in range(output_size):
        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9
        rel_err = np.abs((outputs[:, i] - targets[:, i]) / denominator)

        metrics[f"mean_rel_error_k{i+1}"] = float(rel_err.mean())
        metrics[f"max_rel_error_k{i+1}"] = float(rel_err.max())

    return metrics


def save_metrics_files(arch_dir, metrics):
    # JSON
    with open(os.path.join(arch_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # TXT
    lines = [
        f"Scheme: {metrics['scheme']}",
        f"Hidden size: {metrics['hidden_size']}",
        f"Input size: {metrics['input_size']}",
        f"Output size: {metrics['output_size']}",
        f"Activation: {metrics['activation']}",
        f"Learning rate: {metrics['learning_rate']}",
        f"Batch size: {metrics['batch_size']}",
        f"Epochs ran: {metrics['epochs_ran']}",
        f"Final train loss: {metrics['final_train_loss']}",
        f"Final val loss: {metrics['final_val_loss']}",
        f"Best val loss: {metrics['best_val_loss']}",
        f"Test MSE: {metrics['test_mse']}",
        f"Training time (s): {metrics['training_time_s']}",
    ]

    for i in range(metrics["output_size"]):
        lines.append(f"Mean relative error k{i+1}: {metrics[f'mean_rel_error_k{i+1}']}")
        lines.append(f"Max relative error k{i+1}: {metrics[f'max_rel_error_k{i+1}']}")

    with open(os.path.join(arch_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(lines))


def save_predictions_csv(arch_dir, targets, outputs):
    data = {}
    n_outputs = targets.shape[1]

    for i in range(n_outputs):
        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9
        rel_err = np.abs((outputs[:, i] - targets[:, i]) / denominator)

        data[f"k{i+1}_true"] = targets[:, i]
        data[f"k{i+1}_pred"] = outputs[:, i]
        data[f"k{i+1}_rel_err"] = rel_err

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(arch_dir, "predictions.csv"), index=False)


def save_global_summary(results_root, all_metrics):
    lines = []
    lines.append("Architecture comparison summary")
    lines.append("")

    for metrics in all_metrics:
        lines.append(f"Architecture: {metrics['hidden_size']}")
        lines.append(f"  Test MSE: {metrics['test_mse']}")
        lines.append(f"  Best val loss: {metrics['best_val_loss']}")
        lines.append(f"  Final train loss: {metrics['final_train_loss']}")
        lines.append(f"  Final val loss: {metrics['final_val_loss']}")
        lines.append(f"  Training time (s): {metrics['training_time_s']}")
        for i in range(metrics["output_size"]):
            lines.append(f"  Mean rel err k{i+1}: {metrics[f'mean_rel_error_k{i+1}']}")
            lines.append(f"  Max rel err k{i+1}: {metrics[f'max_rel_error_k{i+1}']}")
        lines.append("")

    with open(os.path.join(results_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines))



if __name__ == "__main__":
    torch.manual_seed(8) # for reproducibility

    # Select scheme
    scheme = 'O2_novib'

    # Parameters for data loading
    src_file_train = dictionary[scheme]['main_dataset']
    src_file_test = dictionary[scheme]['main_dataset_test']
    nspecies = dictionary[scheme]['n_densities']
    num_pressure_conditions = dictionary[scheme]['n_conditions']

    # Load the training data
    dataset_train = LoadMultiPressureDatasetTorch(src_file_train, nspecies, num_pressure_conditions, react_idx=dictionary[scheme]['k_columns'])
    x_train, y_train = dataset_train.get_data()

    # Load the test data
    dataset_test = LoadMultiPressureDatasetTorch(src_file_test, nspecies, num_pressure_conditions, 
                                                 react_idx=dictionary[scheme]['k_columns'], scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)
    x_test, y_test = dataset_test.get_data()

    # Check the shape of the data
    print(f"Shape of x_data: {x_train.shape}") # (2000, 9)
    print(f"Shape of y_data: {y_train.shape}") # (2000, 3)


    # Define the network
    input_size = int(nspecies*num_pressure_conditions)  # 11 densities per each pressure condition
    output_size = len(dictionary[scheme]['k_columns'])  # 3 coefficients
    



    # Experiment Setup
    experiment_name = "all_inputs"

    architectures = [
        (30, 30),
        (50, 50),
        (30, 30, 30),
        (30, 30, 30, 30),
    ]

    results_root = make_results_root(
        base_root="Results_NN",
        scheme=scheme,
        experiment_name=experiment_name,
        add_timestamp=True
    )

    arch_dirs = prepare_results_folders(architectures, root=results_root)

    results = []

    activation = "tanh"
    learning_rate = 0.0001
    batch_size = 16

    for hidden_size in architectures:
        print(f"\n--- Testing architecture: {hidden_size} ---")

        arch_dir = arch_dirs[hidden_size]

        torch.manual_seed(43)

        model = NeuralNet(input_size, output_size, hidden_size, activ_f=activation)

        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        # Shuffle already happens inside 'train_model' (therefore, changed shuffle=True to shuffle=False) 
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

        start = time.time()

        model, loss_history = train_model(model, criterion, optimizer, train_loader, num_epochs=5000, patience=100, val_split=0.1)

        end = time.time()

        test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
        targets, outputs, mse = evaluate_model(model, test_data)

        # Save model
        torch.save(model.state_dict(), os.path.join(arch_dir, "model.pth"))

        # Compute metrics
        metrics = compute_metrics_dict(
            scheme=scheme,
            hidden_size=hidden_size,
            input_size=input_size,
            output_size=output_size,
            activation=activation,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_history=loss_history,
            training_time=end - start,
            mse=mse,
            targets=targets.numpy(),
            outputs=outputs.numpy()
        )

        # Save metrics in JSON and TXT
        save_metrics_files(arch_dir, metrics)

        # Save predictions
        save_predictions_csv(arch_dir, targets.numpy(), outputs.numpy())

        # Save plots
        plot_results(targets.numpy(), outputs.numpy(), output_size, arch_dir)
        plot_loss_curves(loss_history, arch_dir, log_scale=True)

        results.append(metrics)

        print(f"Architecture: {hidden_size}")
        print(f"Training time: {end - start}s")
        print(f"Test MSE: {mse}")

    save_global_summary(results_root, results)
