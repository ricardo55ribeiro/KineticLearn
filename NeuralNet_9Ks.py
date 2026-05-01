import copy
import json
import os
import time
from datetime import datetime
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.NeuralNetworkModels import NeuralNet


class LoadMultiPressureDatasetTorch(torch.utils.data.Dataset):
    def __init__(
        self,
        src_file,
        nspecies,
        num_pressure_conditions,
        react_idx=None,
        m_rows=None,
        columns=None,
        scaler_input=None,
        scaler_output=None,
    ):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(
            src_file,
            max_rows=m_rows,
            usecols=columns,
            delimiter=None,
            comments="#",
            skiprows=0,
            dtype=np.float64,
        )

        if all_data.ndim == 1:
            all_data = all_data.reshape(1, -1)

        if len(all_data) % num_pressure_conditions != 0:
            raise ValueError(
                f"The number of rows in {src_file} ({len(all_data)}) is not divisible by "
                f"num_pressure_conditions ({num_pressure_conditions})."
            )

        ncolumns = len(all_data[0])

        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1)  # densities
        y_columns = react_idx  # k's
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        raw_x_data = all_data[:, x_columns].copy()
        raw_y_data = all_data[:, y_columns].copy()

        x_data = raw_x_data.copy()
        y_data = raw_y_data * 1e30

        # Reshape scaled data
        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        # Reshape unscaled/original data
        raw_x_data = raw_x_data.reshape(num_pressure_conditions, -1, raw_x_data.shape[1])
        raw_y_data = raw_y_data.reshape(num_pressure_conditions, -1, raw_y_data.shape[1])

        # Create scalers
        self.scaler_input = scaler_input or [
            preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)
        ]
        self.scaler_output = scaler_output or [
            preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)
        ]

        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])

            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        # Flatten scaled data
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(
            -1, self.num_pressure_conditions * x_data.shape[-1]
        )
        y_data = y_data[0]

        # Flatten unscaled/original data
        raw_x_data = np.transpose(raw_x_data, (1, 0, 2)).reshape(
            -1, self.num_pressure_conditions * raw_x_data.shape[-1]
        )
        raw_y_data = raw_y_data[0]

        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

        self.x_data_unscaled = torch.from_numpy(raw_x_data).float()
        self.y_data_unscaled = torch.from_numpy(raw_y_data).float()

    def get_unscaled_data(self):
        return self.x_data_unscaled, self.y_data_unscaled

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def get_data(self):
        return self.x_data, self.y_data


def train_model(model, criterion, optimizer, dataloader, num_epochs=100, patience=5, val_split=0.1):
    train_len = int((1.0 - val_split) * len(dataloader.dataset))
    val_len = len(dataloader.dataset) - train_len

    split_generator = torch.Generator().manual_seed(43)
    train_dataset, val_dataset = random_split(
        dataloader.dataset,
        [train_len, val_len],
        generator=split_generator,
    )

    shuffle_generator = torch.Generator().manual_seed(43)
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    min_val_loss = np.inf
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1}, Training loss: {train_loss}, Validation loss: {val_loss}")

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                model.load_state_dict(best_model_wts)
                return model, history

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, test_data):
    model.eval()

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)
            all_targets.append(targets)
            all_outputs.append(outputs)

    targets = torch.cat(all_targets, dim=0)
    outputs = torch.cat(all_outputs, dim=0)

    mse = mean_squared_error(targets.numpy(), outputs.numpy())
    print(f"Mean Squared Error (MSE) on the test data: {mse}")

    return targets, outputs, mse


def plot_results(targets, outputs, output_size, output_dir):
    plt.rcParams.update({"font.size": 16, "text.usetex": False})

    ncols = 3
    nrows = ceil(output_size / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axs = np.atleast_1d(axs).flatten()

    for i in range(output_size):
        ax = axs[i]

        ax.scatter(targets[:, i], outputs[:, i], alpha=0.8, color=(0.0, 0.0, 0.9))

        line_min = min(targets[:, i].min(), outputs[:, i].min())
        line_max = max(targets[:, i].max(), outputs[:, i].max())
        ax.plot([line_min, line_max], [line_min, line_max], "--", color="black")

        ax.set_xlabel("True Values", fontsize=14)
        ax.set_ylabel("Predicted Values", fontsize=14)
        ax.set_title(f"$k_{{{i + 1}}}$")

        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9
        rel_err = np.abs((outputs[:, i] - targets[:, i]) / denominator)

        textstr = "\n".join(
            (
                r"$Mean\ \delta_{rel}=%.2f\%%$" % (rel_err.mean() * 100,),
                r"$Max\ \delta_{rel}=%.2f\%%$" % (rel_err.max() * 100,),
            )
        )

        max_index = np.argmax(rel_err)
        ax.scatter(targets[max_index, i], outputs[max_index, i], color="gold", zorder=2)

        props = dict(boxstyle="round", alpha=0.5)
        ax.text(
            0.58,
            0.25,
            textstr,
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )

    for j in range(output_size, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "NeuralNet.pdf"))
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
    plt.rcParams.update({"font.size": 16, "text.usetex": False})

    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    epochs = np.arange(1, len(train_loss) + 1)

    train_smooth = moving_average(train_loss, window=25)
    val_smooth = moving_average(val_loss, window=25)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_smooth, linewidth=1.8, label="Training Loss")
    plt.plot(epochs, val_smooth, linewidth=1.8, label="Validation Loss")

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("MSE Loss", fontsize=16)

    if log_scale:
        plt.yscale("log")

    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=13, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "NeuralNet_loss_curves.pdf"))
    plt.close()


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
    experiment_name,
    run_timestamp,
    hidden_size,
    input_size,
    output_size,
    activation,
    learning_rate,
    batch_size,
    num_pressure_conditions,
    num_species_total,
    num_species_kept,
    kept_species,
    removed_species,
    num_parameters,
    loss_history,
    training_time,
    mse,
    mse_unscaled,
    rmse_unscaled,
    targets,
    outputs,
):
    metrics = {
        "scheme": scheme,
        "experiment_name": experiment_name,
        "run_timestamp": run_timestamp,
        "hidden_size": list(hidden_size),
        "depth": len(hidden_size),
        "num_parameters": int(num_parameters),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "num_pressure_conditions": int(num_pressure_conditions),
        "num_species_total": int(num_species_total),
        "num_species_kept": int(num_species_kept),
        "kept_species": kept_species,
        "removed_species": removed_species,
        "activation": activation,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs_ran": len(loss_history["train_loss"]),
        "best_epoch": int(np.argmin(loss_history["val_loss"]) + 1),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
        "test_mse": float(mse),
        "test_rmse": float(np.sqrt(mse)),
        "test_mse_unscaled": float(mse_unscaled),
        "test_rmse_unscaled": float(rmse_unscaled),
        "training_time_s": float(training_time),
    }

    for i in range(output_size):
        denominator = outputs[:, i].copy()
        denominator[np.abs(denominator) < 1e-9] = 1e-9
        rel_err = np.abs((outputs[:, i] - targets[:, i]) / denominator)

        metrics[f"mean_rel_error_k{i + 1}"] = float(rel_err.mean())
        metrics[f"max_rel_error_k{i + 1}"] = float(rel_err.max())

    return metrics


def save_metrics_files(arch_dir, metrics):
    with open(os.path.join(arch_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    lines = [
        f"Experiment name: {metrics['experiment_name']}",
        f"Run timestamp: {metrics['run_timestamp']}",
        f"Scheme: {metrics['scheme']}",
        f"Hidden size: {metrics['hidden_size']}",
        f"Input size: {metrics['input_size']}",
        f"Kept species: {metrics['kept_species']}",
        f"Removed species: {metrics['removed_species']}",
        f"Test MSE: {metrics['test_mse']}",
        f"Test MSE (unscaled): {metrics['test_mse_unscaled']}",
        f"Training time (s): {metrics['training_time_s']}",
    ]

    for i in range(metrics["output_size"]):
        lines.append(f"Mean relative error k{i + 1}: {metrics[f'mean_rel_error_k{i + 1}']}")
        lines.append(f"Max relative error k{i + 1}: {metrics[f'max_rel_error_k{i + 1}']}")

    with open(os.path.join(arch_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(lines))


def save_predictions_csv(
    arch_dir,
    targets_scaled,
    outputs_scaled,
    targets_unscaled,
    outputs_unscaled,
):
    data = {"sample_id": np.arange(len(targets_scaled))}
    n_outputs = targets_scaled.shape[1]

    for i in range(n_outputs):
        denominator = outputs_unscaled[:, i].copy()
        denominator[np.abs(denominator) < 1e-30] = 1e-30

        abs_err = np.abs(outputs_unscaled[:, i] - targets_unscaled[:, i])
        sq_err = (outputs_unscaled[:, i] - targets_unscaled[:, i]) ** 2
        rel_err = np.abs((outputs_unscaled[:, i] - targets_unscaled[:, i]) / denominator)

        data[f"k{i + 1}_true_scaled"] = targets_scaled[:, i]
        data[f"k{i + 1}_pred_scaled"] = outputs_scaled[:, i]

        data[f"k{i + 1}_true_unscaled"] = targets_unscaled[:, i]
        data[f"k{i + 1}_pred_unscaled"] = outputs_unscaled[:, i]

        data[f"k{i + 1}_abs_err"] = abs_err
        data[f"k{i + 1}_sq_err"] = sq_err
        data[f"k{i + 1}_rel_err"] = rel_err

    pd.DataFrame(data).to_csv(os.path.join(arch_dir, "predictions.csv"), index=False)


def save_global_summary(results_root, all_metrics):
    lines = []
    lines.append("Architecture comparison summary")
    lines.append("")

    for metrics in all_metrics:
        lines.append(f"Architecture: {metrics['hidden_size']}")
        lines.append(f"  Input size: {metrics['input_size']}")
        lines.append(f"  Kept species: {metrics['kept_species']}")
        lines.append(f"  Removed species: {metrics['removed_species']}")
        lines.append(f"  Test MSE: {metrics['test_mse']}")
        lines.append(f"  Test MSE (unscaled): {metrics['test_mse_unscaled']}")
        lines.append(f"  Training time (s): {metrics['training_time_s']}")
        for i in range(metrics["output_size"]):
            lines.append(f"  Mean rel err k{i + 1}: {metrics[f'mean_rel_error_k{i + 1}']}")
            lines.append(f"  Max rel err k{i + 1}: {metrics[f'max_rel_error_k{i + 1}']}")
        lines.append("")

    with open(os.path.join(results_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines))


def build_feature_names(species_names, num_pressure_conditions):
    if num_pressure_conditions == 1:
        return list(species_names)

    return [
        f"{species}_p{p + 1}"
        for p in range(num_pressure_conditions)
        for species in species_names
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(filepath, obj):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4)


def save_experiment_info(results_root, info):
    save_json(os.path.join(results_root, "experiment_info.json"), info)


def save_summary_csv(results_root, all_metrics):
    df = pd.DataFrame(all_metrics)

    if "hidden_size" in df.columns:
        df["hidden_size"] = df["hidden_size"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
        )

    if "kept_species" in df.columns:
        df["kept_species"] = df["kept_species"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    if "removed_species" in df.columns:
        df["removed_species"] = df["removed_species"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    df.to_csv(os.path.join(results_root, "summary.csv"), index=False)


def save_loss_history_csv(arch_dir, history):
    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_loss_smooth": moving_average(history["train_loss"], window=25),
            "val_loss_smooth": moving_average(history["val_loss"], window=25),
        }
    )
    df.to_csv(os.path.join(arch_dir, "loss_history.csv"), index=False)


def save_model_info(arch_dir, model, hidden_size):
    model_info = {
        "hidden_size": list(hidden_size),
        "depth": len(hidden_size),
        "num_parameters": count_parameters(model),
    }
    save_json(os.path.join(arch_dir, "model_info.json"), model_info)


def save_test_inputs_csv(results_root, x_test_unscaled, feature_names):
    df = pd.DataFrame(x_test_unscaled, columns=feature_names)
    df.insert(0, "sample_id", np.arange(len(df)))
    df.to_csv(os.path.join(results_root, "test_inputs.csv"), index=False)


if __name__ == "__main__":
    scheme = "9Ks_Dataset"

    src_file_train = os.path.join("Old Datasets", "datapoints_all.txt")
    src_file_test = os.path.join("Old Datasets", "datapoints_all_test.txt")

    if not os.path.exists(src_file_train):
        raise FileNotFoundError(f"Training file not found: {src_file_train}")
    if not os.path.exists(src_file_test):
        raise FileNotFoundError(f"Test file not found: {src_file_test}")

    nspecies = 3
    num_pressure_conditions = 1
    k_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    all_species = ["O2(X)", "O2(a)", "O(3P)"]
    kept_species = ["O2(X)", "O2(a)", "O(3P)"]
    removed_species = []

    dataset_train = LoadMultiPressureDatasetTorch(
        src_file_train,
        nspecies,
        num_pressure_conditions,
        react_idx=k_columns,
    )
    x_train, y_train = dataset_train.get_data()

    dataset_test = LoadMultiPressureDatasetTorch(
        src_file_test,
        nspecies,
        num_pressure_conditions,
        react_idx=k_columns,
        scaler_input=dataset_train.scaler_input,
        scaler_output=dataset_train.scaler_output,
    )
    x_test, y_test = dataset_test.get_data()

    print(f"Shape of x_data: {x_train.shape}")
    print(f"Shape of y_data: {y_train.shape}")

    input_size = x_train.shape[1]
    output_size = len(k_columns)

    experiment_name = "3species_to_9Ks"

    activation = "tanh"
    learning_rate = 0.0001
    batch_size = 16

    architectures = [
        (30, 30),
        (50, 50),
        (30, 30, 30),
    ]

    results_root = make_results_root(
        base_root="Results_NN",
        scheme=scheme,
        experiment_name=experiment_name,
        add_timestamp=True,
    )

    arch_dirs = prepare_results_folders(architectures, root=results_root)
    results = []

    num_species_total = len(all_species)
    num_species_kept = len(kept_species)

    run_timestamp = os.path.basename(results_root)
    feature_names = build_feature_names(kept_species, num_pressure_conditions)

    experiment_info = {
        "scheme": scheme,
        "experiment_name": experiment_name,
        "run_timestamp": run_timestamp,
        "train_file": src_file_train,
        "test_file": src_file_test,
        "num_pressure_conditions": num_pressure_conditions,
        "num_species_total": num_species_total,
        "num_species_kept": num_species_kept,
        "species_all": all_species,
        "kept_species": kept_species,
        "removed_species": removed_species,
        "k_columns": k_columns,
        "architectures_tested": [list(a) for a in architectures],
        "activation": activation,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "patience": 100,
        "max_epochs": 5000,
        "split_seed": 43,
        "shuffle_seed": 43,
        "weight_seed": 43,
    }
    save_experiment_info(results_root, experiment_info)

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    save_test_inputs_csv(results_root, x_test_unscaled.numpy(), feature_names)

    for hidden_size in architectures:
        print(f"\n--- Testing architecture: {hidden_size} ---")

        arch_dir = arch_dirs[hidden_size]

        torch.manual_seed(43)

        model = NeuralNet(input_size, output_size, hidden_size, activ_f=activation)

        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

        start = time.time()
        model, loss_history = train_model(
            model,
            criterion,
            optimizer,
            train_loader,
            num_epochs=5000,
            patience=100,
            val_split=0.1,
        )
        end = time.time()

        test_data = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
        targets, outputs, mse = evaluate_model(model, test_data)

        targets_scaled = targets.numpy()
        outputs_scaled = outputs.numpy()

        targets_unscaled = dataset_test.y_data_unscaled.numpy()
        outputs_unscaled = (
            dataset_test.scaler_output[0].inverse_transform(outputs_scaled) / 1e30
        )

        mse_unscaled = mean_squared_error(targets_unscaled, outputs_unscaled)
        rmse_unscaled = np.sqrt(mse_unscaled)

        torch.save(model.state_dict(), os.path.join(arch_dir, "model.pth"))

        metrics = compute_metrics_dict(
            scheme=scheme,
            experiment_name=experiment_name,
            run_timestamp=run_timestamp,
            hidden_size=hidden_size,
            input_size=input_size,
            output_size=output_size,
            activation=activation,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_pressure_conditions=num_pressure_conditions,
            num_species_total=num_species_total,
            num_species_kept=num_species_kept,
            kept_species=kept_species,
            removed_species=removed_species,
            num_parameters=count_parameters(model),
            loss_history=loss_history,
            training_time=end - start,
            mse=mse,
            mse_unscaled=mse_unscaled,
            rmse_unscaled=rmse_unscaled,
            targets=targets_scaled,
            outputs=outputs_scaled,
        )

        save_metrics_files(arch_dir, metrics)
        save_predictions_csv(
            arch_dir,
            targets_scaled=targets_scaled,
            outputs_scaled=outputs_scaled,
            targets_unscaled=targets_unscaled,
            outputs_unscaled=outputs_unscaled,
        )
        save_loss_history_csv(arch_dir, loss_history)
        save_model_info(arch_dir, model, hidden_size)

        plot_results(targets_scaled, outputs_scaled, output_size, arch_dir)
        plot_loss_curves(loss_history, arch_dir, log_scale=True)

        results.append(metrics)

        print(f"Architecture: {hidden_size}")
        print(f"Training time: {end - start}s")
        print(f"Test MSE: {mse}")
        print(f"Test MSE (unscaled): {mse_unscaled}")

    save_global_summary(results_root, results)
    save_summary_csv(results_root, results)
