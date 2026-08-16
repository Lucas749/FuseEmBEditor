"""Training and evaluation for the paper benchmark models."""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    BEDICT_PARAMS,
    BEDICT_V2,
    CRISPRON_BE,
    CRISPRON_PARAMS,
    DEEPBASEEDITOR,
    DEEPBE_PAM,
    DEEPBE_PARAMS,
    FORECAST_BE,
    FORECAST_PARAMS,
    IGRNA_ABE,
    IGRNA_PARAMS,
    MODEL_NAMES,
    SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    UPSTREAM_REPOSITORIES,
    VAL_SPLIT,
    get_deepbase_params,
)
from .adapters import build_crispron, build_deepbaseeditor, build_deepbe
from .architectures import build_bedict, build_igrna
from .data import prepare_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _set_keras_seed(seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def regression_metrics(actuals, predictions):
    mse = mean_squared_error(actuals, predictions)
    return {
        "pearson_r": float(pearsonr(actuals, predictions).statistic),
        "spearman_rho": float(spearmanr(actuals, predictions).statistic),
        "r2": float(r2_score(actuals, predictions)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(actuals, predictions)),
    }


def _evaluate_keras(model, splits):
    return {
        split: regression_metrics(
            splits[split]["labels"],
            model.predict(splits[split]["inputs"], verbose=0).reshape(-1),
        )
        for split in SPLITS
    }


def _fit_keras(model, splits, params, epochs, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            restore_best_weights=True,
        )
    ]
    if "scheduler_factor" in params:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=params["scheduler_factor"],
                patience=params["scheduler_patience"],
                min_lr=params["min_learning_rate"],
            )
        )
    model.fit(
        splits[TRAIN_SPLIT]["inputs"],
        splits[TRAIN_SPLIT]["labels"],
        validation_data=(
            splits[VAL_SPLIT]["inputs"],
            splits[VAL_SPLIT]["labels"],
        ),
        epochs=epochs,
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )
    return model, _evaluate_keras(model, splits)


def _fit_forecast(splits):
    params = dict(FORECAST_PARAMS)
    model = GradientBoostingRegressor(**params)
    model.fit(splits[TRAIN_SPLIT]["inputs"], splits[TRAIN_SPLIT]["labels"])
    metrics = {
        split: regression_metrics(
            splits[split]["labels"],
            model.predict(splits[split]["inputs"]),
        )
        for split in SPLITS
    }
    return model, metrics, params


def _fit_deepbase(dataset, splits, params, epochs, seed):
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    model = build_deepbaseeditor(dataset)
    labels = {
        split: splits[split]["labels"].reshape(-1, 1)
        for split in SPLITS
    }
    session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})

    with tf.compat.v1.Session(config=session_config) as session:
        session.run(tf.compat.v1.global_variables_initializer())
        best_val_r2 = -float("inf")
        stale_epochs = 0
        for _ in range(epochs):
            indices = np.random.permutation(len(labels[TRAIN_SPLIT]))
            for start in range(0, len(indices), params["batch_size"]):
                batch_indices = indices[start:start + params["batch_size"]]
                session.run(
                    model.optimizer,
                    feed_dict={
                        model.inputs: splits[TRAIN_SPLIT]["inputs"][batch_indices],
                        model.targets: labels[TRAIN_SPLIT][batch_indices],
                        model.is_training: True,
                    },
                )
            val_predictions = session.run(
                model.outputs,
                feed_dict={
                    model.inputs: splits[VAL_SPLIT]["inputs"],
                    model.is_training: False,
                },
            ).reshape(-1)
            val_r2 = r2_score(splits[VAL_SPLIT]["labels"], val_predictions)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= params["patience"]:
                    break

        metrics = {}
        for split in SPLITS:
            predictions = session.run(
                model.outputs,
                feed_dict={
                    model.inputs: splits[split]["inputs"],
                    model.is_training: False,
                },
            ).reshape(-1)
            metrics[split] = regression_metrics(
                splits[split]["labels"],
                predictions,
            )
        state = {
            variable.name: session.run(variable)
            for variable in tf.compat.v1.trainable_variables()
        }
    return state, metrics


def _torch_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bedict_predictions(model, inputs, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(inputs)),
        batch_size=batch_size,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            probabilities = torch.softmax(model(batch.to(device)), dim=1)
            predictions.extend(probabilities[:, 1].cpu().numpy())
    return np.asarray(predictions)


def _fit_bedict(splits, epochs, device_name):
    device = _torch_device(device_name)
    model = build_bedict().to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(splits[TRAIN_SPLIT]["inputs"]),
        torch.from_numpy(splits[TRAIN_SPLIT]["labels"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BEDICT_PARAMS["batch_size"],
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=BEDICT_PARAMS["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=BEDICT_PARAMS["learning_rate"],
        max_lr=BEDICT_PARAMS["learning_rate"] * BEDICT_PARAMS["scheduler_multiplier"],
        step_size_up=max(
            1,
            len(train_loader) * BEDICT_PARAMS["scheduler_multiplier"],
        ),
        mode="triangular",
        cycle_momentum=False,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_spearman = -float("inf")
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            target_classes = (labels > 0.5).long()
            targets = torch.zeros_like(outputs)
            targets.scatter_(1, target_classes[:, None], 1.0)
            loss = criterion(torch.log_softmax(outputs, dim=1), targets)
            l2_loss = sum(torch.sum(parameter ** 2) for parameter in model.parameters())
            loss = loss + BEDICT_PARAMS["l2_reg"] * l2_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_predictions = _bedict_predictions(
            model,
            splits[VAL_SPLIT]["inputs"],
            BEDICT_PARAMS["batch_size"],
            device,
        )
        val_spearman = spearmanr(
            splits[VAL_SPLIT]["labels"],
            val_predictions,
        ).statistic
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= BEDICT_PARAMS["patience"]:
                break

    metrics = {
        split: regression_metrics(
            splits[split]["labels"],
            _bedict_predictions(
                model,
                splits[split]["inputs"],
                BEDICT_PARAMS["batch_size"],
                device,
            ),
        )
        for split in SPLITS
    }
    return model, metrics, str(device)


def _effective_epochs(model, dataset, override):
    if override is not None:
        return override
    if model == DEEPBASEEDITOR:
        return get_deepbase_params(dataset)["epochs"]
    params = {
        IGRNA_ABE: IGRNA_PARAMS,
        BEDICT_V2: BEDICT_PARAMS,
        DEEPBE_PAM: DEEPBE_PARAMS,
        CRISPRON_BE: CRISPRON_PARAMS,
    }
    return params[model]["epochs"]


def _save_checkpoint(model_name, model, output_dir):
    if model_name == FORECAST_BE:
        joblib.dump(model, output_dir / "model.joblib")
    elif model_name == DEEPBASEEDITOR:
        np.savez(output_dir / "model.npz", **model)
    elif model_name == BEDICT_V2:
        torch.save(model.state_dict(), output_dir / "model.pt")
    else:
        model.save(output_dir / "model.keras")


def run_benchmark(
    model_name,
    dataset,
    output_root,
    seed,
    device="auto",
    epochs=None,
    max_samples=None,
    save_checkpoint=False,
):
    set_seed(seed)
    splits, metadata = prepare_dataset(model_name, dataset, max_samples=max_samples)
    effective_epochs = None
    training_config = {}

    if model_name == FORECAST_BE:
        model, metrics, training_config = _fit_forecast(splits)
    elif model_name == DEEPBASEEDITOR:
        effective_epochs = _effective_epochs(model_name, dataset, epochs)
        params = get_deepbase_params(dataset)
        model, metrics = _fit_deepbase(
            dataset,
            splits,
            params,
            effective_epochs,
            seed,
        )
        training_config = params
    elif model_name == IGRNA_ABE:
        effective_epochs = _effective_epochs(model_name, dataset, epochs)
        _set_keras_seed(seed)
        model = build_igrna()
        model, metrics = _fit_keras(
            model,
            splits,
            IGRNA_PARAMS,
            effective_epochs,
            seed,
        )
        training_config = IGRNA_PARAMS
    elif model_name == BEDICT_V2:
        effective_epochs = _effective_epochs(model_name, dataset, epochs)
        model, metrics, resolved_device = _fit_bedict(
            splits,
            effective_epochs,
            device,
        )
        training_config = {**BEDICT_PARAMS, "device": resolved_device}
    elif model_name == DEEPBE_PAM:
        effective_epochs = _effective_epochs(model_name, dataset, epochs)
        _set_keras_seed(seed)
        model = build_deepbe()
        model, metrics = _fit_keras(
            model,
            splits,
            DEEPBE_PARAMS,
            effective_epochs,
            seed,
        )
        training_config = DEEPBE_PARAMS
    elif model_name == CRISPRON_BE:
        effective_epochs = _effective_epochs(model_name, dataset, epochs)
        _set_keras_seed(seed)
        model = build_crispron(len(metadata["source_vocab"]))
        model, metrics = _fit_keras(
            model,
            splits,
            CRISPRON_PARAMS,
            effective_epochs,
            seed,
        )
        training_config = {
            **CRISPRON_PARAMS,
            "source_vocab": metadata["source_vocab"],
        }
    else:
        raise ValueError(f"Unknown benchmark model: {model_name}")

    if effective_epochs is not None:
        training_config = {**training_config, "epochs": effective_epochs}
    output_dir = Path(output_root) / model_name / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        _save_checkpoint(model_name, model, output_dir)

    result = {
        "model": MODEL_NAMES[model_name],
        "upstream": UPSTREAM_REPOSITORIES[model_name],
        "dataset": dataset,
        "seed": seed,
        "data": metadata,
        "config": training_config,
        "metrics": metrics,
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(result, handle, indent=2)
    return result
