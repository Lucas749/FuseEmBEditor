"""Training loop, evaluation and embedding scaling."""

import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import (
    ADAM_EPS,
    EPOCHS,
    LABEL_DECIMALS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    TEST_SPLIT,
    WEIGHT_DECAY,
)


def set_seed(seed):
    """Seed every source of randomness that affects a run.

    Weight initialisation and batch order are the only stochastic elements, since
    the encoders are frozen and their embeddings precomputed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standard_scale(train_emb, val_emb, test_emb):
    """Fit a StandardScaler on train and apply it to every split."""
    scaler = StandardScaler()
    train = torch.from_numpy(scaler.fit_transform(train_emb.numpy())).float()
    val = torch.from_numpy(scaler.transform(val_emb.numpy())).float()
    test = torch.from_numpy(scaler.transform(test_emb.numpy())).float()
    return train, val, test, scaler


def train_model(model, train_loader, val_loader, device,
                num_epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Train the head and return one metrics record per epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    criterion = nn.MSELoss()

    epoch_metrics = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(embeddings), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embeddings"].to(device)
                labels = batch["labels"].to(device)
                total_val_loss += criterion(model(embeddings), labels).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")
        epoch_metrics.append({"epoch": epoch + 1,
                              "train_loss": avg_train_loss,
                              "val_loss": avg_val_loss})

    return epoch_metrics


def evaluate_model(model, data_loader, device, split_name=TEST_SPLIT):
    """Return a metrics dict plus the raw predictions and targets.

    Predictions and labels are rounded to two decimals, matching the precision the
    efficiency measurements are reported at.
    """
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            preds = model(batch["embeddings"].to(device))
            predictions.extend(
                round(float(p), LABEL_DECIMALS) for p in preds.cpu().numpy())
            actuals.extend(
                round(float(a), LABEL_DECIMALS) for a in batch["labels"].numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, predictions)
    metrics = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "r2": float(r2_score(actuals, predictions)),
        "corr": float(pearsonr(actuals, predictions)[0]),
        "spearman": float(spearmanr(actuals, predictions)[0]),
    }

    print(f"\n{split_name} metrics: "
          f"Pearson {metrics['corr']:.4f} | Spearman {metrics['spearman']:.4f} | "
          f"R2 {metrics['r2']:.4f} | RMSE {metrics['rmse']:.4f}")
    return metrics, predictions, actuals
