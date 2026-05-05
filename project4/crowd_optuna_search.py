from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from tqdm import tqdm


IMG_EXT = ".jpg"


# Edit these variables directly, then run: python crowd_optuna_search.py
DATA_ROOT = Path(r"C:\Users\mrsig\Desktop\M508\project4\part_A_final")
SEED = 508
EPOCHS_PER_TRIAL = 15
FINAL_EPOCHS = 20
N_TRIALS = 100
VAL_FRACTION = 0.15
NUM_WORKERS = 0
DEVICE_NAME: str | None = None
STUDY_NAME = "crowd-counting-optuna"
STORAGE: str | None = None
OUTPUT_DIR = Path("optuna_results")
INPUT_SIZES = [224, 320, 384, 512, 640]
BATCH_SIZES = [1, 2, 4, 8]
SIGMA_MIN = 4.0
SIGMA_MAX = 30.0
LR_MIN = 1e-6
LR_MAX = 1e-3
WEIGHT_DECAY_MIN = 1e-7
WEIGHT_DECAY_MAX = 1e-2
FREEZE_FRONTEND = True
RUN_FINAL_TEST = True
SHOW_PLOT = True
SAVE_PLOT = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def get_gt_points_from_mat(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    points = data["image_info"][0, 0][0, 0][0]
    return np.asarray(points, dtype=np.float32)


def make_density_map(points: np.ndarray, image_size: tuple[int, int], sigma: float) -> np.ndarray:
    height, width = image_size
    density = np.zeros((height, width), dtype=np.float32)

    for x, y in points:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < width and 0 <= yi < height:
            density[yi, xi] += 1.0

    density = gaussian_filter(density, sigma=sigma, mode="constant")
    return density.astype(np.float32)


class CrowdDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        gt_dir: Path,
        sigma: float,
        transform: transforms.Compose | None = None,
        use_random_crops: bool = False,
        crop_size: int = 384,
    ) -> None:
        self.image_paths = sorted(image_paths)
        self.gt_dir = gt_dir
        self.sigma = sigma
        self.transform = transform
        self.use_random_crops = use_random_crops
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def _gt_path(self, image_path: Path) -> Path:
        return self.gt_dir / image_path.name.replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[idx]
        gt_path = self._gt_path(image_path)

        image = load_image(image_path)
        points = get_gt_points_from_mat(gt_path)
        width, height = image.size

        if self.use_random_crops:
            crop_h = min(self.crop_size, height)
            crop_w = min(self.crop_size, width)
            top = 0 if height == crop_h else np.random.randint(0, height - crop_h + 1)
            left = 0 if width == crop_w else np.random.randint(0, width - crop_w + 1)

            image = image.crop((left, top, left + crop_w, top + crop_h))

            if len(points) > 0:
                mask = (
                    (points[:, 0] >= left)
                    & (points[:, 0] < left + crop_w)
                    & (points[:, 1] >= top)
                    & (points[:, 1] < top + crop_h)
                )
                points = points[mask].copy()
                points[:, 0] -= left
                points[:, 1] -= top

            pad_h = self.crop_size - crop_h
            pad_w = self.crop_size - crop_w
            if pad_h > 0 or pad_w > 0:
                image_np = np.array(image)
                image_np = np.pad(
                    image_np,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                image = Image.fromarray(image_np)

            height, width = self.crop_size, self.crop_size
        else:
            height, width = image.height, image.width

        density = make_density_map(points, image_size=(height, width), sigma=self.sigma)
        image_tensor = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        count = torch.tensor(float(len(points)), dtype=torch.float32)

        return image_tensor, density_tensor, count, str(image_path)


class CrowdCounter(nn.Module):
    def __init__(self, freeze_frontend: bool = True) -> None:
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:17])

        if freeze_frontend:
            for param in self.frontend.parameters():
                param.requires_grad = False

        dilation = 2
        channels = 64

        backend = [
            nn.Conv2d(256, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
        ]
        for _ in range(5):
            backend.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )

        self.backend = nn.Sequential(*backend)
        self.final = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        z = self.frontend(x)
        z = self.backend(z)
        z = self.final(z)
        return F.interpolate(z, size=(height, width), mode="bilinear", align_corners=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, density_gt, _counts_gt, _paths in loader:
        images = images.to(device)
        density_gt = density_gt.to(device)

        optimizer.zero_grad(set_to_none=True)
        density_pred = model(images)
        loss = criterion(density_pred, density_gt) / images.size(0)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    abs_errors: list[float] = []
    sq_errors: list[float] = []

    for images, _density_gt, counts_gt, _paths in loader:
        images = images.to(device)
        density_pred = model(images)

        count_pred = float(density_pred.sum().cpu().item())
        count_gt = float(counts_gt.item())

        error = abs(count_gt - count_pred)
        abs_errors.append(error)
        sq_errors.append(error**2)

    mae = float(np.mean(abs_errors)) if abs_errors else float("inf")
    rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("inf")
    return mae, rmse


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    history = {"train_loss": [], "mae": [], "rmse": []}
    best_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    progress = tqdm(range(1, epochs + 1), desc="Epoch", leave=False)
    for epoch in progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        mae, rmse = evaluate_model(model, eval_loader, device)

        history["train_loss"].append(train_loss)
        history["mae"].append(mae)
        history["rmse"].append(rmse)

        if mae < best_mae:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())

        if trial is not None:
            trial.report(mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        progress.set_postfix(
            epoch=f"{epoch:02d}",
            train_loss=f"{train_loss:.3f}",
            mae=f"{mae:.3f}",
            rmse=f"{rmse:.3f}",
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_mae"] = best_mae
    return history


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def split_paths(image_dir: Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    image_paths = sorted(image_dir.glob(f"*{IMG_EXT}"))
    if len(image_paths) < 2:
        raise ValueError("Need at least two training images to create a validation split.")

    shuffled = image_paths[:]
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val_paths = sorted(shuffled[:val_size])
    train_paths = sorted(shuffled[val_size:])
    return train_paths, val_paths


def build_loaders(
    params: dict[str, Any],
    *,
    use_full_train_split: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    train_images = DATA_ROOT / "train_data" / "images"
    train_gt = DATA_ROOT / "train_data" / "ground_truth"
    test_images = DATA_ROOT / "test_data" / "images"
    test_gt = DATA_ROOT / "test_data" / "ground_truth"

    base_transform = image_transform()
    all_train_paths = sorted(train_images.glob(f"*{IMG_EXT}"))
    split_train_paths, split_val_paths = split_paths(train_images, VAL_FRACTION, SEED)

    if use_full_train_split:
        train_paths = all_train_paths
        eval_paths = None
    else:
        train_paths = split_train_paths
        eval_paths = split_val_paths

    train_ds = CrowdDataset(
        train_paths,
        train_gt,
        sigma=float(params["sigma"]),
        transform=base_transform,
        use_random_crops=True,
        crop_size=int(params["input_size"]),
    )
    pin_memory = DEVICE_NAME != "cpu" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    eval_loader = None
    if eval_paths is not None:
        eval_ds = CrowdDataset(
            eval_paths,
            train_gt,
            sigma=float(params["sigma"]),
            transform=base_transform,
            use_random_crops=False,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

    test_loader = None
    if use_full_train_split:
        test_ds = CrowdDataset(
            sorted(test_images.glob(f"*{IMG_EXT}")),
            test_gt,
            sigma=float(params["sigma"]),
            transform=base_transform,
            use_random_crops=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

    return train_loader, eval_loader, test_loader


def make_model_and_optimizer(
    params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    model = CrowdCounter(freeze_frontend=FREEZE_FRONTEND).to(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.MSELoss(reduction="sum")
    return model, optimizer, criterion


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "sigma": trial.suggest_float("sigma", SIGMA_MIN, SIGMA_MAX),
        "lr": trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            WEIGHT_DECAY_MIN,
            WEIGHT_DECAY_MAX,
            log=True,
        ),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
        "input_size": trial.suggest_categorical("input_size", INPUT_SIZES),
    }


def json_safe_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "system_attrs": trial.system_attrs,
    }


def save_study_artifacts(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    (output_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    trials_payload = [json_safe_trial(trial) for trial in study.trials]
    (output_dir / "trials.json").write_text(json.dumps(trials_payload, indent=2), encoding="utf-8")


def plot_study_progress(study: optuna.Study, output_dir: Path) -> None:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed_trials:
        return

    trial_numbers = [trial.number for trial in completed_trials]
    mae_values = np.array([float(trial.value) for trial in completed_trials], dtype=np.float32)
    rmse_values = np.array(
        [float(trial.user_attrs["best_rmse"]) for trial in completed_trials],
        dtype=np.float32,
    )
    best_mae_so_far = np.minimum.accumulate(mae_values)
    best_rmse_so_far = np.minimum.accumulate(rmse_values)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(trial_numbers, mae_values, marker="o", linewidth=1.5, label="Validation MAE")
    axes[0].plot(trial_numbers, best_mae_so_far, linewidth=2.5, label="Best MAE so far")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Optuna Search Performance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(trial_numbers, rmse_values, marker="o", linewidth=1.5, label="Validation RMSE")
    axes[1].plot(trial_numbers, best_rmse_so_far, linewidth=2.5, label="Best RMSE so far")
    axes[1].set_xlabel("Optuna trial")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    if SAVE_PLOT:
        fig.savefig(output_dir / "optuna_trial_performance.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def run_final_training(
    params: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> None:
    print("\nRetraining best configuration on the full training split...")
    set_seed(SEED)
    train_loader, _val_loader, test_loader = build_loaders(params, use_full_train_split=True)
    if test_loader is None:
        raise RuntimeError("Test loader was not created for final evaluation.")

    model, optimizer, criterion = make_model_and_optimizer(params, device)
    train_history = {"train_loss": []}
    progress = tqdm(range(1, FINAL_EPOCHS + 1), desc="Final train", leave=False)
    for epoch in progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_history["train_loss"].append(train_loss)
        progress.set_postfix(
            epoch=f"{epoch:02d}",
            train_loss=f"{train_loss:.3f}",
        )

    test_mae, test_rmse = evaluate_model(model, test_loader, device)

    payload = {
        "params": params,
        "train_history": train_history,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Final test MAE = {test_mae:.3f} | Final test RMSE = {test_rmse:.3f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = resolve_device(DEVICE_NAME)
    print(f"Using device: {device}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=bool(STORAGE),
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED + trial.number)
        params = suggest_params(trial)
        trial.set_user_attr("params", params)

        train_loader, val_loader, _test_loader = build_loaders(params, use_full_train_split=False)
        if val_loader is None:
            raise RuntimeError("Validation loader was not created for the Optuna objective.")
        model, optimizer, criterion = make_model_and_optimizer(params, device)

        history = fit_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epochs=EPOCHS_PER_TRIAL,
            device=device,
            trial=trial,
        )
        best_mae = float(history["best_mae"])
        trial.set_user_attr("best_rmse", min(history["rmse"]))
        return best_mae

    study.optimize(objective, n_trials=N_TRIALS)
    completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    if not completed:
        raise RuntimeError("Optuna finished without any completed trials. Try fewer epochs or a looser pruner.")

    save_study_artifacts(study, OUTPUT_DIR)
    plot_study_progress(study, OUTPUT_DIR)
    print(
        f"\nStudy complete: {len(completed)} completed, {len(pruned)} pruned, "
        f"best validation MAE = {study.best_value:.3f}"
    )
    print("Best params:")
    print(json.dumps(study.best_params, indent=2))

    if RUN_FINAL_TEST:
        run_final_training(study.best_params, device, OUTPUT_DIR)


if __name__ == "__main__":
    main()
