"""
Plant leaf disease classification with handcrafted image features.

Expected dataset layout:
data/PlantVillage/.../Potato___Early_blight/*.jpg
data/PlantVillage/.../Potato___Late_blight/*.jpg
data/PlantVillage/.../Potato___healthy/*.jpg

The script also searches recursively below --data-dir for class folders.
It creates outputs/results.json, outputs/metrics.csv, outputs/class_counts.csv,
and outputs/confusion_matrices/*.csv.

No performance numbers are hard-coded: all reported metrics come from the run.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

CLASSES = {
    "Potato___Early_blight": "Early blight",
    "Potato___Late_blight": "Late blight",
    "Potato___healthy": "Healthy",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def locate_images(data_dir: Path):
    found = []
    for folder in data_dir.rglob("*"):
        if folder.is_dir() and folder.name in CLASSES:
            label = CLASSES[folder.name]
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    found.append((p, label))
    if not found:
        raise FileNotFoundError(
            f"No potato class folders found under {data_dir}. "
            "Expected folders named Potato___Early_blight, Potato___Late_blight, "
            "and Potato___healthy. See README.md for dataset setup."
        )
    return found

def extract_features(path: Path, size=128):
    """Return three separate handcrafted feature vectors: colour, GLCM, LBP."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size))
        rgb = np.asarray(im, dtype=np.uint8)
    # Colour: HSV channel moments + compact histograms
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8).astype(np.float32)
    colour = []
    for c in range(3):
        channel = hsv[:, :, c]
        colour.extend([float(channel.mean()), float(channel.std())])
        hist, _ = np.histogram(channel, bins=16, range=(0, 256), density=True)
        colour.extend(hist.astype(float).tolist())

    # Texture: GLCM on 8-bit grayscale, quantized to 32 levels
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    quant = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    glcm_features = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        glcm_features.extend(graycoprops(glcm, prop).ravel().astype(float).tolist())

    # Texture: uniform LBP histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = 10  # P + 2 for uniform LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    lbp_features = hist.astype(float).tolist()

    return np.asarray(colour), np.asarray(glcm_features), np.asarray(lbp_features)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/PlantVillage", help="Root containing PlantVillage class folders")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=0, help="Optional cap for quick trial; 0 means all")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    out = root / "outputs"
    (out / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    rows = locate_images(data_dir)
    # Remove exact duplicate files by content hash, keeping the first occurrence.
    seen = set()
    unique = []
    for p, label in rows:
        digest = sha1_file(p)
        if digest not in seen:
            seen.add(digest)
            unique.append((p, label))
    rows = unique

    if args.max_per_class:
        limited = []
        counts = {}
        for item in rows:
            label = item[1]
            if counts.get(label, 0) < args.max_per_class:
                limited.append(item)
                counts[label] = counts.get(label, 0) + 1
        rows = limited

    labels = np.array([label for _, label in rows])
    paths = [p for p, _ in rows]
    counts = pd.Series(labels).value_counts().sort_index()
    if counts.min() < 5:
        raise ValueError("Each class needs at least 5 images for stratified splitting and CV.")

    print(f"Found {len(paths)} unique images")
    print(counts.to_string())

    colour, glcm, lbp = [], [], []
    good_paths, good_labels = [], []
    for i, (p, label) in enumerate(rows, 1):
        try:
            c, g, l = extract_features(p)
            colour.append(c); glcm.append(g); lbp.append(l)
            good_paths.append(str(p.relative_to(root)))
            good_labels.append(label)
        except Exception as e:
            print(f"Skipping unreadable image {p}: {e}")
        if i % 200 == 0:
            print(f"Processed {i}/{len(rows)}")

    y = np.asarray(good_labels)
    feature_sets = {
        "Colour": np.vstack(colour),
        "GLCM": np.vstack(glcm),
        "LBP": np.vstack(lbp),
        "Colour + GLCM + LBP": np.hstack([np.vstack(colour), np.vstack(glcm), np.vstack(lbp)])
    }
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    model_specs = {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
            {"model__C": [0.1, 1, 10]}
        ),
        "SVM (RBF)": (
            Pipeline([("scale", StandardScaler()), ("model", SVC(class_weight="balanced"))]),
            {"model__C": [0.5, 1, 5], "model__gamma": ["scale", "auto"]}
        ),
        "Random Forest": (
            Pipeline([("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                        random_state=args.seed, n_jobs=-1))]),
            {"model__max_features": ["sqrt", 0.7], "model__min_samples_leaf": [1, 2]}
        )
    }

    results = []
    confusion_payload = {}
    for fs_name, X in feature_sets.items():
        X_train, X_test = X[train_idx], X[test_idx]
        # Majority-class baseline on every feature set is identical; record once per feature set for context.
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        pred = dummy.predict(X_test)
        results.append({
            "feature_set": fs_name, "model": "Majority baseline",
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "best_cv_macro_f1": None
        })
        confusion_payload[f"{fs_name}__Majority baseline"] = confusion_matrix(y_test, pred, labels=sorted(np.unique(y))).tolist()

        for model_name, (pipe, grid) in model_specs.items():
            search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            pred = best.predict(X_test)
            results.append({
                "feature_set": fs_name, "model": model_name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
                "best_cv_macro_f1": search.best_score_,
                "best_params": search.best_params_
            })
            confusion_payload[f"{fs_name}__{model_name}"] = confusion_matrix(y_test, pred, labels=sorted(np.unique(y))).tolist()
            print(f"{fs_name:22s} | {model_name:20s} | test macro-F1={results[-1]['macro_f1']:.4f}")

    metrics = pd.DataFrame(results)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({"class": counts.index, "images": counts.values}).to_csv(out / "class_counts.csv", index=False)
    for key, matrix in confusion_payload.items():
        safe = key.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        pd.DataFrame(matrix, index=sorted(np.unique(y)), columns=sorted(np.unique(y))).to_csv(
            out / "confusion_matrices" / f"{safe}.csv", index=True
        )

    payload = {
        "project_title": "Plant Disease Identification Using Handcrafted Image Features",
        "dataset": "PlantVillage — potato subset",
        "classes": sorted(np.unique(y).tolist()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "n_images": int(len(y)),
        "test_size": args.test_size,
        "random_seed": args.seed,
        "split": {"train": int(len(train_idx)), "test": int(len(test_idx))},
        "metrics": metrics.replace({np.nan: None}).to_dict(orient="records"),
        "confusion_matrices": confusion_payload,
        "notes": [
            "All metrics are measured by this run; they are not pre-filled example values.",
            "Exact duplicate image files were removed before splitting.",
            "Feature extraction is deterministic; scaling and hyperparameter tuning are fitted within training folds.",
            "PlantVillage images are collected under relatively controlled conditions; field generalisation is not established."
        ]
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved outputs/results.json and outputs/metrics.csv")
    print(metrics.sort_values("macro_f1", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
