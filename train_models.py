"""
Comprehensive training pipeline for:
- BERT fine-tuning on phishing/legitimate emails
- URL classifier training on phishing/legitimate URLs
- MobileNetV2 fine-tuning on website screenshots
"""

import os
import glob
import json
import time
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm


DATASET_DIR = Path("datasets")
DEFAULT_EMAIL_DATASET = "training_data/new_training_data.csv"
LEGACY_EMAIL_DATASETS = [
    "phishing_email_datasets/enron_phishing_and_legitimate_email_dataset.csv",
    "phishing_email_datasets/ling_phishing_and_legitimate_email_dataset.csv",
    "phishing_email_datasets/spam_assassin_phishing_and_legitimate_email_dataset.csv",
]
URL_DATASETS = [
    "phishing_url_datasets/kaggle_phishing_and_legitimate_url_dataset.csv",
    "phishing_url_datasets/mendeley_phishing_url_dataset.csv",
    "phishing_url_datasets/mendeley_legitimate_url_dataset.csv",
]
URL_FEATURE_DATASET = "training_data/new_url_training_data.csv"
URL_COMPUTABLE_FEATURES = [
    "Querylength",
    "domainlength",
    "pathLength",
    "urlLen",
    "tld",
    "isPortEighty",
    "NumberofDotsinURL",
    "ISIpAddressInDomainName",
    "URL_DigitCount",
    "host_DigitCount",
    "Query_DigitCount",
    "URL_Letter_Count",
    "host_letter_count",
    "Query_LetterCount",
    "NumberRate_URL",
    "NumberRate_Domain",
    "SymbolCount_URL",
    "SymbolCount_Domain",
    "Entropy_URL",
    "Entropy_Domain",
    "URL_sensitiveWord",
    "URLQueries_variable",
    "executable",
]
IMAGE_DIR = "phishing_website_image_datasets/circl_phishing_website_imageset"
VISION_PHISHING_SUBDIRS = ["phishing", "malicious", "fake"]
VISION_LEGIT_SUBDIRS = ["legitimate", "benign", "safe"]

MODELS_DIR = Path("trained_models")
MODELS_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = MODELS_DIR / "training_progress.json"

# Relative weights used to compute an overall progress percentage.
PHASE_WEIGHTS = {
    "bert": 70.0,
    "url": 20.0,
    "vision": 10.0,
}


def ensure_transformers_datasets_compat():
    """Ensure transformers can safely reference `datasets.Dataset`.

    In this repo, a local `datasets/` directory may shadow the Hugging Face
    package name. Trainer only needs this attribute for an `isinstance` check.
    """
    ds_module = sys.modules.get("datasets")
    if ds_module is not None and not hasattr(ds_module, "Dataset"):
        ds_module.Dataset = type("Dataset", (), {})


def _new_progress_state():
    now = time.time()
    return {
        "status": "running",
        "overall_percent": 0.0,
        "started_at": now,
        "updated_at": now,
        "phases": {
            "bert": {"status": "pending", "percent": 0.0, "detail": "Waiting"},
            "url": {"status": "pending", "percent": 0.0, "detail": "Waiting"},
            "vision": {"status": "pending", "percent": 0.0, "detail": "Waiting"},
        },
    }


def _compute_overall(phases):
    total = 0.0
    for phase_name, weight in PHASE_WEIGHTS.items():
        total += (phases[phase_name]["percent"] / 100.0) * weight
    return round(total, 2)


def write_progress(update_fn=None):
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = _new_progress_state()

    if update_fn is not None:
        update_fn(state)

    state["overall_percent"] = _compute_overall(state["phases"])
    state["updated_at"] = time.time()

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def set_phase_progress(phase, status=None, percent=None, detail=None):
    def _update(state):
        phase_state = state["phases"][phase]
        if status is not None:
            phase_state["status"] = status
        if percent is not None:
            phase_state["percent"] = max(0.0, min(100.0, float(percent)))
        if detail is not None:
            phase_state["detail"] = detail

    write_progress(_update)


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, phase_name):
        self.phase_name = phase_name

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps and state.max_steps > 0:
            pct = (state.global_step / state.max_steps) * 100.0
            set_phase_progress(
                self.phase_name,
                status="running",
                percent=pct,
                detail=f"Step {state.global_step}/{state.max_steps}",
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        set_phase_progress(self.phase_name, status="completed", percent=100.0, detail="Completed")
        return control


def get_latest_checkpoint(checkpoint_root):
    """Return latest Hugging Face checkpoint dir (checkpoint-*) or None."""
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        return None

    candidates = []
    for p in checkpoint_root.glob("checkpoint-*"):
        if p.is_dir():
            try:
                step = int(p.name.split("-")[-1])
                candidates.append((step, p))
            except ValueError:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_email_datasets():
    """Load and combine email datasets.

    Default behavior:
    - Use datasets/training_data/new_training_data.csv when available.
    - Fallback to legacy datasets if the default file is missing.
    """
    print("Loading email datasets...")
    dfs = []

    default_path = DATASET_DIR / DEFAULT_EMAIL_DATASET
    email_sources = [DEFAULT_EMAIL_DATASET] if default_path.exists() else LEGACY_EMAIL_DATASETS

    if default_path.exists():
        print(f"  [OK] Using default email dataset: {DEFAULT_EMAIL_DATASET}")
    else:
        print("  [WARN] Default email dataset not found, falling back to legacy email datasets")

    for email_file in email_sources:
        path = DATASET_DIR / email_file
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
                df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]

                if "body" not in df.columns or "label" not in df.columns:
                    raise ValueError("CSV must contain at least 'body' and 'label' columns")

                if "subject" not in df.columns:
                    # If subject is missing, derive it from the first line of body.
                    df["subject"] = df["body"].fillna("").astype(str).apply(
                        lambda text: next((line.strip() for line in text.splitlines() if line.strip()), "")[:120]
                    )

                df = df[["subject", "body", "label"]]
                df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
                df = df.dropna(subset=["subject", "body"])
                df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
                dfs.append(df[["text", "label"]])
                print(f"  [OK] {email_file}: {len(df)} samples")
            except Exception as e:
                print(f"  [ERROR] {email_file}: {e}")
    
    if not dfs:
        raise ValueError("No email datasets loaded!")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total email samples: {len(combined_df)}")
    print(f"  Phishing (1): {(combined_df['label'] == 1).sum()}")
    print(f"  Legitimate (0): {(combined_df['label'] == 0).sum()}")
    return combined_df


def load_url_datasets():
    """Load and combine all URL datasets."""
    print("\nLoading URL datasets...")

    # Prefer new precomputed feature dataset when available.
    feature_path = DATASET_DIR / URL_FEATURE_DATASET
    if feature_path.exists():
        try:
            df = pd.read_csv(feature_path, low_memory=False, encoding="utf-8-sig")
            original_cols = list(df.columns)
            col_map = {str(c).strip().lower().lstrip("\ufeff"): c for c in original_cols}

            label_key = "url_type_obf_type"
            if label_key not in col_map:
                raise ValueError("Missing URL_Type_obf_Type label column")

            raw_labels = df[col_map[label_key]]
            if pd.api.types.is_numeric_dtype(raw_labels):
                labels = pd.to_numeric(raw_labels, errors="coerce").fillna(0).astype(int)
                labels = (labels > 0).astype(int)
            else:
                normalized = raw_labels.astype(str).str.strip().str.lower()
                benign_tokens = {"benign", "legitimate", "safe", "normal", "0", "false"}
                labels = (~normalized.isin(benign_tokens)).astype(int)

            selected = []
            for feature in URL_COMPUTABLE_FEATURES:
                key = feature.lower()
                if key in col_map:
                    selected.append(col_map[key])

            if len(selected) < 8:
                raise ValueError(
                    f"Insufficient usable feature overlap. Found {len(selected)} of {len(URL_COMPUTABLE_FEATURES)} required features."
                )

            feature_df = df[selected].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            feature_df.columns = selected
            feature_df["label"] = labels
            feature_df = feature_df.dropna(subset=["label"])

            print(f"  [OK] Using feature URL dataset: {URL_FEATURE_DATASET}")
            print(f"  [OK] Selected computable features: {len(selected)}")
            print(f"Total URL samples: {len(feature_df)}")
            print(f"  Phishing (1): {(feature_df['label'] == 1).sum()}")
            print(f"  Legitimate (0): {(feature_df['label'] == 0).sum()}")
            return feature_df
        except Exception as e:
            print(f"  [WARN] Could not use feature URL dataset ({URL_FEATURE_DATASET}): {e}")
            print("  [WARN] Falling back to legacy URL datasets")

    dfs = []
    for url_file in URL_DATASETS:
        path = DATASET_DIR / url_file
        if path.exists():
            try:
                df = pd.read_csv(path, dtype={"url": str, "type": int})
                df.columns = ["url", "label"]
                df["label"] = 1 - df["label"]
                dfs.append(df)
                print(f"  [OK] {url_file}: {len(df)} samples")
            except Exception as e:
                print(f"  [ERROR] {url_file}: {e}")
    
    if not dfs:
        raise ValueError("No URL datasets loaded!")
    
    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["url"])
    print(f"Total unique URLs: {len(combined_df)}")
    print(f"  Phishing (1): {(combined_df['label'] == 1).sum()}")
    print(f"  Legitimate (0): {(combined_df['label'] == 0).sum()}")
    
    sample_size = min(50000, len(combined_df))
    sampled_df = combined_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), sample_size//2), random_state=42)
    )
    print(f"Sampled to: {len(sampled_df)} URLs for training")
    return sampled_df


def load_image_dataset():
    """Load website screenshots for vision fine-tuning.

    Preferred layout:
    datasets/.../circl_phishing_website_imageset/
      phishing/*.png
      legitimate/*.png
    """
    print("\nLoading website images...")
    image_path = DATASET_DIR / IMAGE_DIR
    if not image_path.exists():
        print(f"  [ERROR] Image directory not found: {image_path}")
        return [], []

    image_paths = []
    labels = []

    # Preferred: class subfolders for phishing(1) and legitimate(0).
    for subdir in VISION_PHISHING_SUBDIRS:
        sub_path = image_path / subdir
        if sub_path.exists() and sub_path.is_dir():
            pngs = sorted(glob.glob(str(sub_path / "*.png")))
            image_paths.extend(pngs)
            labels.extend([1] * len(pngs))

    for subdir in VISION_LEGIT_SUBDIRS:
        sub_path = image_path / subdir
        if sub_path.exists() and sub_path.is_dir():
            pngs = sorted(glob.glob(str(sub_path / "*.png")))
            image_paths.extend(pngs)
            labels.extend([0] * len(pngs))

    if image_paths:
        print(f"  [OK] Labeled PNG images found: {len(image_paths)}")
        print(f"  Phishing (1): {sum(1 for y in labels if y == 1)}")
        print(f"  Legitimate (0): {sum(1 for y in labels if y == 0)}")
        return image_paths, labels

    # Backward compatibility: root-level PNGs exist but no labels.
    root_pngs = sorted(glob.glob(str(image_path / "*.png")))
    if root_pngs:
        print(f"  [WARN] Found {len(root_pngs)} root PNGs but no class subfolders.")
        print("  [WARN] Vision training requires both phishing and legitimate labeled folders.")
        return root_pngs, [1] * len(root_pngs)

    print("  [ERROR] No PNG images found")
    return [], []


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])[:512]
        label = int(self.labels[idx])
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_email_classifier(df_emails, epochs=1, batch_size=32, resume_from_checkpoint=True):
    """Fine-tune BERT for email phishing classification."""
    print("\n" + "="*70)
    print("FINE-TUNING BERT ON EMAIL DATASET")
    print("="*70)
    
    set_phase_progress("bert", status="running", percent=0.0, detail="Preparing dataset")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    text_values = df_emails["text"].astype(str).to_numpy()
    label_values = df_emails["label"].astype(int).to_numpy()

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        text_values,
        label_values,
        test_size=0.2,
        random_state=42,
        stratify=label_values,
    )
    
    train_dataset = EmailDataset(texts_train, labels_train, tokenizer)
    test_dataset = EmailDataset(texts_test, labels_test, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "bert_email_checkpoint"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[TrainingProgressCallback("bert")],
    )
    
    checkpoint_root = MODELS_DIR / "bert_email_checkpoint"
    latest_checkpoint = get_latest_checkpoint(checkpoint_root) if resume_from_checkpoint else None

    ensure_transformers_datasets_compat()

    print("Starting training...")
    if latest_checkpoint is not None:
        print(f"[OK] Resuming BERT from checkpoint: {latest_checkpoint}")
        set_phase_progress("bert", status="running", detail=f"Resuming from {latest_checkpoint.name}")
        trainer.train(resume_from_checkpoint=str(latest_checkpoint))
    else:
        if not resume_from_checkpoint:
            print("[OK] Starting fresh BERT training (checkpoint resume disabled)")
        trainer.train()
    
    model_save_path = MODELS_DIR / "bert_email_classifier"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"[OK] Email classifier saved to {model_save_path}")
    
    return model, tokenizer


class URLFeatureExtractor:
    """Extract statistical features from URLs for classification."""
    
    @staticmethod
    def extract_features(url):
        """Extract 15 features from URL."""
        features = []
        features.append(len(url))
        domain = url.split("//")[-1].split("/")[0]
        features.append(len(domain))
        features.append(url.count("."))
        features.append(url.count("/"))
        features.append(url.count("-"))
        features.append(1 if any(c.isdigit() for c in domain.split(".")[0]) else 0)
        features.append(1 if "@" in url else 0)
        features.append(url.count("//") - 1)
        domain_chars = len(set(domain))
        features.append(domain_chars / max(len(domain), 1))
        features.append(1 if "xn--" in domain else 0)
        features.append(sum(1 for c in domain if c.isdigit()))
        features.append(domain.count("."))
        tld = domain.split(".")[-1]
        features.append(len(tld))
        features.append(1 if "?" in url else 0)
        labels = domain.split(".")
        avg_label_len = sum(len(l) for l in labels) / len(labels) if labels else 0
        features.append(avg_label_len)
        return np.array(features, dtype=np.float32)


def train_url_classifier(df_urls):
    """Train URL classifier using extracted features."""
    print("\n" + "="*70)
    print("TRAINING URL CLASSIFIER")
    print("="*70)
    
    set_phase_progress("url", status="running", percent=0.0, detail="Extracting URL features")

    extractor = None
    if "url" in df_urls.columns:
        extractor = URLFeatureExtractor()
        total_urls = len(df_urls)
        feature_rows = []
        update_every = max(100, total_urls // 100)
        for idx, url in enumerate(tqdm(df_urls["url"], desc="Extracting features"), start=1):
            feature_rows.append(extractor.extract_features(url))
            if idx % update_every == 0 or idx == total_urls:
                pct = (idx / max(total_urls, 1)) * 70.0
                set_phase_progress("url", status="running", percent=pct, detail=f"Features {idx}/{total_urls}")

        features = np.array(feature_rows)
        feature_columns = None
        labels = df_urls["label"].values
    else:
        feature_columns = [c for c in df_urls.columns if c != "label"]
        features = df_urls[feature_columns].astype(np.float32).to_numpy()
        labels = df_urls["label"].astype(int).to_numpy()
        set_phase_progress("url", status="running", percent=70.0, detail=f"Using {len(feature_columns)} precomputed features")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    print("Training gradient boosting classifier...")
    set_phase_progress("url", status="running", percent=75.0, detail="Training gradient boosting model")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    
    import pickle
    model_path = MODELS_DIR / "url_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "clf": clf,
                "scaler": scaler,
                "extractor": extractor,
                "mode": "legacy_extractor" if extractor is not None else "precomputed",
                "feature_columns": feature_columns,
            },
            f,
        )
    print(f"[OK] URL classifier saved to {model_path}")
    set_phase_progress("url", status="completed", percent=100.0, detail="Completed")
    
    return clf, scaler, extractor


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        processed = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": processed["pixel_values"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_vision_classifier(image_paths, labels):
    """Fine-tune MobileNetV2 on website screenshots."""
    print("\n" + "="*70)
    print("FINE-TUNING MOBILENETV2 ON WEBSITE IMAGES")
    print("="*70)
    
    set_phase_progress("vision", status="running", percent=0.0, detail="Preparing image dataset")

    if not image_paths:
        print("  [ERROR] No images available for vision training")
        set_phase_progress("vision", status="failed", percent=0.0, detail="No images available")
        return None, None

    if len(set(labels)) < 2:
        print("  [ERROR] Vision dataset must include both phishing and legitimate classes")
        print("  [HINT] Add PNG files under phishing/ and legitimate/ subfolders")
        set_phase_progress("vision", status="failed", percent=0.0, detail="Need both phishing and legitimate image classes")
        return None, None
    
    valid_paths = []
    valid_labels = []
    for path, label in zip(image_paths, labels):
        try:
            img = Image.open(path)
            img.verify()
            valid_paths.append(path)
            valid_labels.append(label)
        except:
            pass
    
    print(f"  Valid images: {len(valid_paths)}")
    
    if len(valid_paths) < 2:
        print("  [ERROR] Not enough images for training")
        set_phase_progress("vision", status="failed", percent=0.0, detail="Not enough valid images")
        return None, None

    if len(set(valid_labels)) < 2:
        print("  [ERROR] After validation, only one class remains")
        set_phase_progress("vision", status="failed", percent=0.0, detail="Only one class left after image validation")
        return None, None
    
    model_name = "google/mobilenet_v2_1.0_224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "legitimate", 1: "phishing"}
    model.config.label2id = {"legitimate": 0, "phishing": 1}
    
    paths_train, paths_test, labels_train, labels_test = train_test_split(
        valid_paths, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
    )
    
    train_dataset = ImageDataset(paths_train, labels_train, processor)
    test_dataset = ImageDataset(paths_test, labels_test, processor)
    
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "mobilenet_vision_checkpoint"),
        num_train_epochs=5,
        max_steps=30,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=5,
        eval_steps=5,
        logging_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[TrainingProgressCallback("vision")],
    )

    ensure_transformers_datasets_compat()
    
    print("Starting vision model training...")
    trainer.train()

    # Summarize phishing-class metrics on held-out validation split.
    eval_output = trainer.predict(test_dataset)
    y_true = np.array(labels_test)
    y_pred = np.argmax(eval_output.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    print(f"  Vision phishing precision: {precision:.4f}")
    print(f"  Vision phishing recall:    {recall:.4f}")
    print(f"  Vision phishing F1:        {f1:.4f}")
    
    model_save_path = MODELS_DIR / "mobilenet_vision_classifier"
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    print(f"[OK] Vision classifier saved to {model_save_path}")
    
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Train phishing detection models.")
    parser.add_argument(
        "--force-bert-retrain",
        action="store_true",
        help="Retrain BERT even if an existing trained model is found.",
    )
    parser.add_argument(
        "--force-url-retrain",
        action="store_true",
        help="Retrain URL classifier even if an existing model is found.",
    )
    args = parser.parse_args()

    write_progress(lambda s: (s.update(_new_progress_state()), s.pop("error", None)))
    print("\n" + "="*70)
    print("PHISHING DETECTOR - COMPREHENSIVE MODEL TRAINING PIPELINE")
    print("="*70)

    bert_output_exists = (MODELS_DIR / "bert_email_classifier").exists() and not args.force_bert_retrain
    url_output_exists = (MODELS_DIR / "url_classifier.pkl").exists() and not args.force_url_retrain

    df_emails = load_email_datasets() if not bert_output_exists else None
    df_urls = load_url_datasets() if not url_output_exists else None
    image_paths, image_labels = load_image_dataset()
    
    try:
        if bert_output_exists:
            print("[OK] Skipping BERT training (existing model found)")
            set_phase_progress("bert", status="completed", percent=100.0, detail="Skipped (already trained)")
            bert_model, bert_tokenizer = None, None
        else:
            bert_model, bert_tokenizer = train_email_classifier(
                df_emails,
                epochs=1,
                batch_size=32,
                resume_from_checkpoint=not args.force_bert_retrain,
            )

        if url_output_exists:
            print("[OK] Skipping URL training (existing model found)")
            set_phase_progress("url", status="completed", percent=100.0, detail="Skipped (already trained)")
            url_clf, url_scaler, url_extractor = None, None, None
        else:
            url_clf, url_scaler, url_extractor = train_url_classifier(df_urls)

        vision_model, vision_processor = train_vision_classifier(image_paths, image_labels)

        write_progress(lambda s: (s.update({"status": "completed"}), s.pop("error", None)))
    except Exception as exc:
        write_progress(lambda s: s.update({"status": "failed", "error": str(exc)}))
        raise
    
    print("\n" + "="*70)
    print("[SUCCESS] ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print(f"  Location: {MODELS_DIR}")
    print("="*70)
    print("\nNext steps:")
    print("1. Models are saved in trained_models/")
    print("2. Run the app with: streamlit run main.py")


if __name__ == "__main__":
    main()
