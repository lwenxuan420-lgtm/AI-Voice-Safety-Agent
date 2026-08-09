import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader

from model import CNN2D

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ASVSPOOF + PHONE FINETUNE TRAINING")
print("=" * 60)
print("DEVICE:", DEVICE)

# =========================================================
# CONFIG
# =========================================================

TARGET_SR = 16000
TARGET_LEN = 64000

BATCH_SIZE = 16

EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 8

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4

# =========================================================
# SWITCH
# =========================================================

RUN_STAGE1 = True
RUN_STAGE2 = True

# =========================================================
# PATH
# =========================================================

# -----------------------
# ASVSPOOF
# -----------------------

ASV_ROOTS = [
    r"D:\data\data\train-renamepart_006\train-rename",
    r"D:\data\data\train-renamepart_009\train-rename"
]

ASV_CSV = r"D:\csv\train.csv"

# -----------------------
# PHONE
# -----------------------

PHONE_ROOTS = [
    r"D:\ASVspoof2021\2021-main\2021-main\LA\Baseline-RawNet2\phone_dataset"
]

PHONE_CSV = r"D:\csv\train-clean.csv"

# =========================================================
# EER
# =========================================================

def compute_eer(y_true, y_score):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )

    fnr = 1 - tpr

    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    return eer

# =========================================================
# AUDIO PROCESSOR
# =========================================================

class AudioProcessor:

    def __init__(self):

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

    def __call__(self, wav, sr):

        wav = torch.tensor(wav).float()

        # stereo -> mono
        if wav.dim() == 2:
            wav = wav.mean(dim=1)

        # resample
        if sr != TARGET_SR:

            wav = torchaudio.functional.resample(
                wav,
                sr,
                TARGET_SR
            )

        # waveform normalize
        wav = (
            wav - wav.mean()
        ) / (
            wav.std() + 1e-8
        )

        # pad / truncate
        if wav.shape[0] < TARGET_LEN:

            wav = torch.nn.functional.pad(
                wav,
                (
                    0,
                    TARGET_LEN - wav.shape[0]
                )
            )

        else:

            wav = wav[:TARGET_LEN]

        # mel
        mel = self.mel(wav)

        mel = torch.log(mel + 1e-6)

        # feature normalize
        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-8
        )

        return mel

# =========================================================
# DATASET
# =========================================================

class AudioDataset(Dataset):

    def __init__(self, df, roots):

        self.samples = []

        for row in df.itertuples():

            for root in roots:

                path = os.path.join(
                    root,
                    row.audio_name
                )

                if os.path.exists(path):

                    self.samples.append(
                        (
                            path,
                            float(row.target)
                        )
                    )

                    break

        print("Samples:", len(self.samples))

        self.processor = AudioProcessor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        wav, sr = sf.read(path)

        feat = self.processor(wav, sr)

        return (
            feat.unsqueeze(0),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

# =========================================================
# EVALUATE
# =========================================================

def evaluate(model, loader):

    model.eval()

    scores = []
    labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            out = model(x)

            # AUTO COMPATIBLE
            if out.dim() > 1:
                out = out.squeeze(1)

            prob_real = torch.sigmoid(out)

            scores.extend(
                prob_real.cpu().numpy()
            )

            labels.extend(
                y.numpy()
            )

    eer = compute_eer(
        np.array(labels),
        np.array(scores)
    )

    return eer

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_stage(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    save_name
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr
    )

    best_eer = 999

    for epoch in range(epochs):

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        total_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for x, y in loop:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            out = model(x)

            # AUTO COMPATIBLE
            if out.dim() > 1:
                out = out.squeeze(1)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=loss.item()
            )

        avg_loss = total_loss / len(train_loader)

        # =================================================
        # VALIDATION
        # =================================================

        eer = evaluate(
            model,
            val_loader
        )

        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"EER : {eer:.6f}")
        print("=" * 50)

        # =================================================
        # SAVE BEST
        # =================================================

        if eer < best_eer:

            best_eer = eer

            torch.save(
                model.state_dict(),
                save_name
            )

            print(f"🔥 SAVED BEST: {save_name}")

# =========================================================
# STAGE 1
# =========================================================

if RUN_STAGE1:

    print("\n" + "=" * 60)
    print("STAGE 1 - ASVSPOOF PRETRAIN")
    print("=" * 60)

    asv_df = pd.read_csv(ASV_CSV)

    train_df, val_df = train_test_split(
        asv_df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    train_set = AudioDataset(
        train_df,
        ASV_ROOTS
    )

    val_set = AudioDataset(
        val_df,
        ASV_ROOTS
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = CNN2D().to(DEVICE)

    train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS_STAGE1,
        lr=LR_STAGE1,
        save_name="best_model.pth"
    )

