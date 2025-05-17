import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import upload_folder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import VariableVBatchSampler, collate_signseq, load_default_dataset
from .model import SignSeq

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16  # Adjust based on your GPU memory
NUM_EPOCHS = 50  # Adjust as needed

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = f"runs/signseq_training_{TIMESTAMP}"
MODEL_SAVE_DIR = "saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"signseq_model_{TIMESTAMP}.pt")

# Ensure save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def train_signseq():
    print(f"Using device: {DEVICE}")
    writer = SummaryWriter(LOG_DIR)
    print(f"TensorBoard logs will be saved to: {LOG_DIR}")

    # 1. Load Dataset
    print("Loading dataset...")
    ds = load_default_dataset()
    ds = ds.with_format("torch")

    train_sampler = VariableVBatchSampler(
        ds["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_sampler = VariableVBatchSampler(
        ds["test"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        ds["train"],
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_signseq,
    )
    test_loader = DataLoader(
        ds["test"],
        batch_sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_signseq,
    )
    print(
        f"Train dataset size: {len(ds['train'])}, Test dataset size: {len(ds['test'])}"
    )

    # 2. Initialize Model, Optimizer, Loss
    print("Initializing model...")
    model = SignSeq().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    print("Model, Optimizer, and Loss function initialized.")
    print(
        f"Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # 3. Training Loop
    print("Starting training...")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False
        )
        for batch_idx, batch in enumerate(progress_bar):
            refs = batch["refs"].to(DEVICE)  # (B, V, T, F)
            ref_controls = batch["ref_controls"].to(DEVICE)  # (B, V, 2)
            control = batch["control"].to(DEVICE)  # (B, 2)
            target = batch["target"].to(DEVICE)  # (B, T, F)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(refs, ref_controls, control)  # (B, T, F)

            loss = criterion(predictions, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            global_step += 1

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_epoch_train_loss, epoch)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_epoch_train_loss:.4f}")

        # 4. Evaluation Step
        model.eval()
        epoch_eval_loss = 0.0
        with torch.no_grad():
            progress_bar_eval = tqdm(
                test_loader,
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Evaluating]",
                leave=False,
            )
            for batch in progress_bar_eval:
                refs = batch["refs"].to(DEVICE)
                ref_controls = batch["ref_controls"].to(DEVICE)
                control = batch["control"].to(DEVICE)
                target = batch["target"].to(DEVICE)

                predictions = model(refs, ref_controls, control)
                loss = criterion(predictions, target)
                epoch_eval_loss += loss.item()
                progress_bar_eval.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_epoch_eval_loss = epoch_eval_loss / len(test_loader)
        writer.add_scalar("Loss/eval_epoch", avg_epoch_eval_loss, epoch)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Eval Loss: {avg_epoch_eval_loss:.4f}")

        # 5. Save Model
        if (
            epoch + 1
        ) % 10 == 0 or epoch == NUM_EPOCHS - 1:  # Save every 10 epochs and at the end
            current_model_save_path = os.path.join(
                MODEL_SAVE_DIR, f"signseq_model_epoch_{epoch+1}_{TIMESTAMP}.pt"
            )
            torch.save(model.state_dict(), current_model_save_path)
            print(f"Model saved to {current_model_save_path}")

    writer.close()
    print("Training finished.")
    print(
        f"Final model saved to {MODEL_SAVE_PATH} (Note: this is a template path, actual saves are epoch-specific)"
    )
    # Save final model
    final_model_path = os.path.join(
        MODEL_SAVE_DIR, f"signseq_model_final_{TIMESTAMP}.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model state dict saved to {final_model_path}")

    REPO_ID = "JacobLinCool/SignSeq-exp-01"
    model.push_to_hub(REPO_ID)
    upload_folder(
        repo_id=REPO_ID,
        folder_path=LOG_DIR,
        path_in_repo=LOG_DIR,
        commit_message="Upload training logs",
    )


if __name__ == "__main__":
    train_signseq()
