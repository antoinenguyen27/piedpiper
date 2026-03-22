import modal
import os

app = modal.App("meetingbank-distillation")

# Volumes
data_volume = modal.Volume.from_name("meetingbank-data-vol", create_if_missing=True)
weights_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

# Container environment
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "tqdm"
    )
)

DATA_DIR = "/workspace/data"
WEIGHTS_DIR = "/workspace/weights"
TOKENIZED_PATH = f"{DATA_DIR}/tokenized_meetingbank"


# ==========================================
# TRAINING LOOP
# ==========================================
@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={
        DATA_DIR: data_volume,
        WEIGHTS_DIR: weights_volume
    }
)
def train(run_num: int = 1, resume_from_run: int = None, num_epochs: int = 3):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from torch.optim import AdamW
    from datasets import load_from_disk
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Directory Setup
    run_dir = os.path.join(WEIGHTS_DIR, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)

    # Determine where to load weights from
    load_dir = os.path.join(WEIGHTS_DIR, f"run_{resume_from_run}") if resume_from_run else run_dir

    print(f"Running on device: {device} | Output Directory: {run_dir}")
    if resume_from_run:
        print(f"Forking from previous run: run_{resume_from_run}")

    # 2. Load Pre-Tokenized Dataset
    print(f"Loading tokenized dataset from {TOKENIZED_PATH}...")
    try:
        tokenized_datasets = load_from_disk(TOKENIZED_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {TOKENIZED_PATH}.")
        print("Make sure the setup step was run previously to populate the data volume.")
        return

    dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True)

    # 3. Load Architectures
    teacher_id = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    student_id = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)

    print("Loading models with Token Classification heads (num_labels=2)...")
    teacher = AutoModelForTokenClassification.from_pretrained(teacher_id, num_labels=2).to(device)
    teacher.eval()  # Freeze teacher

    # 4. Checkpoint Loading & Epoch Parsing
    latest_checkpoint = None
    start_epoch = 0

    if os.path.exists(load_dir):
        checkpoints = [d for d in os.listdir(load_dir) if d.startswith("checkpoint-epoch-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = os.path.join(load_dir, checkpoints[-1])
            # Extract the epoch number from the folder name to continue the count accurately
            start_epoch = int(latest_checkpoint.split("-")[-1])

    if latest_checkpoint:
        print(f"Loading previous weights! Resuming from: {latest_checkpoint}")
        student = AutoModelForTokenClassification.from_pretrained(latest_checkpoint).to(device)
        print(f"Resuming run at Epoch {start_epoch + 1}...")
    else:
        print(f"No checkpoints found in {load_dir}. Starting fresh from: {student_id}")
        student = AutoModelForTokenClassification.from_pretrained(student_id, num_labels=2).to(device)

    student.train()

    # 5. Setup Optimization & Loss
    optimizer = AdamW(student.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()

    # 6. Distillation Loop
    print("Starting distillation loop...")

    # The loop runs from the start_epoch for the requested number of num_epochs
    for epoch in range(start_epoch, start_epoch + num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_outputs = student(input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Calculate MSE Loss only on active tokens (ignore padding)
            active_loss = attention_mask.view(-1) == 1
            active_teacher_logits = teacher_logits.view(-1, 2)[active_loss]
            active_student_logits = student_logits.view(-1, 2)[active_loss]

            loss = loss_fn(active_student_logits, active_teacher_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # 7. End of Epoch Checkpointing
        checkpoint_dir = f"{run_dir}/checkpoint-epoch-{epoch + 1}"
        print(f"\nSaving Epoch {epoch + 1} weights to {checkpoint_dir}...")
        student.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        weights_volume.commit()

    # 8. Final Output Weights
    final_output_dir = f"{run_dir}/final_model"
    print(f"Saving final student weights to {final_output_dir}...")
    student.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    weights_volume.commit()

    print("Training complete! Weights persisted to volume.")