import modal
import os

app = modal.App("meetingbank-distillation")

# Two separate volumes: one for your dataset, one for your final model weights
data_volume = modal.Volume.from_name("meetingbank-data-vol", create_if_missing=True)
weights_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

# Define the container environment
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
# STEP 1: SETUP (Run this exactly once)
# ==========================================
@app.function(
    image=image,
    timeout=3600,  # 1 hour timeout for downloading and processing
    volumes={DATA_DIR: data_volume}
)
def setup():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Downloading MeetingBank dataset...")
    dataset = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")

    # Use the specific LLMLingua-2 multilingual tokenizer
    teacher_id = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    print(f"Loading tokenizer from {teacher_id}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    print("Tokenizing dataset (this might take a few minutes)...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    print(f"Saving tokenized dataset to {TOKENIZED_PATH}...")
    tokenized_datasets.save_to_disk(TOKENIZED_PATH)

    # Commit the volume so the tokenized data is permanently saved
    data_volume.commit()
    print("Setup complete! Data is cached on the Modal volume.")


# ==========================================
# STEP 2: TRAIN (Run this as many times as needed)
# ==========================================
@app.function(
    image=image,
    gpu="A10G",  # Switched to A10G for cost-effective 24GB VRAM
    timeout=86400,  # 24 hours max
    volumes={
        DATA_DIR: data_volume,
        WEIGHTS_DIR: weights_volume
    }
)
def train():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from torch.optim import AdamW
    from datasets import load_from_disk
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Load Pre-Tokenized Dataset instantly from the volume
    print(f"Loading tokenized dataset from {TOKENIZED_PATH}...")
    try:
        tokenized_datasets = load_from_disk(TOKENIZED_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {TOKENIZED_PATH}.")
        print("Did you forget to run `modal run fast_bert_train.py::setup` first?")
        return

    dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True)

    # 2. Load Architectures (Token Classification for Keep/Discard)
    teacher_id = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    student_id = "distilbert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(teacher_id)

    print("Loading models with Token Classification heads (num_labels=2)...")
    teacher = AutoModelForTokenClassification.from_pretrained(teacher_id, num_labels=2).to(device)
    student = AutoModelForTokenClassification.from_pretrained(student_id, num_labels=2).to(device)

    teacher.eval()  # Freeze teacher

    latest_checkpoint = None

    # Check the mounted Modal volume for existing checkpoints
    if os.path.exists(WEIGHTS_DIR):
        checkpoints = [d for d in os.listdir(WEIGHTS_DIR) if d.startswith("checkpoint-epoch-")]
        if checkpoints:
            # Sort by epoch number to grab the most recent one
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = os.path.join(WEIGHTS_DIR, checkpoints[-1])

    if latest_checkpoint:
        print(f"Loading previous weights! Resuming from: {latest_checkpoint}")
        # Load the student from your saved Modal volume directory
        student = AutoModelForTokenClassification.from_pretrained(latest_checkpoint).to(device)
    else:
        print(f"No checkpoints found. Starting fresh from: {student_id}")
        student = AutoModelForTokenClassification.from_pretrained(student_id, num_labels=2).to(device)

    student.train()

    # 3. Setup Optimization & Loss
    optimizer = AdamW(student.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()
    epochs = 3

    # 4. Distillation Loop
    print("Starting distillation loop...")
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
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

        # 5. End of Epoch Checkpointing
        checkpoint_dir = f"{WEIGHTS_DIR}/checkpoint-epoch-{epoch + 1}"
        print(f"\nSaving Epoch {epoch + 1} weights to {checkpoint_dir}...")
        student.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        weights_volume.commit()

    # 6. Final Output Weights
    final_output_dir = f"{WEIGHTS_DIR}/post_trained_distilbert_final"
    print(f"Saving final student weights to {final_output_dir}...")
    student.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    weights_volume.commit()

    print("Training complete! Weights persisted to volume.")