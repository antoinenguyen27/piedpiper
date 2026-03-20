import os
import torch
import torch.nn as nn

# --- Environment Setup ---
# This ensures the checkpoint folder is always relative to this script's location,
# which is the most reliable way to handle paths on a remote RunPod instance.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_FOLDER = os.path.join(SCRIPT_DIR, "model_checkpoints")

# Specify the model number and the epoch number
def save_model(model, epoch_num, folder=DEFAULT_CHECKPOINT_FOLDER):
    """
    Abstracted saving logic to handle directory creation and naming.
    """
    # exist_ok=True is a cleaner way to handle the "check and make" logic
    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(folder, f'checkpoint_model_{model}_epoch_{epoch_num}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'>>> Model checkpoint saved at: {save_path}')


# Create config class to hold hyperparameters
class TrainConfig:
    def __init__(self, loss_fn, optimizer_cls, train_loader=None, test_loader=None,
                 learning_rate=1e-4, batch_size=16, num_epochs=10, save_freq=2,
                 checkpoint_folder=DEFAULT_CHECKPOINT_FOLDER):
        # Renamed to optimizer_cls to clarify it expects the class (e.g., torch.optim.Adam),
        # not an instantiated optimizer, since you instantiate it in the train loop.
        self.optimizer_cls = optimizer_cls
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # Note: DataLoader usually handles batching, but good to keep for logging
        self.num_epochs = num_epochs
        self.save_freq = save_freq  # How often to save model checkpoints (in epochs)
        self.checkpoint_folder = checkpoint_folder


def train_model(model, teacher_model, config):
    print(f"--- Training Initialization ---")
    print(f"Checkpoints will be saved to: {os.path.abspath(config.checkpoint_folder)}")
    print(f"-------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimiser = config.optimizer_cls(model.parameters(), lr=config.learning_rate)
    epoch_history = {'total_loss': [], 'semantic_loss': [], 'compression_loss': []}

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        total_semantic = 0.0
        total_compression = 0.0

        num_batches = 0
        for batch_idx, video in enumerate(config.train_loader):
            video = video.to(device, non_blocking=True)

            masks = model(video)
            if masks.dim() != 2:
                raise ValueError(f"Expected masks with shape (B, T), got {tuple(masks.shape)}")

            mask_5d = masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            masked_video = video * mask_5d

            with torch.no_grad():
                orig_embeds, _ = teacher_model(video)
                masked_embeds, _ = teacher_model(masked_video)

            loss, semantic_loss, compression_loss = config.loss_fn(
                orig_embeds, masked_embeds, masks
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            total_semantic += semantic_loss.item()
            total_compression += compression_loss.item()
            num_batches += 1

            if batch_idx % 5 == 0:
                embed_delta = (orig_embeds - masked_embeds).abs().mean().item()
                print(
                    f"Epoch [{epoch + 1}/{config.num_epochs}] Batch [{batch_idx}] "
                    f"Loss: {loss.item():.4f} | Sem: {semantic_loss.item():.4f} | "
                    f"Comp: {compression_loss.item():.4f} | MaskMean: {masks.mean().item():.4f} | "
                    f"EmbedDelta: {embed_delta:.6f}",
                    flush=True,
                )

        if num_batches == 0:
            print(f"Epoch [{epoch + 1}/{config.num_epochs}] | No batches yielded.", flush=True)
            continue

        avg_loss = total_loss / num_batches
        avg_sem = total_semantic / num_batches
        avg_comp = total_compression / num_batches

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] | Total: {avg_loss:.4f} | Sem: {avg_sem:.4f} | Comp: {avg_comp:.4f}",
            flush=True,
        )

        epoch_history['total_loss'].append(avg_loss)
        epoch_history['semantic_loss'].append(avg_sem)
        epoch_history['compression_loss'].append(avg_comp)

        if (epoch + 1) % config.save_freq == 0:
            save_model(model, epoch + 1, folder=config.checkpoint_folder)

    return epoch_history