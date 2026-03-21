import json
import os
import time
import torch
import torch.nn as nn

# --- Environment Setup ---
# This ensures the checkpoint folder is always relative to this script's location,
# which is the most reliable way to handle paths on any remote instance.
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
                 checkpoint_folder=DEFAULT_CHECKPOINT_FOLDER, telemetry=False,
                 telemetry_interval=10):
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
        self.telemetry = telemetry
        self.telemetry_interval = telemetry_interval


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

    try:
        total_batches_str = str(len(config.train_loader))
    except TypeError:
        total_batches_str = "?"

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        total_semantic = 0.0
        total_compression = 0.0

        num_batches = 0
        loader_iter = iter(config.train_loader)
        batch_idx = 0
        while True:
            step_start = time.perf_counter()
            wait_start = step_start
            try:
                video = next(loader_iter)
            except StopIteration:
                break

            next_batch_wait_s = time.perf_counter() - wait_start

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            h2d_start = time.perf_counter()
            video = video.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_copy_s = time.perf_counter() - h2d_start

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_start = time.perf_counter()
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gpu_compute_s = time.perf_counter() - compute_start
            step_total_s = time.perf_counter() - step_start

            total_loss += loss.item()
            total_semantic += semantic_loss.item()
            total_compression += compression_loss.item()
            num_batches += 1

            if config.telemetry and batch_idx % config.telemetry_interval == 0:
                current_batch_size = int(video.shape[0])
                clip_length = int(video.shape[1]) if video.dim() > 1 else 0
                telemetry = {
                    "event": "train_batch",
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "batch_size": current_batch_size,
                    "clip_length": clip_length,
                    "next_batch_wait_s": round(next_batch_wait_s, 6),
                    "h2d_copy_s": round(h2d_copy_s, 6),
                    "gpu_compute_s": round(gpu_compute_s, 6),
                    "step_total_s": round(step_total_s, 6),
                    "clips_per_s": round(current_batch_size / step_total_s, 4) if step_total_s > 0 else None,
                    "frames_per_s": round((current_batch_size * clip_length) / step_total_s, 4)
                    if step_total_s > 0 and clip_length > 0 else None,
                    "loss": round(loss.item(), 8),
                    "semantic_loss": round(semantic_loss.item(), 8),
                    "compression_loss": round(compression_loss.item(), 8),
                    "mask_mean": round(masks.mean().item(), 8),
                    "embed_delta": round((orig_embeds - masked_embeds).abs().mean().item(), 8),
                }
                if device.type == "cuda":
                    telemetry.update({
                        "gpu_mem_alloc_gb": round(torch.cuda.memory_allocated(device) / (1024 ** 3), 4),
                        "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved(device) / (1024 ** 3), 4),
                        "gpu_mem_max_alloc_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4),
                    })
                print(json.dumps(telemetry), flush=True)
            elif batch_idx % 5 == 0:
                embed_delta = (orig_embeds - masked_embeds).abs().mean().item()
                
                batches_remaining = "?" if total_batches_str == "?" else str(int(total_batches_str) - (batch_idx + 1))
                
                print(
                    f"Epoch [{epoch + 1}/{config.num_epochs}] Batch [{batch_idx}/{total_batches_str}] (Rem: {batches_remaining}) "
                    f"Loss: {loss.item():.8f} | Sem: {semantic_loss.item():.8f} | "
                    f"Comp: {compression_loss.item():.8f} | MaskMean: {masks.mean().item():.8f} | "
                    f"EmbedDelta: {embed_delta:.8f}",
                    flush=True,
                )
            batch_idx += 1

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
