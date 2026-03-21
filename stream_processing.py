import os
import json
import time
import cv2
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2


class StreamingVideoDataset(IterableDataset):
    def __init__(
        self,
        video_dir,
        clip_length=16,
        sample_every_n=4,
        resolution=240,
        telemetry=False,
        telemetry_interval=50,
    ):
        """
        Args:
            video_dir: Path to the directory containing video files.
            clip_length: Number of frames per training clip (sequence length).
            sample_every_n: Subsamples video (e.g., 4 means keep 1 out of every 4 frames).
            resolution: The target shorter-edge resolution (e.g., 240p).
        """
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if
                            f.endswith(('.mp4', '.avi', '.mkv'))]
        self.clip_length = clip_length
        self.sample_every_n = sample_every_n
        self.telemetry = telemetry
        self.telemetry_interval = telemetry_interval

        # PyTorch v2 transforms can efficiently process (T, C, H, W) tensors.
        self.transform = v2.Compose([
            v2.Resize(size=resolution, antialias=True),
            # Standard ImageNet normalization (stabilizes training dynamics)
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __iter__(self):
        # In a multi-worker setup, we split the files across workers to avoid duplication
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_to_process = self.video_files
        else:
            # Partition the video files among workers
            files_to_process = [
                f for i, f in enumerate(self.video_files)
                if i % worker_info.num_workers == worker_info.id
            ]
        worker_id = 0 if worker_info is None else worker_info.id
        stats = {
            "video_opens": 0,
            "clips_yielded": 0,
            "frames_read": 0,
            "frames_kept": 0,
            "open_s": 0.0,
            "decode_read_s": 0.0,
            "decode_cvt_s": 0.0,
            "tensorize_s": 0.0,
            "transform_s": 0.0,
        }

        for video_path in files_to_process:
            open_start = time.perf_counter()
            cap = cv2.VideoCapture(video_path)
            stats["open_s"] += time.perf_counter() - open_start
            stats["video_opens"] += 1
            frames_buffer = []
            frame_idx = 0

            while cap.isOpened():
                read_start = time.perf_counter()
                ret, frame = cap.read()
                stats["decode_read_s"] += time.perf_counter() - read_start
                if not ret:
                    break  # End of video file
                stats["frames_read"] += 1

                # Subsampling: Keep every n-th frame
                if frame_idx % self.sample_every_n == 0:
                    # Convert OpenCV BGR to standard RGB
                    cvt_start = time.perf_counter()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stats["decode_cvt_s"] += time.perf_counter() - cvt_start

                    # Convert to tensor, scale to [0, 1], rearrange to (C, H, W)
                    tensor_start = time.perf_counter()
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    stats["tensorize_s"] += time.perf_counter() - tensor_start
                    frames_buffer.append(frame_tensor)
                    stats["frames_kept"] += 1

                    # Once we have enough frames for a sequence/clip, yield it
                    if len(frames_buffer) == self.clip_length:
                        clip_tensor = torch.stack(frames_buffer)
                        transform_start = time.perf_counter()
                        clip_tensor = self.transform(clip_tensor)
                        stats["transform_s"] += time.perf_counter() - transform_start

                        # Teacher embeddings are computed on-GPU in the training loop.
                        stats["clips_yielded"] += 1
                        if self.telemetry and stats["clips_yielded"] % self.telemetry_interval == 0:
                            total_decode_s = (
                                stats["open_s"]
                                + stats["decode_read_s"]
                                + stats["decode_cvt_s"]
                                + stats["tensorize_s"]
                                + stats["transform_s"]
                            )
                            payload = {
                                "event": "dataset_worker",
                                "worker_id": worker_id,
                                "video_path": video_path,
                                "clips_yielded": stats["clips_yielded"],
                                "frames_read": stats["frames_read"],
                                "frames_kept": stats["frames_kept"],
                                "video_opens": stats["video_opens"],
                                "open_s": round(stats["open_s"], 6),
                                "decode_read_s": round(stats["decode_read_s"], 6),
                                "decode_cvt_s": round(stats["decode_cvt_s"], 6),
                                "tensorize_s": round(stats["tensorize_s"], 6),
                                "transform_s": round(stats["transform_s"], 6),
                                "decode_total_s": round(total_decode_s, 6),
                                "raw_read_fps": round(stats["frames_read"] / total_decode_s, 4)
                                if total_decode_s > 0 else None,
                                "kept_frame_fps": round(stats["frames_kept"] / total_decode_s, 4)
                                if total_decode_s > 0 else None,
                            }
                            print(json.dumps(payload), flush=True)
                        yield clip_tensor

                        # Clear buffer for the next clip
                        frames_buffer = []

                frame_idx += 1

            cap.release()


# --- Usage Example ---
if __name__ == "__main__":
    import os as _os
    _data_dir = _os.environ.get("PIEDPIPER_DATA_DIR", "/data/datasets/eastgate/")
    dataset = StreamingVideoDataset(
        video_dir=_data_dir,
        clip_length=16,
        sample_every_n=4,
        resolution=240
    )

    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    for batch_idx, video_batch in enumerate(dataloader):
        print(f"Yielded batch {batch_idx} with shape: {video_batch.shape}")
        break
