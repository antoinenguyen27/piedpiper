import os
import cv2
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2


class StreamingVideoDataset(IterableDataset):
    def __init__(self, video_dir, clip_length=16, sample_every_n=4, resolution=240):
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

        for video_path in files_to_process:
            cap = cv2.VideoCapture(video_path)
            frames_buffer = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video file

                # Subsampling: Keep every n-th frame
                if frame_idx % self.sample_every_n == 0:
                    # Convert OpenCV BGR to standard RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to tensor, scale to [0, 1], rearrange to (C, H, W)
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames_buffer.append(frame_tensor)

                    # Once we have enough frames for a sequence/clip, yield it
                    if len(frames_buffer) == self.clip_length:
                        # Stack buffer into shape (T, C, H, W)
                        clip_tensor = torch.stack(frames_buffer)

                        # Apply resize and normalization transforms
                        clip_tensor = self.transform(clip_tensor)

                        # Yield the processed tensor (process and train simultaneously)
                        yield clip_tensor

                        # Clear buffer for the next clip
                        frames_buffer = []

                frame_idx += 1

            cap.release()


# --- Usage Example ---
if __name__ == "__main__":
    # Initialize the dataset
    dataset = StreamingVideoDataset(
        video_dir="data/eastgate/",
        clip_length=16,  # Model looks at 16 frames at a time
        sample_every_n=4,  # "Drops" intermediate frames by taking a stride of 4
        resolution=240  # Downgrades to 240p
    )

    # DataLoader streams the data. Using multiple workers speeds up I/O and preprocessing.
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    # Training loop concept
    for batch_idx, video_batch in enumerate(dataloader):
        # video_batch shape: (Batch_Size, Temporal_Clip_Length, Channels, Height, Width)
        # e.g., (4, 16, 3, 240, 426)
        print(f"Yielded batch {batch_idx} with shape: {video_batch.shape}")

        # pass to model...
        break