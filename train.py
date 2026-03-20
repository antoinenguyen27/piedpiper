from torch import optim
from torch.utils.data import DataLoader

from cnn_compress import VideoCompressor
from stream_processing import StreamingVideoDataset
from loss_func import SemanticCompressionLoss
from train_helpers import train_model, save_model, TrainConfig

compressor = VideoCompressor()
train_dataset = StreamingVideoDataset(
    video_dir="data/eastgate/",
    clip_length=16,  # Model looks at 16 frames at a time
    sample_every_n=4,  # "Drops" intermediate frames by taking a stride of 4
    resolution=240  # Downgrades to 240p
)



# Create a config object to hold our training settings
optimizer_cls = optim.Adam
loss_fn = SemanticCompressionLoss()
# DataLoader streams the data. Using multiple workers speeds up I/O and preprocessing.
train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=2)
test_loader = None # You can create a separate test dataset and dataloader if you have one
learning_rate = 0.001
batch_size = 4
num_epochs = 10
save_freq = 2  # Save a checkpoint every 2 epochs
checkpoint_folder = "./model_checkpoints"

train_config = TrainConfig(
    loss_fn=loss_fn,
    optimizer_cls=optimizer_cls,
    train_loader=train_dataloader,
    test_loader=test_loader,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs,
    save_freq=save_freq,
    checkpoint_folder=checkpoint_folder
)

# Start training
epoch_history = train_model(compressor, train_config)
