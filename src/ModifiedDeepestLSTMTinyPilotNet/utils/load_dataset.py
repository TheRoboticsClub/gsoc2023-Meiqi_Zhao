import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import h5py
import torch.nn.functional as F

class CARLADataset(Dataset):
    def __init__(self, directory, transform=None, hlc_one_hot=True):
        """
        Args:
            directory (string): Directory with all the hdf5 files containing the episode data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_paths = glob.glob(os.path.join(directory, "*.hdf5"))
        self.file_paths.sort()  # Ensuring data is processed in order
        self.transform = transform

        self.lengths = []
        self.total_length = 0
        self.files = []
        for file_path in self.file_paths:
            file = h5py.File(file_path, 'r')
            self.files.append(file)
            length = file['frame'].shape[0]
            self.lengths.append(length)
            self.total_length += length
        
        self.hlc_one_hot = hlc_one_hot

    def __len__(self):
        return self.total_length
    

    # one hot encoding for high-level ommand
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # find the file that contains the data for the given index
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            idx -= self.lengths[file_idx]
            file_idx += 1

        file = self.files[file_idx]

        # create a dictionary for the data sample
        hlc = torch.tensor(file['hlc'][idx][0], dtype=torch.long)
        if self.hlc_one_hot:
            hlc = F.one_hot(hlc.to(torch.int64), num_classes=4)
        sample = {
            'hlc': hlc,
            'controls': file['controls'][idx],
            'measurements': torch.tensor(file['measurements'][idx][0], dtype=torch.float32),
            'rgb': torch.tensor(file['rgb'][idx], dtype=torch.float32).permute(2, 0, 1),
            'segmentation': torch.tensor(file['segmentation'][idx], dtype=torch.float32).permute(2, 0, 1)
        }

        if self.transform:
            sample = self.transform(sample)

        # normlize the images and stack RGB with segmentation
        rgb = sample['rgb'] / 255.0
        segmentation = sample['segmentation'] / 255.0
        img = torch.cat((rgb, segmentation), dim=0)

        # normalize speed measurement
        sample['measurements'] /= 90.0
        
        # convert steer from [-1, 1] to [0, 1]
        controls = sample['controls']
        controls = torch.tensor([controls[0], (controls[1]+ 1.0) / 2.0, controls[2]], dtype=torch.float32)
        
        return img, sample['measurements'], sample['hlc'], controls

    def close(self):
        for file in self.files:
            file.close()

if __name__ == '__main__':
    # Set the directory where your .pkl files are
    data_dir = "data/"

    # Create the dataset
    dataset = CARLADataset(data_dir)

    # Create the dataloader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #set shuffle=False to take advantage of caching

    # Get a batch of data
    img_batch, speed_batch, hlc_batch, controls_batch = next(iter(dataloader))

    # Check the shapes of the data
    print("Image batch shape:", img_batch.shape)
    print("Speed batch shape:", speed_batch.shape)
    print("Controls batch shape:", controls_batch.shape)

    # Check the range of values in the image batch (should be [0, 1])
    print("Image batch min:", img_batch.min().item())
    print("Image batch max:", img_batch.max().item())

    # Check the range of values in the speed batch (should be [0, 1])
    print("Speed batch min:", speed_batch.min().item())
    print("Speed batch max:", speed_batch.max().item())

    # Check the range of values in the speed batch (should be [0, 1])
    print("HLC batch min:",hlc_batch.min().item())
    print("HLC batch max:", hlc_batch.max().item())

    # Check the range of values in the controls batch
    print("Controls batch min:", controls_batch.min().item())
    print("Controls batch max:", controls_batch.max().item())
