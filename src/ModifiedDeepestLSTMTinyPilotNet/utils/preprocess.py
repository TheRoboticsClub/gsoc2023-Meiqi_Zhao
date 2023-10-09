import torch
import numpy as np
import random
import torchvision.transforms.functional as F


class ShiftAndAdjustSteer:
    def __init__(self, shift_fraction=0.1, steer_adjust=1.0):
        self.shift_fraction = shift_fraction
        self.steer_adjust = steer_adjust

    def __call__(self, sample):
        # sample a random shift
        max_shift_px = self.shift_fraction * sample['rgb'].shape[2]
        shift_px = torch.tensor([random.uniform(-max_shift_px, max_shift_px), 0])

        # apply the shift to the images
        shifted_rgb = F.affine(sample['rgb'], angle=0, translate=shift_px.tolist(), scale=1, shear=0, interpolation=F.InterpolationMode.NEAREST)
        shifted_seg = F.affine(sample['segmentation'], angle=0, translate=shift_px.tolist(), scale=1, shear=0, interpolation=F.InterpolationMode.NEAREST)

        # adjust the steering angle according to the shift
        #print(f"Steer after: {sample['controls'][1]}")
        shift_fraction = shift_px[0] / sample['rgb'].shape[2]
        shifted_steer = sample['controls'][1] + self.steer_adjust * shift_fraction
        shifted_steer = torch.clamp(shifted_steer, -1.0, 1.0)

        sample['rgb'] = shifted_rgb
        sample['segmentation'] = shifted_seg
        sample['controls'][1] = shifted_steer
        #print(f"Steer after: {sample['controls'][1]}")

        return sample


class FilterClassesTransform:
    def __init__(self, classes_to_keep = [1, 7, 12, 13, 14, 15, 16, 17, 18, 19, 24], mode='both'):
        self.classes = {
            0: [0, 0, 0],         # Unlabeled  
            1: [128,  64, 128],   # Road ***
            2: [244,  35, 232],   # Sidewalk
            3: [70,  70,  70],    # Building
            4: [102, 102, 156],   # Wall
            5: [190, 153, 153],   # Fence
            6: [153, 153, 153],   # Pole
            7: [250, 170,  30],   # Traffic Light ***
            8: [220, 220,   0],   # Traffic Sign
            9: [107, 142,  35],   # Vegetation
            10: [152, 251, 152],  # Terrain
            11: [70, 130, 180],   # Sky
            12: [220,  20,  60],  # Pedestrain ***
            13: [255,   0,   0],  # Rider ***
            14: [0,   0, 142],    # Car ***
            15: [0,   0,  70],    # Truck ***
            16: [0,  60, 100],    # Bus ***
            17: [0,  80, 100],    # Train ***
            18: [0,   0, 230],    # Motorcycle ***
            19: [119,  11,  32],  # Bicycle ***
            20: [110, 190, 160],  # Static
            21: [170, 120,  50],  # Dynamic
            22: [55,  90,  80],   # Other
            23: [45,  60, 150],   # Water
            24: [157, 234,  50],  # Road Line ***
            25: [81,   0,  81],   # Ground
            26: [150, 100, 100],  # Bridge
            27: [230, 150, 140],  # Rail Track
            28: [180, 165, 180]   # Guard Rail
        }
        self.classes_to_keep = classes_to_keep
        self.mode = mode

    def __call__(self, sample):
        segmentation = sample['segmentation']
        rgb = sample['rgb']

        # Generate the mask from the original segmentation image
        mask = self.get_mask(segmentation, self.classes, self.classes_to_keep)

        if self.mode in ['both', 'segmentation']:
            # Apply mask to the segmentation image
            segmentation = self.apply_mask(segmentation, mask)
            sample['segmentation'] = segmentation

        if self.mode in ['both', 'rgb']:
            # Apply mask to the rgb image
            filtered_rgb = self.apply_mask(rgb, mask)
            sample['rgb'] = filtered_rgb

        return sample

    def apply_mask(self, image, mask):
        # Extend the 2D mask to 3D to match the image shape (C, H, W)
        mask_3d = mask.unsqueeze(0).expand_as(image)
        # Apply the mask
        return image * mask_3d

    def get_mask(self, segmentation, classes_dict, classes_to_keep):
        mask = torch.zeros(segmentation.shape[1:3], dtype=torch.bool)
        classes_of_interest = [torch.tensor(classes_dict[c]) for c in classes_to_keep]

        for class_rgb in classes_of_interest:
            mask = mask | torch.all(segmentation == class_rgb[:, None, None], dim=0)

        return mask