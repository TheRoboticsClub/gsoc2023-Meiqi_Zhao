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
    def __init__(self, classes_to_keep = [4, 6, 7, 10], mode='both'):
        self.classes = {
            0: [0, 0, 0],         # Unlabeled
            1: [70, 70, 70],      # Buildings
            2: [100, 40, 40],     # Fences
            3: [55, 90, 80],      # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 142],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0],    # TrafficSigns
            13: [70, 130, 180],   # Sky
            14: [81, 0, 81],      # Ground
            15: [150, 100, 100],  # Bridge
            16: [230, 150, 140],  # RailTrack
            17: [180, 165, 180],  # GuardRail
            18: [250, 170, 30],   # TrafficLight
            19: [110, 190, 160],  # Static
            20: [170, 120, 50],   # Dynamic
            21: [45, 60, 150],    # Water
            22: [145, 170, 100],  # Terrain
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