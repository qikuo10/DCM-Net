# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
***  
Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Chinedu Innocent Nwoye
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pickle




class CholecT50():
    def __init__(self,
                 args,
                 dataset_dir,
                 dataset_variant="cholect45-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                 model=None):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int, 
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label))
        """
        self.args = args
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect45-challenge": "for CholecT45 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
        }
        assert dataset_variant in self.list_dataset_variant.keys(), print(dataset_variant,
                                                                          "is not a valid dataset variant")
        video_split = self.split_selector(case=dataset_variant)
        # Fix: kfold starts from 0, but dict keys start from 1
        # kfold=0 -> fold 1, kfold=1 -> fold 2, etc.
        actual_fold = self.args.kfold + 1 if 'crossval' in dataset_variant else self.args.kfold
        
        train_videos = sum([v for k, v in video_split.items() if k != actual_fold],
                           []) if 'crossval' in dataset_variant else video_split['train']
        test_videos = sum([v for k, v in video_split.items() if k == actual_fold],
                          []) if 'crossval' in dataset_variant else video_split['test']
        if 'crossval' in dataset_variant:
            val_videos = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos = video_split['val']
        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
        # self.test_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.augmentations = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90, expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
        }
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])
        trainform, testform = self.transform()

        self.build_train_dataset(testform, model=model)
        self.build_val_dataset(testform, model=model)
        self.build_test_dataset(testform, model=model)
        self.build_test_train_dataset(testform, model=model)

    def list_dataset_variants(self):
        print(self.list_dataset_variant)

    def list_augmentations(self):
        print(self.augmentations.keys())

    def split_selector(self, case='cholect50'):
        switcher = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35,
                          48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val': [8, 12, 29, 50, 78],
                'test': [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25,
                          49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val': [5, 18, 36, 65, 74],
                'test': [92, 96, 103, 110, 111]
            },
            'cholect45-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 5, 23, 35, 48, 60, 18, 13, 25,
                          49, 62, 65, 8, 12, 29, 50, 78, 6, 51, 10, 36, 14, 32, 80, 42],
                'val': [68, 70, 73, 74, 75],
                # 'test': [92, 96, 103, 110, 111]
                'test': [68, 70, 73, 74, 75]
            },
            'cholect45-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50, ],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70, ],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73, ],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, ],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13, ],
            },
            'cholect50-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70, 96],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13, 92],
            },
        }
        return switcher.get(case)

    def no_augumentation(self, x):
        return x

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        op_test = [transforms.Resize((256, 448)), transforms.Resize((256, 448)), transforms.ToTensor(), normalize, ]
        op_train = [transforms.Resize((256, 448))] + self.augmentation_list + [transforms.Resize((256, 448)),
                                                                               transforms.ToTensor(), normalize, ]
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform

    def build_train_dataset(self, transform, model):
        iterable_dataset = []
        print(f"[DEBUG] Building train dataset with {len(self.train_records)} videos: {self.train_records}")
        for video in self.train_records:
            try:
                dataset = T50(args=self.args, split='train', img_dir=os.path.join(self.dataset_dir, 'data', video),
                              triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                              tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                              verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                              target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                              transform=transform,
                              model=model)
                iterable_dataset.append(dataset)
                print(f"[DEBUG] Successfully loaded train video: {video}")
            except Exception as e:
                print(f"[ERROR] Failed to load train video {video}: {e}")
                # Continue without this video
        print(f"[DEBUG] Train dataset built with {len(iterable_dataset)} videos")
        self.train_dataset = ConcatDataset(iterable_dataset)
        # self.train_dataset = iterable_dataset

    def build_val_dataset(self, transform, model):
        iterable_dataset = []
        print(f"[DEBUG] Building val dataset with {len(self.val_records)} videos: {self.val_records}")
        for video in self.val_records:
            dataset = T50(args=self.args, split='val', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
        print(f"[DEBUG] Val dataset built with {len(iterable_dataset)} videos")
        self.val_dataset = iterable_dataset

    def build_test_dataset(self, transform, model):
        iterable_dataset = []
        print(f"[DEBUG] Building test dataset with {len(self.test_records)} videos: {self.test_records}")
        for video in self.test_records:
            try:
                dataset = T50(args=self.args, split='test', img_dir=os.path.join(self.dataset_dir, 'data', video),
                              triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                              tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                              verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                              target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                              transform=transform,
                              model=model)
                iterable_dataset.append(dataset)
                print(f"[DEBUG] Successfully loaded test video: {video}")
            except Exception as e:
                print(f"[ERROR] Failed to load test video {video}: {e}")
                # Continue without this video
        print(f"[DEBUG] Test dataset built with {len(iterable_dataset)} videos")
        self.test_dataset = iterable_dataset

    def build_test_train_dataset(self, transform, model):
        iterable_dataset = []
        for video in self.train_records[-9:]:
            dataset = T50(args=self.args, split='test', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
        self.test_train_dataset = iterable_dataset

    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset, self.test_train_dataset)


class T50(Dataset):
    def __init__(self, args, split, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None,
                 target_transform=None, model=None):
        self.args = args
        self.split = split
        self.img_dir = img_dir
        
        # Modified: Use SwinB384 features (1024-dim) instead of Swin-Tiny (768-dim)
        # self.feats_dir = '../0-5fold/data_feats/run_{}'.format(self.args.version1)
        
        # Feature directory: determined by --version1 argument
        script_dir = os.path.dirname(os.path.abspath(__file__))
        feat_name = self.args.version1  # e.g. swinB_features, swinT_features, swinB384_features
        possible_paths = [
            os.path.join(script_dir, '..', 'Backbone_0-5fold', 'data_feats', feat_name),
            os.path.join(script_dir, '..', '0-5fold', 'data_feats', feat_name),
        ]
        
        self.feats_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                self.feats_dir = path
                break
        
        if self.feats_dir is None:
            raise FileNotFoundError(
                f"Could not find feature directory '{feat_name}'. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nPlease ensure features are extracted to Backbone_0-5fold/data_feats/{feat_name}/"
            )
        
        print(f"Using features directory: {self.feats_dir}")
        
        # Load features based on split
        if split == 'train':
            # For training, load the training set features (excluding current fold)
            feats_file = os.path.join(self.feats_dir, 'k' + str(self.args.kfold) + '_train_feats.pkl')
        else:
            # For validation/test, load the test set features (current fold only)
            feats_file = os.path.join(self.feats_dir, 'k' + str(self.args.kfold) + '_feats.pkl')
            # Fallback: if test file doesn't exist, try train file (for cholect45 with all videos in one file)
            if not os.path.exists(feats_file):
                print(f"Warning: Test features file not found, using train features file")
                feats_file = os.path.join(self.feats_dir, 'k' + str(self.args.kfold) + '_train_feats.pkl')
        
        print(f"Loading features from: {feats_file}")
        
        # Handle numpy version compatibility issue
        # This fixes "No module named 'numpy._core'" error when loading pickles
        # created with newer numpy versions
        import sys
        try:
            import numpy.core._multiarray_umath as _multiarray_umath
            sys.modules['numpy._core'] = sys.modules['numpy.core']
            sys.modules['numpy._core._multiarray_umath'] = _multiarray_umath
        except (ImportError, AttributeError):
            # If already using new numpy, no need to patch
            pass
        
        with open(feats_file, 'rb') as f:
            feats_dict = pickle.load(f)
            # Extract video ID from img_dir (e.g., 'data/VID79' -> 'VID79')
            # Try different formats: 'VID79', '79', or last part of path
            video_id = os.path.basename(self.img_dir)  # Get 'VID79' from 'data/VID79'
            
            # Try to find the video in the dictionary
            if video_id in feats_dict:
                self.feats = feats_dict[video_id]
            elif video_id[-2:] in feats_dict:  # Try just the number '79'
                self.feats = feats_dict[video_id[-2:]]
            elif f'VID{video_id[-2:]}' in feats_dict:  # Try 'VID79' format
                self.feats = feats_dict[f'VID{video_id[-2:]}']
            else:
                # Video not found - this might be a cholect50 video when using cholect45 features
                available_videos = sorted(feats_dict.keys())
                error_msg = (
                    f"\n{'='*80}\n"
                    f"ERROR: Video {video_id} not found in features file!\n"
                    f"{'='*80}\n"
                    f"Tried: {video_id}, {video_id[-2:]}, VID{video_id[-2:]}\n"
                    f"Available videos ({len(available_videos)}): {available_videos[:10]}...\n"
                    f"\n"
                    f"Possible causes:\n"
                    f"1. Using cholect50 dataset with cholect45 features\n"
                    f"2. Missing video in feature extraction\n"
                    f"\n"
                    f"Solution: Ensure --dataset_variant matches your features\n"
                    f"  - If features are from cholect45: use --dataset_variant cholect45-crossval\n"
                    f"  - If features are from cholect50: use --dataset_variant cholect50-crossval\n"
                    f"{'='*80}\n"
                )
                raise KeyError(error_msg)
        
        print(f"Loaded features for video {video_id}, shape: {self.feats.shape} (Swin-Base 1024-dim)")
        
        sub1 = self.feats[1:, :] - self.feats[:-1, :]
        idx1 = np.where(np.sum(sub1, axis=-1) == 0)[0]
        idx2 = np.unique(np.concatenate((idx1, idx1 + 1)))
        idx = [i for i in range(len(self.feats)) if i not in list(idx2)]
        self.idx = [1 if i in idx else 0 for i in range(len(self.feats))]
        self.feats = self.feats[idx, :]
        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',', )[idx, :]
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',', )[idx, :]
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',', )[idx, :]
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',', )[idx, :]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        if self.split == 'train' and random.random() > 0.7:
            num_clips = random.choice(range(10, 1000 if len(self.feats) > 1000 else len(self.feats)))
            random_index = random.choice(range(0, len(self.feats) - num_clips))
            idx = [random_index + i for i in range(num_clips)]
        else:
            idx = [i for i in range(len(self.feats))]
        triplet_label = self.triplet_labels[idx, 1:]
        tool_label = self.tool_labels[idx, 1:]
        verb_label = self.verb_labels[idx, 1:]
        target_label = self.target_labels[idx, 1:]
        image = self.feats[idx]
        if self.target_transform:
            triplet_label = self.target_transform(triplet_label)
        return image, (tool_label, verb_label, target_label, triplet_label), self.img_dir


if __name__ == "__main__":
    print("Refers to https://github.com/CAMMA-public/cholect45 for the usage guide.")
