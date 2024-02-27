import os

import torchvision.transforms as transforms
import torch.utils.data as data
import torch

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from glob import glob
from os.path import join
from tqdm import tqdm


def input_transform(resize=(480, 640)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def build_sequences(seqLen, dataset_path, desc=None):
    base_path = os.path.dirname(dataset_path)
    paths, all_paths, idx_frame_to_seq = list(), list(), list()
    seqs_folders = sorted(glob(join(dataset_path, '*'), recursive=True))
    for seq in tqdm(seqs_folders, ncols=100, desc=desc):
        start_index = len(all_paths)
        frame_nums = np.array(list(map(lambda x: int(x.split('@')[4]), sorted(glob(join(seq, '*'))))))
        full_seq_paths = sorted(glob(join(seq, '*')))
        seq_paths = np.array([s_p.replace(f'{base_path}/', '') for s_p in full_seq_paths])

        sorted_idx_frames = np.argsort(frame_nums)
        all_paths += list(seq_paths[sorted_idx_frames])

        for idx, _ in enumerate(frame_nums):
            if idx < (seqLen // 2) or idx >= (len(frame_nums) - seqLen // 2): continue
            seq_idx = np.arange(-seqLen // 2, seqLen // 2) + 1 + idx 
            if (np.diff(frame_nums[sorted_idx_frames][seq_idx]) == 1).all():
                paths.append(",".join(seq_paths[sorted_idx_frames][seq_idx]))
                idx_frame_to_seq.append(seq_idx + start_index)
    
    return paths, np.array(all_paths), np.array(idx_frame_to_seq)


class PlaceDataset(data.Dataset):
    def __init__(self, config, dataset_path):
        super().__init__()

        self.seqLen = int(config['seqLen'])
        self.resize = (int(config['imageresizeH']), int(config['imageresizeW']))
        self.mytransform = input_transform(self.resize)
        self.posThresh = int(config['posThresh'])

        self.dataset_path = dataset_path
        query_dataset_path = join(self.dataset_path, "queries")
        index_dataset_path = join(self.dataset_path, "database")

        self.query_seq_paths, self.all_q_paths, self.q_idx_frame_to_seq = build_sequences(self.seqLen, query_dataset_path, desc='loading query..')              # 11185, (14901, ), (11185, 5)
        self.index_seq_paths, self.all_index_paths, self.index_idx_frame_to_seq = build_sequences(self.seqLen, index_dataset_path, desc='loading database..')
        self.seq_paths = self.index_seq_paths + self.query_seq_paths

        self.compute_positives()           
        
        self.num_query_seq = len(self.qIdx)      
        self.num_index_seq = len(self.index_seq_paths)        

    
    def compute_positives(self):
        q_unique_idxs = np.unique([idx for seq_frames_idx in self.q_idx_frame_to_seq for idx in seq_frames_idx])
        index_unique_idxs = np.unique([idx for seq_frames_idx in self.index_idx_frame_to_seq for idx in seq_frames_idx])

        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.all_index_paths[index_unique_idxs]]).astype(np.float64)  # (31594, 2)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.all_q_paths[q_unique_idxs]]).astype(np.float64)          # (14729, 2)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=self.posThresh,
                                                             return_distance=False)         # shape: (14729, )
                                                                                                                                        
        self.qIdx, self.pIdx = list(), list()
        for q in tqdm(range(len(self.q_idx_frame_to_seq)), ncols=100, desc='Finding positives and negatives...'):
            q_frame_idxs = self.q_idx_frame_to_seq[q]
            unique_q_frame_idxs = np.where(np.in1d(q_unique_idxs, q_frame_idxs))

            p_uniq_frame_idxs = np.unique(
                [p for pos in self.hard_positives_per_query[unique_q_frame_idxs] for p in pos])

            if len(p_uniq_frame_idxs) > 0:                                                  # 555 query sequences don't meet this condition
                p_seq_idx = np.where(np.in1d(self.index_idx_frame_to_seq, index_unique_idxs[p_uniq_frame_idxs])
                                     .reshape(self.index_idx_frame_to_seq.shape))[0]
                self.qIdx.append(q)
                self.pIdx.append(np.unique(p_seq_idx))

        self.qIdx = np.array(self.qIdx)
        self.pIdx = np.array(self.pIdx, dtype=object)           # len(self.pIdx) is 11185-555 = 10630


    def __getitem__(self, index):
        old_index = index
        if index >= self.num_index_seq:
            q_index = index - self.num_index_seq
            index = self.qIdx[q_index] + self.num_index_seq
        img = torch.stack([self.mytransform(Image.open(join(self.dataset_path, im))) for im in self.seq_paths[index].split(',')])
        return img, old_index

    def __len__(self):
        return len(self.seq_paths)

    def get_positives(self):
        return self.pIdx


class PCADataset(data.Dataset):
    def __init__(self, config, dataset_folder="dataset", split="train"):
        super().__init__()
        self.seq_len = int(config['seqLen'])
        self.resize = (int(config['imageresizeH']), int(config['imageresizeW']))
        self.base_transform = input_transform(self.resize)

        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist.")

        '''if 'robotcar' in dataset_folder:
            folders = list(product(['train', 'val'], ['queries', 'database'])) + [('test', 'database')]
            self.db_paths = []
            for folder in folders:
                split, subset = folder[0], folder[1]
                load_folder = join(dataset_folder, split, subset)
                paths, _, _ = build_sequences(load_folder, seq_len=self.seq_len,
                                              cities=cities, desc="Loading database to compute PCA...")
                self.db_paths += paths
        else:'''
        self.dataset_folder = join(dataset_folder, split)
        database_folder = join(self.dataset_folder, "database")
        self.db_paths, _, _ = build_sequences(self.seq_len, database_folder,
                                                desc="Loading database to compute PCA...")

        self.db_num = len(self.db_paths)

    def __getitem__(self, index):
        # Note MSLSBase uses an old_index variable that is also returned
        img = torch.stack([self.base_transform(Image.open(join(self.dataset_folder, path))) for path in self.db_paths[index].split(',')])
        return img

    def __len__(self):
        return self.db_num

    def __repr__(self):
        return (
            f"< {self.__class__.__name__}, ' #database: {self.db_num} >")