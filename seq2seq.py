import time
import os
from os.path import join, exists
import argparse
import configparser
import random
import einops

import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils import data
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import faiss

from main import VPRModel
from dataset import PlaceDataset, PCADataset


def compute_pca(opt, config, model, full_features_dim):
    model = model.eval()
    pca_ds = PCADataset(config, dataset_folder=opt.trainset_path, split='train')
    num_images = min(len(pca_ds), 2 ** 14)
    if num_images < len(pca_ds):
        idxs = random.sample(range(0, len(pca_ds)), k=num_images)
    else:
        idxs = list(range(len(pca_ds)))
    subset_ds = Subset(pca_ds, idxs)
    dl = torch.utils.data.DataLoader(subset_ds, opt.infer_batch_size)

    pca_features = np.empty([num_images, full_features_dim])
    with torch.no_grad():
        for i, sequences in enumerate(tqdm(dl, ncols=100, desc="Database sequence descriptors for PCA: ")):
            if len(sequences.shape) == 5:
                sequences = einops.rearrange(sequences, "b s c h w -> (b s) c h w")         # seqeunces.size(): [40, 3, 320, 320]
            features = model(sequences).cpu().numpy()                                       # features.shape: (40, 4096)    
            features = features.reshape(-1, full_features_dim)                           
            pca_features[i * opt.infer_batch_size : (i * opt.infer_batch_size ) + len(features)] = features
    pca = PCA(opt.pca_dim)
    print(f'Fitting PCA from {full_features_dim} to 4096...')
    pca.fit(pca_features)
    return pca


def compute_recall(gt, predictions, numQ, n_values):
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
    return all_recalls


class InferencePipeline:
    def __init__(self, model, dataset, opt, config, batch_size=4, num_workers=4, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.feature_dim = opt.pca_dim
        self.opt = opt
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def run(self, split: str = 'db', pca=None) -> np.ndarray:

        if os.path.exists(join(self.opt.output_features_dir, f'global_descriptors_{split}.npy')):
            print(f"Skipping {split} features extraction, loading from cache")
            return np.load(join(self.opt.output_features_dir, f'global_descriptors_{split}.npy'))
        
        if not exists(self.opt.output_features_dir):
            os.makedirs(self.opt.output_features_dir)

        self.model.to(self.device)
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), int(self.config['seqLen'])*self.feature_dim), dtype=np.float32)
            for batch in tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features'):
                imgs, indices = batch
                imgs = imgs.to(self.device)
                imgs = imgs.view(-1, 3, int(self.config['imageresizeH']), int(self.config['imageresizeW']))
                indices = indices - self.dataset.dataset.num_index_seq if split == 'query' else indices

                # model inference
                descriptors = self.model(imgs)                                      # torch.Size([bsz*seqLen, 4096])
                descriptors = descriptors.view(-1, global_descriptors.shape[1])     # (bsz, seqLen*4096)
                descriptors = descriptors.detach().cpu().numpy()

                # add to global descriptors
                global_descriptors[indices] = descriptors

        if pca:
            global_descriptors = pca.transform(global_descriptors)

        # save global descriptors
        np.save(join(self.opt.output_features_dir, f'global_descriptors_{split}.npy'), global_descriptors)
        return global_descriptors


def load_model(ckpt_path):
    # Note that images must be resized to 320x320
    model = VPRModel(backbone_arch='resnet50',
                     layers_to_crop=[4],
                     agg_arch='MixVPR',
                     agg_config={'in_channels': 1024,
                                 'in_h': 20,
                                 'in_w': 20,
                                 'out_channels': 1024,
                                 'mix_depth': 4,
                                 'mlp_ratio': 1,
                                 'out_rows': 4},
                     )

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Feature-Extract')
    parser.add_argument('--config_path', type=str, default='config.ini',
                        help='File name (with extension) to the config file')
    parser.add_argument('--trainset_path', type=str, default='/home/divya/Datasets/MSLS_reformatted',
                        help='Path to the train set for training PCA')
    parser.add_argument('--dataset_path', type=str, default='/home/divya/Datasets/MSLS_val_reformatted/sf',
                        help='Full path to the directory containing the sequence directories')
    parser.add_argument('--pca_dim', type=int, default=4096, help='output size with PCA')
    parser.add_argument('--infer_batch_size', type=int, default=8)
    parser.add_argument('--output_features_dir', type=str, default='./output_features/sf',
                        help='Path to store all features')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')


    # Load the config file
    opt = parser.parse_args();  print(opt)
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    # Check GPU availability
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    # load model
    model = load_model(config['global_params']['resumepath'] + '.ckpt')

    dataset = PlaceDataset(config['global_params'], opt.dataset_path)
    query_dataset = Subset(dataset, range(dataset.num_index_seq, dataset.num_index_seq+dataset.num_query_seq))
    index_dataset = Subset(dataset, range(dataset.num_index_seq))

    # set up inference pipeline
    database_pipeline = InferencePipeline(model=model, dataset=index_dataset, opt=opt, 
                                          config=config['global_params'], device=device)
    query_pipeline = InferencePipeline(model=model, dataset=query_dataset, opt=opt, 
                                       config=config['global_params'], device=device)
    
    # Train PCA
    '''pca = compute_pca(opt, config['global_params'], model, 
                      full_features_dim=opt.pca_dim*int(config['global_params']['seqLen']))'''

    # run inference
    db_global_descriptors = database_pipeline.run(split='db', pca=None)         # shape: (num_db, feature_dim)
    query_global_descriptors = query_pipeline.run(split='query', pca=None)      # shape: (num_query, feature_dim)

    # calculate top-k matches
    '''top_k_matches = calculate_top_k(q_matrix=query_global_descriptors, db_matrix=db_global_descriptors, top_k=10)

    # record query_database_matches
    record_matches(top_k_matches, query_dataset, database_dataset, out_file='./LOGS/record.txt')

    # visualize top-k matches
    visualize(top_k_matches, query_dataset, database_dataset, visual_dir='./LOGS/visualize')'''

    tqdm.write('====> Building faiss index')
    faiss_index_cpu = faiss.IndexFlatL2(db_global_descriptors.shape[1])
    faiss_index_cpu.add(db_global_descriptors)

    n_values = [1, 5, 10, 20, 50, 100]

    start = time.time(); print("Getting predictions ...")
    _, predictions = faiss_index_cpu.search(query_global_descriptors, min(len(db_global_descriptors), max(n_values)))            
    print("Completed faiss index search for global descriptors in {} seconds".format(time.time() - start))
    del query_global_descriptors, db_global_descriptors

    gt = dataset.get_positives()    # predictions.shape = (len(query_dataset), 100), len(gt) = len(query_dataset)
    compute_recall(gt, predictions, dataset.num_query_seq, n_values)


if __name__ == '__main__':
    main()
