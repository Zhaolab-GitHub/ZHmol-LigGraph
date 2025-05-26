import os
import ast
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import time
from model import Net_coor

SPACE = 100
BOND_TH = 6.0

class PDBBindCoor(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', data_type='coor2', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.split = split
        self.data_type = data_type
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        if self.data_type == 'autodock':
            return ['test']
        return [self.split]
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        if self.data_type == 'autodock':
            return ['test.pt']
        return [self.split+'.pt']
        return ['train.pt', 'test.pt']

    def download(self):
        print('Hello Download')
        pass

    def process(self):
        if self.data_type == 'autodock':
            splits = ['test']
        else:
            splits = ['train', 'test']
        splits = [self.split]
        for split in splits:

            dataset_dir = os.path.join(self.raw_dir, f'{split}')
            files_num = len(
                glob.glob(os.path.join(dataset_dir, '*_data-G.json')))

            data_list = []
            graph_idx = 0

            pbar = tqdm(total=files_num)
            pbar.set_description(f'Processing {split} dataset')
            print(f'dataset_dir: {dataset_dir}')
        
            for f in range(files_num):

                with open(os.path.join(dataset_dir, f'{f}_data-G.json')) as gf:
                    graphs = gf.readlines()
                num_graphs_per_file = len(graphs)//3

                pbar.total = num_graphs_per_file * files_num
                pbar.refresh()

                feat_file = open(os.path.join(dataset_dir, f'{f}_data-feats'), 'rb')
                label_file = open(os.path.join(dataset_dir, f'{f}_label'), 'rb')


                for idx in range(num_graphs_per_file):
                    features = np.load(feat_file)
                    indptr = ast.literal_eval(graphs[3*idx])
                    indices = ast.literal_eval(graphs[3*idx+1])
                    dist = ast.literal_eval(graphs[3*idx+2])
                    flexible_len = np.load(label_file)# * 100
                    labels = np.load(label_file)# * 100
                    bonds = np.load(label_file)# * 100
                    pdb = np.load(label_file)


                    indptr = torch.LongTensor(indptr)
                    indices = torch.LongTensor(indices)
                    dist = torch.tensor(dist, dtype=torch.float)
                    row_idx = torch.ops.torch_sparse.ind2ptr(indptr,len(indices))[1:]
                    edge_index = torch.stack((row_idx, indices), dim=0)
                    

                    x = torch.Tensor(features)
                    y = torch.Tensor(labels)
                    bonds = torch.Tensor(bonds)
                    dist = dist.reshape(dist.size()[0], 3)
                    flexible_idx = torch.tensor(torch.LongTensor(range(features.shape[0])) < flexible_len[0])
                    flexible_len = torch.tensor(flexible_len)
                    y = y - x[flexible_idx, -3:]
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.bonds = bonds
                    data.dist = dist
                    data.pdb = pdb
                    data.flexible_idx = flexible_idx
                    data.flexible_len = flexible_len

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                
                    pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))

class PDBBindNextStep2(InMemoryDataset):
    def __init__(self, root, model_dir, pre_root, gpu_id=0, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.model_dir = model_dir
        self.pre_root = pre_root
        self.split = split

        gpu_id_ = str(gpu_id)
        device_str = 'cuda:' + gpu_id_ if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train', 'test'
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        print('Hello Download')
        pass

    @torch.no_grad()
    def process(self):
        for split in [self.split]:
            pre_dataset = PDBBindCoor(root=self.pre_root, split=split)
            pre_loader=DataLoader(pre_dataset, batch_size=1)
            model = torch.load(self.model_dir).to(self.device)
            model.eval()

            pbar = tqdm(total=len(pre_dataset))
            pbar.set_description(f'Processing {split} dataset')
            tot = 0
            data_list = []
            pbar.total = len(pre_dataset)
            pbar.refresh()
            loss_op = torch.nn.MSELoss()
            
            for data2 in pre_loader:
                data3 = data2.to(self.device)
                out1 = model(data3.x, data3.edge_index, data3.dist)
                out = out1.cpu().numpy()
                del out1
                del data3
                data = data2.to('cpu')
                
                x = data.x.numpy()
                idx = (data.flexible_idx==1).nonzero(as_tuple=True)[0].numpy()
                flexible_len = data.flexible_len.item()
                for i in idx:
                    x[i, -3:] += out[i]
                    data.y[i] -= out[i]
                coor = x[:, -3:]
                x = torch.Tensor(x)
               
                distt = []
                edges = [[], []]

                l1 = (data.flexible_idx[data.edge_index[0]]==0).nonzero(as_tuple=True)[0].numpy()
                l2 = (data.flexible_idx[data.edge_index[1]]==0).nonzero(as_tuple=True)[0].numpy()
                fix_idx = np.intersect1d(l1, l2)
                edge_index = data.edge_index[:, fix_idx]
                dist = data.dist[fix_idx]
                bond_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                bonds = data.edge_index[:, bond_idx].cpu().numpy()

                for i in range(flexible_len, x.size()[0]):
                    for j in idx:
                        if i <= j:
                            continue
                        c_i = coor[i]
                        c_j = coor[j]
                        dis = distance.euclidean(c_i, c_j)
                        dis = round(dis*100000) / 100000
                        if dis * SPACE < BOND_TH:
                            edges[0].append(i)
                            edges[1].append(j)
                            edges[0].append(j)
                            edges[1].append(i)
                            distt.append([0.0, dis, 0.0])
                            distt.append([0.0, dis, 0.0])

                for i in range(bonds.shape[1]):
                    ii = int(bonds[0][i])
                    jj = int(bonds[1][i])
                    edges[0].append(ii)
                    edges[1].append(jj)

                    c_i = coor[ii]
                    c_j = coor[jj]
                    dis = distance.euclidean(c_i, c_j)
                    dis = round(dis*100000) / 100000
                    distt.append([dis, 0.0, 0.0])
                edge_index = torch.cat((edge_index, torch.LongTensor(edges)), 1)
                dist = torch.cat((dist, torch.Tensor(distt)), 0)

                data_new = Data(x=x, edge_index=edge_index, y=data.y)
                data_new.dist = dist
                data_new.pdb = data.pdb
                data_new.flexible_idx = data.flexible_idx
                data_new.bonds = data.bonds
                data_new.flexible_len = data.flexible_len
                data_list.append(data_new)

                pbar.update(1)
            
            pbar.close()
            torch.save(self.collate(data_list),
                       os.path.join(self.processed_dir, f'{split}.pt'))
