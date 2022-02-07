import os,math

import numpy as np
import pandas as pd
import tqdm
from PIL import Image

import torch
from torch.autograd import grad
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

def get_transform_cub():
    scale = 256.0/224.0
    target_resolution = (224, 224)
    center_transforms= [
        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        transforms.CenterCrop(target_resolution)
    ]
    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    augmentation_list = center_transforms + tensor_transforms
    transform = transforms.Compose(augmentation_list)
    return transform


def group_eval(X, y, g):
    output = {}
    for i, item in enumerate(g):
        if item.item() in output:
            output[item.item()][0].append(X[i])
            output[item.item()][1].append(y[i])
        else:
            output[item.item()] = [[X[i]],[y[i]]]
    return output


class WaterbirdsDataset(Dataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """
    def __init__(self, datapath):
        self.dataset_name = 'Waterbirds'
        self.data_dir = datapath
        # self.target_name = args.target_name
        self.confounder_names = 'place'

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.ID_array = self.metadata_df['img_filename'].str.split(".").str.get(0).astype(int)

        self.indices_by_group = self._get_group_indices()
        self.features_mat = None
        self.train_transform = get_transform_cub()
        self.eval_transform = get_transform_cub()

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        ID = self.ID_array[idx]
        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.eval_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
          self.eval_transform):
            img = self.eval_transform(img)
        x = img
        split = self.split_array[idx]
        return x,y,g,ID, split

    def _get_group_indices(self):
        group2indices = {}
        for g in range(self.n_groups):
            group_indices = np.where((self.group_array == g) & (self.split_array == self.split_dict['train']))[0]
            group2indices[g] = group_indices
        return group2indices

    def explicit_subsample(self, sizes):
        # We use this for support device confounder
        assert len(sizes) == self.n_groups
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        group_counts = np.array([(self.group_array[train_indices] == group_idx).sum() for group_idx in range(self.n_groups)])
        for i in range(self.n_groups):
            final_num_points = min([sizes[i], group_counts[i]])
            self.undersample(i, final_num_points)

    def undersample(self, undersample_group, new_size):
        # We use this for support device confounder
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        current_undersample_count  = (self.group_array[train_indices] == undersample_group).sum()
        if new_size > current_undersample_count:
            raise ValueError("Currently exist {} train examples in group {}, so cannot reduce to {}".format(current_undersample_count,
                undersample_group, new_size))
        undersample_indices = np.where((self.split_array == self.split_dict['train']) & (self.group_array == undersample_group))[0]
        undersample_delete_indices = np.random.choice(undersample_indices, current_undersample_count - new_size, replace = False) 
        self.y_array = np.delete(self.y_array, undersample_delete_indices)
        self.group_array = np.delete(self.group_array, undersample_delete_indices)
        self.filename_array = np.delete(self.filename_array, undersample_delete_indices)
        self.split_array = np.delete(self.split_array, undersample_delete_indices)

    def get_splits(self, splits):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            subsets[split] = Subset(self, indices)
            subsets[split].y_array = self.y_array[indices]
            subsets[split].group_array = self.group_array[indices]
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        assert len(self.confounder_names) == 1
        return group_name

class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn, group_weight_adjust = 0):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn

        if (self.dataset.y_array is not None) and (self.dataset.group_array is not None):
            y_array = self.dataset.y_array
            group_array = self.dataset.group_array
        else:
            group_array = np.empty(len(self))
            y_array = np.empty(len(self))
            for i,(x,y,g) in enumerate(self):
                group_array[i]=g
                y_array[i]=y

        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        if len(y_array.shape) == 2 and y_array.shape[1] > 1:
            #Just going to use the first class...
            self._y_counts = (torch.arange(2).unsqueeze(1) == self._y_array[:,0]).sum(1).float()
        else:
            self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g,ID in self:
            return x.size()

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = True
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]
            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader


class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]

class ChexpertSmall(Dataset):
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # folder to match the filename
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # select only the competition labels
    # attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    attr_names = ['Atelectasis']

    def __init__(self, root, mode='train', transform=None, data_filter=None, mini_data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        assert mode in ['train', 'valid', 'test', 'vis']
        self.mode = mode

        # if mode is test; root is path to csv file (in test mode), construct dataset from this csv;
        # if mode is train/valid; root is path to data folder with `train`/`valid` csv file to construct dataset.
        if mode == 'test':
            self.data = pd.read_csv(self.root, keep_default_na=True)
            self.root = self.root[:self.root.find('C')]  # base path; to be joined to filename in csv file in __getitem__
            self.data[self.attr_names] = pd.DataFrame(np.zeros((len(self.data), len(self.attr_names))))  # attr is vector of 0s under test
        else:
            # self._maybe_download_and_extract()
            self._maybe_process(data_filter)

            data_file = os.path.join(self.root, 'valid.pt' if mode in ['valid', 'vis'] else 'train.pt')
            self.data = torch.load(data_file)

            if mini_data is not None:
                # truncate data to only a subset for debugging
                self.data = self.data[:mini_data]

            if mode=='vis':
                # select a subset of the data to visualize:
                #   3 examples from each condition category + no finding category + multiple conditions category

                # 1. get data idxs with a/ only 1 condition, b/ no findings, c/ 2 conditions, d/ >2 conditions; return list of lists
                idxs = []
                data = self.data
                for attr in self.attr_names:                                                               # 1 only condition category
                    idxs.append(self.data.loc[(self.data[attr]==1) & (self.data[self.attr_names].sum(1)==1), self.attr_names].head(3).index.tolist())
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)==0, self.attr_names].head(3).index.tolist())  # no findings category
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)==2, self.attr_names].head(3).index.tolist())  # 2 conditions category
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)>2, self.attr_names].head(3).index.tolist())   # >2 conditions category
                # save labels to visualize with a list of list of the idxs corresponding to each attribute
                self.vis_attrs = self.attr_names + ['No findings', '2 conditions', 'Multiple conditions']
                self.vis_idxs = idxs

                # 2. select only subset
                idxs_flatten = torch.tensor([i for sublist in idxs for i in sublist])
                self.data = self.data.iloc[idxs_flatten]

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        img = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype('int')
        # attr = torch.from_numpy(attr)

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[idx]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
                                    # self.data.index(idx) pulls the index in the original dataframe and not the subset

        return img, attr[0], idx

    def __len__(self):
        return len(self.data)

    # def _maybe_download_and_extract(self):
    #     fpath = os.path.join(self.root, os.path.basename(self.url))
    #     # if data dir does not exist, download file to root and unzip into dir_name
    #     if not os.path.exists(os.path.join(self.root, self.dir_name)):
    #         # check if zip file already downloaded
    #         if not os.path.exists(os.path.join(self.root, os.path.basename(self.url))):
    #             print('Downloading ' + self.url + ' to ' + fpath)
    #             def _progress(count, block_size, total_size):
    #                 sys.stdout.write('\r>> Downloading %s %.1f%%' % (fpath,
    #                     float(count * block_size) / float(total_size) * 100.0))
    #                 sys.stdout.flush()
    #             request.urlretrieve(self.url, fpath, _progress)
    #             print()
    #         print('Extracting ' + fpath)
    #         with zipfile.ZipFile(fpath, 'r') as z:
    #             z.extractall(self.root)
    #             if os.path.exists(os.path.join(self.root, self.dir_name, '__MACOSX')):
    #                 os.rmdir(os.path.join(self.root, self.dir_name, '__MACOSX'))
    #         os.unlink(fpath)
    #         print('Dataset extracted.')

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        # check for processed .pt files
        train_file = os.path.join(self.root, 'train.pt')
        valid_file = os.path.join(self.root, 'valid.pt')
        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            # load data and preprocess training data
            valid_df = pd.read_csv(os.path.join(self.root, 'valid.csv'), keep_default_na=True)
            train_df = self._load_and_preprocess_training_data(os.path.join(self.root, 'train.csv'), data_filter)

            # save
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1,1)

        # if data_filter is not None:
        #     # 3. apply attr filters
        #     # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
        #     for k, v in data_filter.items():
        #         train_df = train_df[train_df[k]==v]

        #     with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
        #         json.dump(data_filter, f)

        return train_df


def extract_patient_ids(dataset, idxs):
    # extract a list of patient_id for prediction/eval results as ['CheXpert-v1.0-small/valid/patient64541/study1', ...]
    #    extract from image path = 'CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg'
    #    NOTE -- patient_id is non-unique as there can be multiple views under the same study
    return dataset.data['Path'].loc[idxs].str.rsplit('/', expand=True, n=1)[0].values


def compute_mean_and_std(dataset):
    m = 0
    s = 0
    k = 1
    for img, _, _ in tqdm(dataset):
        x = img.mean().item()
        new_m = m + (x - m)/k
        s += (x - m)*(x - new_m)
        m = new_m
        k += 1
    print('Number of datapoints: ', k)
    return m, math.sqrt(s/(k-1))