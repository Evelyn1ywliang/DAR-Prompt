import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import pickle
import h5py
# from keytotext import pipeline
from tqdm import tqdm

class NUSWIDE_ZSL_TEXT_IMAGE(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, p_text=1, annFile="", label_mask=None, partial=1+1e-6, h5_file ='train_data_0_143528_with_sentence.h5'):
        # data_split = train / val
        ann_file_names = {'train': 'formatted_train_all_labels_filtered.npy',
                          'val': 'formatted_val_all_labels_filtered.npy',
                          'val_gzsl': 'formatted_val_gzsl_labels_filtered_small.npy',
                          'test_gzsl': 'formatted_val_gzsl_labels_filtered.npy'}
        img_list_name = {'train': 'formatted_train_images_filtered.npy',
                         'val': 'formatted_val_images_filtered.npy',
                         'val_gzsl': 'formatted_val_gzsl_images_filtered_small.npy',
                         'test_gzsl': 'formatted_val_gzsl_images_filtered.npy'}
        self.root = root
        class_name_files = os.path.join(self.root, 'annotations', 'Tag_all', 'all_labels.txt')
        with open(class_name_files) as f:
            classnames = f.readlines()
        self.classnames = [a.strip() for a in classnames]

        if annFile == "":
            annFile = os.path.join(self.root, 'annotations', 'zsl', ann_file_names[data_split])
        else:
            raise NotImplementedError
        cls_id = pickle.load(open(os.path.join(self.root, 'annotations', 'zsl', "cls_id.pickle"), "rb"))
        if data_split == 'train':
            cls_id = cls_id['seen']
        elif data_split == 'val':
            cls_id = cls_id['unseen']
        elif data_split in ['val_gzsl', 'test_gzsl']:
            cls_id = list(range(1006))
        else:
            raise ValueError
        self.cls_id = cls_id
        image_list = os.path.join(self.root, 'annotations', 'zsl', img_list_name[data_split])
        self.anns = np.load(annFile)

        self.image_list = np.load(image_list)
        assert len(self.anns) == len(self.image_list)  

        self.data_split = data_split
        ids = list(range(len(self.image_list)))
        if data_split == 'train':
            num_examples = len(ids)
            pick_example = int(num_examples * p)
            self.ids = ids[:pick_example]
            pick_text_example = int(num_examples * p_text)
            self.text_ids = ids[:pick_text_example]  
        else:
            self.ids = ids
            

        with h5py.File(h5_file, 'r') as h5f:
            self.st_data = h5f['sentence'][:]
            self.st_data = [str(it, encoding='utf-8') for it in self.st_data.tolist()]
            self.label = h5f['label'][:]

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(img_size)
            transforms.Resize((img_size, img_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_transform = transforms.Compose([
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        if self.data_split == 'train':
            self.transform = train_transform
        elif self.data_split in ['val', 'val_gzsl', 'test_gzsl']:
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Nus Wide' % self.data_split)

        # create the label mask
        self.mask = None
        self.partial = partial
        if data_split == 'train' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.ids), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask], dim=1)
                torch.save(mask, os.path.join(self.root, 'annotations', 'partial_label_%.2f.pt' % partial))
            else:
                mask = torch.load(os.path.join(self.root, 'annotations', label_mask))
            self.mask = mask.long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, 'images', self.image_list[img_id].strip())
        img = Image.open(img_path).convert('RGB')
        
        targets = self.anns[index]
        targets = torch.from_numpy(targets).long()
        
        if index in self.text_ids:
            # Check whether the sentence's label matches the current image labels.
            sanity_label = self.label[index]
            sanity_label = torch.from_numpy(sanity_label).long()
            
            sanity_check = torch.eq(targets, sanity_label).all() if targets.shape == sanity_label.shape else False
            
            assert sanity_check, "targets and sanity_label are not the same."
            
            st = self.st_data[index]
        else:
            st = ''
        
        target = targets[None, ]
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

        if self.transform is not None:
            img = self.transform(img)

        return img, st, target

    def name(self):
        return 'nus_wide'
    
    def generate_text(self):
        
        pick_arr = self.anns[self.ids]
        target_arr = np.array(pick_arr) == 1  
        
        
        cls_name_arr = np.array(self.classnames).astype(np.str_).reshape(1, -1)
        cls_name_arr = np.repeat(cls_name_arr, repeats=len(self.ids), axis=0)
        
        target_names = []
        sentence_list = []
        label = []
        name_list = None
        for i, ids in enumerate(tqdm(self.ids)):
            name_list = (cls_name_arr[i][self.anns[ids].astype(bool)].tolist())
            target_names.append(name_list)
            # new_sentences =  self.nlp(name_list, num_return_sequences=10, do_sample=True)
            # choose the best sentence
            best_inter = 0
            # generate 10 sentences
            for st_idx in range(10):
                
                st =  self.nlp(name_list, num_return_sequences=1, do_sample=True)
                intersection = set(st.split()) & set(name_list)
                if len(intersection) == len(name_list):
                    new_sentence = st
                    break
                elif len(intersection) > best_inter:
                    new_sentence = st
                    best_inter = len(intersection) 
                
            sentence_list.append(new_sentence)
            label.append(target_arr[i].astype(np.int))
            
        
        with h5py.File('train_data_{}_with_sentence.h5'.format(len(self.ids)),'w') as h5f:
            # h5f.create_dataset("name", data=np.asarray(name_list) )
            # h5f.create_dataset("sentence", data=np.asarray(sentence_list ))
            h5f.create_dataset("label", data=np.asarray(label) )   
            dt = h5py.special_dtype(vlen=str)
            h5_st = h5f.create_dataset('sentence', len(label), dtype=dt)
            # h5_name = h5f.create_dataset('name', len(label), dtype=dt)
            for i in range(len(label)):
                h5_st[i] = sentence_list[i]
                # h5_name[i] = target_names[i]
        
        print("finish!")
