import os
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import nibabel as nib
import torch


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = \
            cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[
                ..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "idx": idx}
        return sample




class DSADataSets(Dataset):
    def __init__(self, data_path, transform=None, transform_msk=None, mode='Training'):

        self.transform = transform
        self.mode = mode
        self.data_path = data_path
        self.transform_msk = transform_msk
        # 打开文件
        if self.mode == 'Test':
            with open(os.path.join(data_path, 'test_data.txt'), 'r') as file:
                # 读取文件的每一行
                lines = file.readlines()
        elif self.mode == 'Training':
            with open(os.path.join(data_path, 'train_data.txt'), 'r') as file:
                # 读取文件的每一行
                lines = file.readlines()

        self.name_list = [line.strip() for line in lines]
        ori = []
        mask = []
        for name in self.name_list:
            img_path = os.path.join(self.data_path, 'ori', name)
            msk_path = os.path.join(self.data_path, 'mask', name)
            img1 = np.array(Image.open(img_path).convert('L'))
            mask1 = np.array(Image.open(msk_path).convert('L'))
            ori.append(img1)
            mask.append(mask1)
        self.ori = np.array(ori).astype(np.uint8)
        self.mask = np.array(mask).astype(np.uint8)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):

        """Get the images"""
        img = self.ori[index, :, :]
        mask = self.mask[index, :, :]
        img = self.transform(img)
        mask = self.transform(mask)
        # img=np.stack((img, img1, img2), axis=2)
        # print(img.shape)

        return {
            'image': img,
            'mask': mask
        }


def load_nii_files(file_paths, mode='ori'):
    data_list = []
    index_list = []
    for index, file_path in enumerate(file_paths):
        nii_img = nib.load(file_path)
        numpy_array = nii_img.get_fdata()

        data_list.append(numpy_array)
        index_list.append((numpy_array.shape[-1]))
    data = np.concatenate(data_list, axis=2).astype(np.float32)
    index = np.array(index_list)

    return data, index



def get_index(index):
    temp = np.arange(np.sum(index))
    # print(len(temp),temp)
    index1 = []

    tail = []
    head = []
    total = 0
    for each in index:
        index1.append(temp[total + 1:total + each])
        total += each
        tail0 = total - 1
        for i in range(each):  # each - 1
            tail.append(tail0)
            head.append(total - each)

    index1 = np.concatenate(index1)
    tail = np.array(tail)
    head = np.array(head)

    return index1, temp, tail, head


train_transformer3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class DSA_keyframe(Dataset):
    '''处理dsa数据，包括dsa图片、标签、图片名称
    parameters
        data_path: str, 数据路径
        is_train: bool, 训练集 or 测试集
        online_aug: bool, 在线数据增强 or not
        offline_aug: bool, 离线数据增强 or not
        fold_num: int, 十折交叉的fold_num折
    '''

    def __init__(self, data_path, fold=1, type='1d', mode='Training', num_class=2):

        self.mode = mode
        self.data_path = data_path
        self.type = type
        self.num_class = num_class

        if self.mode == 'Test':
            with open(os.path.join(data_path, '5fold index', 'test_data{}.txt').format(fold), 'r') as file:
                # 读取文件的每一行
                lines = file.readlines()
        elif self.mode == 'Training':
            with open(os.path.join(data_path, '5fold index', 'train_data{}.txt').format(fold), 'r') as file:
                # 读取文件的每一行
                lines = file.readlines()

        self.name_list = [line.strip() for line in lines]

        file_paths = [os.path.join(data_path, f + ' ori.nii.gz') for f in self.name_list]
        # 加载NIfTI文件并拼接成一个大数组
        self.ori_data, self.ori_index = load_nii_files(file_paths)

        file_paths = [os.path.join(data_path, f + ' mask.nii.gz') for f in self.name_list]
        # 加载NIfTI文件并拼接成一个大数组
        self.mask_data, _ = load_nii_files(file_paths, mode='mask')
        self.mask_data[self.mask_data > 0] = 1

        if self.num_class == 2 or self.num_class == 6:
            file_paths = [os.path.join(data_path, f + ' catheter.nii.gz') for f in self.name_list]
            # 加载NIfTI文件并拼接成一个大数组
            self.catheter_data, _ = load_nii_files(file_paths, mode='mask')
            self.catheter_data[self.catheter_data > 0] = 1
            print('catheter mask ready...')


        print(mode, "data 数据形状:", self.ori_data.shape)
        print("索引形状:", self.ori_index.shape)
        self.real_index, self.ori0_index, self.tail_index, self.head_index = get_index(self.ori_index)
        '''get namlst'''
        self.name_list1 = []
        for num, string in zip(self.ori_index, self.name_list):
            self.name_list1.extend([string] * num)

        print(self.__getlen__())

    def __len__(self):
        if self.type == '4d' or self.type == '4d-' or self.type == 'catheter_vessel_2calssification':
            return len(self.real_index)
        elif self.type == '1d' or self.type == 'catheter' or self.type == 'catheter_vessel' or self.type == '3d' or self.type == '4dnew':
            return np.sum(self.ori_index)

    def __getlen__(self):
        print(self.real_index)
        print(self.tail_index)
        return [len(self.name_list), np.sum(self.ori_index), len(self.real_index)]

    def __getitem__(self, index):

        """Get the images"""

        if self.num_class == 2:
            patient_seg_label = self.mask_data[:, :, index]
            patient_ca_label = self.catheter_data[:, :, index]
            patient_seg_label = train_transformer3(patient_seg_label)
            patient_ca_label = train_transformer3(patient_ca_label)
            patient_seg_label = torch.cat((patient_seg_label, patient_ca_label), dim=0)  # 获得三通道图像，匹配维度

        elif self.num_class == 1:
            patient_seg_label = self.mask_data[:, :, index]  # self.mask_data[:, :, index]
            patient_seg_label = train_transformer3(patient_seg_label)



        if self.type == '1d':
            patient = self.ori_data[:, :, index].astype(np.uint8)
            patient_name = self.name_list1[index] + ' ' + str(index)
            # print('before train_transformer3:',patient.min(),patient.max())
            # print('before totensor:',patient[:10,:10])
            patient = train_transformer3(patient)
            # print('after totensor:',patient[:10,:10])
            # print('after train_transformer3:',patient.min(),patient.max())
            patient = torch.cat((patient, patient, patient), dim=0)  # 获得三通道图像，匹配维度
            # print(patient.max())
            sample = {'image': patient, 'mask': patient_seg_label, 'patient_name': patient_name}


        elif self.type == '3d':
            headindex = self.head_index[index]
            tailindex = self.tail_index[index]

            # index1 = self.real_index[index]
            patient1 = self.ori_data[:, :, headindex].astype(np.uint8)
            patient2 = self.ori_data[:, :, index].astype(np.uint8)
            patient3 = self.mask_data[:, :, tailindex].astype(np.uint8)

            patient_name = str(index)
            patient1 = train_transformer3(patient1)
            patient2 = train_transformer3(patient2)
            patient3 = train_transformer3(patient3)

            patient = torch.cat((patient1, patient2, patient3), dim=0)  # 获得三通道图像，匹配维度
            sample = {'image': patient, 'mask': patient_seg_label, 'patient_name': patient_name}






        elif self.type == '4dnew1':
            tailindex = self.tail_index[index]
            headindex = self.head_index[index]
            patient1 = self.ori_data[:, :, index].astype(np.uint8)
            patient2 = self.ori_data[:, :, tailindex].astype(np.uint8)
            patient3 = self.mask_data[:, :, tailindex].astype(np.uint8)
            patient4 = self.ori_data[:, :, headindex].astype(np.uint8)

            patient_name = self.name_list1[index] + ' ' + str(index)
            patient1 = train_transformer3(patient1)
            patient2 = train_transformer3(patient2)
            patient3 = train_transformer3(patient3)
            patient4 = train_transformer3(patient4)
            # print('after totensor:',patient[:10,:10])
            # print('after train_transformer3:',patient.min(),patient.max())
            patient = torch.cat((patient1, patient2, patient3, patient4), dim=0)  # 获得三通道图像，匹配维度
            # print(patient.max())
            sample = {'image': patient, 'mask': patient_seg_label, 'patient_name': patient_name}


        elif self.type == '4d':
            index1 = self.real_index[index]
            patient1 = self.ori_data[:, :, index1].astype(np.uint8)
            patient2 = self.ori_data[:, :, index1 - 1].astype(np.uint8)
            patient3 = self.mask_data[:, :, index1 - 1].astype(np.uint8)
            patient4 = self.ori_data[:, :, self.tail_index[index]].astype(np.uint8)

            patient_seg_label = self.mask_data[:, :, index1]  # .astype(np.uint8)
            patient_name = str(index)
            patient1 = train_transformer3(patient1)
            patient2 = train_transformer3(patient2)
            patient3 = train_transformer3(patient3)
            patient4 = train_transformer3(patient4)

            patient_seg_label = train_transformer3(patient_seg_label)
            patient = torch.cat((patient1, patient2, patient3, patient4), dim=0)  # 获得三通道图像，匹配维度
            # print('patient max',patient.max())
            sample = {'patient': patient, 'seg_label': patient_seg_label, 'patient_name': patient_name}


        elif self.type == 'catheter':  # self.ori0_index,self.tail_index,self.head_index

            img0 = self.ori_data[:, :, self.head_index[index]].astype(np.uint8)
            imgt = self.ori_data[:, :, index].astype(np.uint8)
            imgT = (self.ori_data[:, :, self.tail_index[index]] - self.ori_data[:, :, index]).astype(np.uint8)
            mask0 = self.catheter_mask[:, :, self.head_index[index]].astype(np.uint8)
            # maskt=self.catheter_mask[:,:,index].astype(np.uint8)

            patient_seg_label = self.mask_data[:, :, index]  # .astype(np.uint8)
            # print(patient1.max(),patient2.max(),patient3.max(),patient4.max(),patient_seg_label.max())
            patient_name = str(index)
            patient1 = train_transformer3(img0)
            patient2 = train_transformer3(mask0)

            patient3 = train_transformer3(imgt)
            # patient4 = train_transformer3(maskt)

            patient5 = train_transformer3(imgT)

            patient_seg_label = train_transformer3(patient_seg_label)

            patient = torch.cat((patient1, patient2, patient3, patient5), dim=0)  # 获得三通道图像，匹配维度
            # print('patient max',patient.max())
            sample = {'patient': patient, 'seg_label': patient_seg_label, 'patient_name': patient_name}


        return sample

