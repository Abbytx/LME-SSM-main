import  numpy as np

import pickle


data1_path='C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\index\\amci_namci_5fold_train_1_data.npy'
data1=np.load(data1_path)
data2_path='C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\index\\amci_namci_5fold_test_1_data.npy'
data2=np.load(data2_path)

data=np.concatenate((data1,data2))
#data=data1
print(data.shape)
num_NC=0
num_MCI=0

data1_path='C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\index\\amci_namci_5fold_train_1_label.pkl'
with open(data1_path, 'rb') as f:
    _,lable1=pickle.load(f)


data2_path='C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\index\\amci_namci_5fold_test_1_label.pkl'
with open(data2_path, 'rb') as f:
    _,lable2=pickle.load(f)

lable=lable1+lable2
#=np.array(lable)


for i in range(len(lable)):
    #i=i+100
    temp=data[i,0,:,:,0]
    print(temp.shape)
    #np.savetxt('sub_01.txt', xx)
    np.savetxt('C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\amci_namci\\amcinamci_{}.txt'.format((i+1)),temp)
    # if lable[i]==0:
    #     np.savetxt('C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\adni2_k1_check\\sub_NC_0{}.txt'.format(num_NC),temp)
    #     num_NC=num_NC+1
    # elif lable[i]==1:
    #     np.savetxt('C:\\Users\\Administrator\\Desktop\\Brainnetclass_data\\adni2_k1_check\\sub_MCI_0{}.txt'.format(num_MCI),temp)
    #     num_MCI=num_MCI+1

# data1_path='D:\\Download\\ADNI2\\adni2_5split_train_1_label.pkl'
# with open(data1_path, 'rb') as f:
#     _,lable1=pickle.load(f)
#
#
# data2_path='D:\\Download\\ADNI2\\adni2_5split_test_1_label.pkl'
# with open(data2_path, 'rb') as f:
#     _,lable2=pickle.load(f)
#
# lable=lable1+lable2
# lable=np.array(lable)
# lable[lable==0]=-1
# print(lable.shape)
#
# dd=np.zeros((410,1))
# dd[:,0]=lable
# print(dd)
# np.savetxt('adni2_lable.txt', dd,fmt='%d')  #424  410

#真tm服了
