import os
import pickle

img_list = pickle.load(open('CGIntrinsics/IIW/train_list/img_batch.p'))

directory = 'CGIntrinsics/IIW/train_val_list/train_list'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = 'CGIntrinsics/IIW/train_val_list/val_list'
if not os.path.exists(directory):
    os.makedirs(directory)

train_list = []
val_list = []
amount_in_val = 12
for i in range(len(img_list)):
    train_list.append([p for e,p in enumerate(img_list[i]) if e % amount_in_val != 0])
    val_list.append([p for e,p in enumerate(img_list[i]) if e % amount_in_val == 0])

pickle.dump(train_list, open('CGIntrinsics/IIW/train_val_list/train_list/img_batch.p','w'))
pickle.dump(val_list, open('CGIntrinsics/IIW/train_val_list/val_list/img_batch.p', 'w'))


