import os
import pickle

img_list = pickle.load(open('CGIntrinsics/intrinsics_final/train_list/img_batch.p'))

directory = 'CGIntrinsics/intrinsics_final/train_val_list/train_list'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = 'CGIntrinsics/intrinsics_final/train_val_list/val_list'
if not os.path.exists(directory):
    os.makedirs(directory)

train_list = []
val_list = []
amount_in_val = 25
train_list = [p for e,p in enumerate(img_list) if e % amount_in_val != 0]
val_list = [p for e,p in enumerate(img_list) if e % amount_in_val == 0]

pickle.dump(train_list, open('CGIntrinsics/intrinsics_final/train_val_list/train_list/img_batch.p','w'))
pickle.dump(val_list, open('CGIntrinsics/intrinsics_final/train_val_list/val_list/img_batch.p', 'w'))
