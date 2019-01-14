import pickle

img_list = pickle.load(open('img_batch.p'))
train_list = []
val_list = []
amount_in_val = 5

train_list = [p for e,p in enumerate(img_list) if e % amount_in_val != 0]
val_list = [p for e,p in enumerate(img_list) if e % amount_in_val == 0]

pickle.dump(train_list, open('train_list/img_batch.p','w'))
pickle.dump(val_list, open('train_list/img_batch.p', 'w'))
