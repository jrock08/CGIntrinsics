import pickle

img_list = pickle.load(open('img_batch.p'))

train_list = []
val_list = []
amount_in_val = 5
for i in range(len(img_list)):
    train_list.append([p for e,p in enumerate(img_list[i]) if e % amount_in_val != 0])
    val_list.append([p for e,p in enumerate(img_list[i]) if e % amount_in_val == 0])

pickle.dump(train_list, open('train_list/img_batch.p','w'))
pickle.dump(val_list, open('train_list/img_batch.p', 'w'))


