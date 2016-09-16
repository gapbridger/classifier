dataset1 = range(1000)
dataset2 = range(1000, 2000)
nb_picle = 2

nb_sample = len(dataset1)+len(dataset2)
batch_size = 128
nb_batch = nb_sample/batch_size    #20

def ImageNet():
    while 1:
        for i in range(nb_batch):
            if(i<10):
                a = dataset1
                yield a[(i)*batch_size:(i + 1)*batch_size]
            else:
                a = dataset2
                yield a[(i-10)*batch_size:(i+1-10)*batch_size]

epoch_size = 100
# for i in range(epoch_size):
for ele in ImageNet():
    print(ele)
