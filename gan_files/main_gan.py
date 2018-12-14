import math
from char_embeddings import *
from gan_files.GAN import *
import pickle

import sys


def create_batches(batch_size, images, captions):
    num_batches = math.ceil(len(images) / batch_size)
    batches_x = np.array_split(images, num_batches)
    batches_y = np.array_split(captions, num_batches)
    return batches_x, batches_y

# def norm(img):
#     img = cv2.resize(img, (64,64))
#     norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     return norm_image.reshape(3, 64, 64)

# data = json.load(open('/data/ra2630/SNLP/data/annotations/instances_train2014.json'))
# coco_train = COCO('/data/ra2630/SNLP/data/annotations/instances_train2014.json')
# # initialize COCO api for caption annotations
# annFile = '/data/ra2630/SNLP/data/annotations/captions_train2014.json'
# coco_caps=COCO(annFile)

# coco_img_ids = coco_train.getImgIds(catIds=18)
# images = coco_train.loadImgs(coco_img_ids)
# rgb_images = []
# captions = []
# for img in images:
#     actualImage = io.imread(img['coco_url'])
#     if len(actualImage.shape) != 3:
#         continue
#     else:
#         annotation = coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=[img['id']]))[0]
#         rgb_images.append(norm(actualImage))
#         captions.append(annotation['caption'])


fileObject = open("/Users/sudarshinityagi/PycharmProjects/GAN/GAN_data/images_dogs64.pkl",'rb')
# load the object from the file into var b
rgb_images = pickle.load(fileObject) 

fileObject2 = open("/Users/sudarshinityagi/PycharmProjects/GAN/GAN_data/captions_dogs.pkl", 'rb')

captions =  pickle.load(fileObject2) 


batched_images, batched_captions = create_batches(32, rgb_images, captions)


embedding = CharEmbeddings()
trainer = GANTrainer({
    "embedding":embedding.embedding,
    "GPU" : True

}, {}, {})


for epoch in range(1, 201):
    for batch_num, batch_images in enumerate(batched_images):
        batch_len = batch_images.__len__()
        batch_captions = batched_captions[batch_num]
        if(batch_len!=32):
            continue
        imagesActual = np.array(batch_images, dtype=np.float32)


        true_caption_tokens = np.array(embedding.convert_chars_to_tokens(batch_captions, 100), dtype=np.long).reshape(
            batch_len, 102)

        temp = true_caption_tokens.copy()
        wrong_caption_tokens = temp[1:, ]

        wrong_caption_tokens =  np.concatenate((wrong_caption_tokens, temp[0:1,]), axis = 0)


        trainer.train(imagesActual, true_caption_tokens, wrong_caption_tokens, batch_num, epoch)
        sys.stdout.flush()

    trainer.save(filepath="/data/ra2630/SNLP/models/GAN/stack-gan-1_" + str(epoch))






