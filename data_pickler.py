import json
import pickle

import skimage.io as io
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO

data = json.load(
    open('/Users/sudarshinityagi/PycharmProjects/ImageGeneration/code/annotations/instances_train2014.json'))
coco_train = COCO('/Users/sudarshinityagi/PycharmProjects/ImageGeneration/code/annotations/instances_train2014.json')
# initialize COCO api for caption annotations
annFile = '/Users/sudarshinityagi/PycharmProjects/ImageGeneration/code/annotations/captions_train2014.json'
coco_caps = COCO(annFile)

coco_img_ids = coco_train.getImgIds(catIds=18)
images = coco_train.loadImgs(coco_img_ids)
final_data = []
rgb_images64 = []
rgb_images128 = []
captions = []
for i, img in enumerate(images):
    actualImage = io.imread(img['coco_url'])
    if len(actualImage.shape) != 3:
        continue
    else:
        annotations = coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=[img['id']]))
        transformed_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(
            transforms.CenterCrop((224))(transforms.Resize((224))(Image.fromarray(actualImage))))).data.numpy()
        for annotation in annotations:
            final_data.append((transformed_image, annotation['caption']))

    print("Done: %v/%v".format(i, len(images)))

file_name_dogs = "final_data_dogs.pkl"

# open the file for writing
fileObject = open(file_name_dogs, 'wb')

# this writes the object a to the
pickle.dump(final_data, fileObject)

# here we close the fileObject
fileObject.close()
