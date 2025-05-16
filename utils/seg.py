from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os

def get_seg(image, model, image_processor, h, w):

    # CITYSCAPES_PALETTE_MAP as provided
    CITYSCAPES_PALETTE_MAP = np.array([
        [0, 0, 0],        # unlabeled
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # pedestrian
        [255, 0, 0],      # rider
        [0, 0, 142],      # Car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
        [110, 190, 160],  # static
        [170, 120, 50],   # dynamic
        [55, 90, 80],     # other
        [45, 60, 150],    # water
        [157, 234, 50],   # road line
        [81, 0, 81],      # ground
        [150, 100, 100],  # bridge
        [230, 150, 140],  # rail track
        [180, 165, 180]   # guard rail
    ], dtype=np.uint8)


    # [1, 3, 512, 512]
    # [batch, channel, h, w]
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    predicted_label = logits.argmax(1)
    rescale = torchvision.transforms.Resize((h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    predicted_label = rescale(predicted_label)+1
    predicted_label = CITYSCAPES_PALETTE_MAP[predicted_label]
    return predicted_label


def main():
    # model
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    
    # image
    # (h, w, 3), 0-255
    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
   

    path = "leftturn_town5_3car1/with_eye_tracking1/actor1707880391/rgb/0/0011.png"
    image = Image.open(path)
    h, w, _ = np.array(image).shape

    path_seg = "leftturn_town5_3car1/with_eye_tracking1/actor1707880391/semantic/0/0011.png"
    image_seg = Image.open(path_seg)
    
    # query
    # [batch, h, w, 3]
    pred = get_seg(image, model, image_processor, h, w)
    plt.subplot(1, 2, 1)
    plt.imshow(image_seg)
    plt.subplot(1, 2, 2)
    plt.imshow(pred[0])
    plt.show()

if __name__=="__main__":
    main()
