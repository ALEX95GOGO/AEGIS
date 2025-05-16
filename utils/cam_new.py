import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet18
from torchcam.methods import CAM, ISCAM, ScoreCAM, SSCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image


model = resnet18(pretrained=True).eval()
model.cuda()
cam = CAM(model, 'layer4')#, 'fc')
#path = "car_following/with_eye_tracking1/actor1706009411/rgb/success_2/001488.png"
path = "/projects/CIBCIGroup/00DataUploading/ChengYou/eccv/leftturn/with_eye_tracking1/actor1707442492/rgb/fail_1/0140.png"
rgb_img = Image.open(path)
rgb_img = rgb_img.resize((300, 300))
rgb_img = np.array(rgb_img, dtype=np.float32)/255.
#rgb_image = normalize(rgb_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
h, w, _ = np.array(rgb_img).shape
print(h, w)
# [batch, 3, h, w]
input_tensor = torch.tensor([rgb_img]).permute(0, 3, 1, 2)
input_tensor = input_tensor.cuda()
input_tensor2 = input_tensor #normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

for i in range(3):
    start = time.time()
    with torch.no_grad(): 
        out = model(input_tensor2)
        cam_out = cam(class_idx=817)[0]
    end = time.time()
    print(end-start)
    r = overlay_mask(to_pil_image(input_tensor[0]), to_pil_image(cam_out[0], mode='F'), alpha=0.5)
    plt.imshow(r)
    plt.show()
 
