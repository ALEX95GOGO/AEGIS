import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def convert_grayscale_to_heatmap(src_path, dst_path, image_name):
    image_path = os.path.join(src_path, image_name)
    # Step 1: Read the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 2: Normalize the image to range 0-1
    normalized_image = cv2.normalize(grayscale_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Step 3: Apply a colormap (there are many choices, 'jet' is a popular one)
    heatmap_image = plt.cm.jet(normalized_image)
    
    # Step 4a: Display the heatmap
    #plt.imshow(heatmap_image)
    #plt.axis('off')  # Remove axes
    #plt.show()
    
    os.makedirs(dst_path, exist_ok=True)

    image_path = os.path.join(dst_path, image_name)
    # Step 4b: Alternatively, save the heatmap to a file
    plt.imsave(image_path, heatmap_image)

def main():
    count = 0
    #root = os.path.join("leftturn")
    #root = os.path.join("car_following")
    #root = os.path.join("leftturn_fixed_action")
    #root = os.path.join("car_following_fixed_action")
    #root = os.path.join("leftturn_town5_3car1")
    #root = os.path.join("car_following_town7_")
    root = os.path.join("leftturn_town1")
    for mode in ["noeye"]: #["bc1", "vanilla1", "with_eye_tracking1"]:
        models = sorted(glob(os.path.join(root, mode, "*")))
        for i in range(0, len(models), 2):
            model = models[i].split(os.sep)[-1]
            csv_f = models[i+1].split(os.sep)[-1]
            assert model in csv_f
            img_f = "machine_att" 
            #img_f = "human_att"
            trials = sorted(glob(os.path.join(root, mode, model, img_f, "*")))      
            for trial in trials:
                src_path = os.path.join(trial)
                print(src_path)
                dst_path = os.path.join(trial+"_heat")
                images_name = glob(os.path.join(src_path, "*.png"))
                assert len(images_name) > 0
                count += 1
                print(src_path)
                for image_name in images_name:
                    image_name = image_name.split(os.sep)[-1]
                    convert_grayscale_to_heatmap(src_path, dst_path, image_name)
    print(count)


if __name__=="__main__":
    main()

