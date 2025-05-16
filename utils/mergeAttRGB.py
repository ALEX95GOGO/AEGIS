import cv2
import os
import numpy as np
from glob import glob

def blend_images(src_rgb_path, src_att_path, dst_path, image_name, alpha=0.5):
    # Load the RGB image
    image_path = os.path.join(src_rgb_path, image_name)
    rgb_image = cv2.imread(image_path)

    # Load the heatmap (assuming it's already in RGB format)
    image_path = os.path.join(src_att_path, image_name)
    heatmap = cv2.imread(image_path)

    # Resize heatmap to match the rgb_image size
    heatmap_resized = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    #print(heatmap_resized.shape, heatmap.shape, rgb_image.shape)
    # Blend the images
    blended_image = cv2.addWeighted(rgb_image, alpha, heatmap_resized, 1 - alpha, 0)

    # Display the blended image
    #cv2.imshow('Blended Image', blended_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Optionally, save the blended image to a file
    image_path = os.path.join(dst_path, image_name)
    os.makedirs(dst_path, exist_ok=True)
    cv2.imwrite(image_path, blended_image)

    return blended_image

def main():
    count = 0
    #root = os.path.join("leftturn")
    #root = os.path.join("car_following")
    #root = os.path.join("leftturn_fixed_action")
    #root = os.path.join("car_following_fixed_action")
    #root = os.path.join("leftturn_town5_3car1")
    root = os.path.join("leftturn_town1")
    for mode in ["noeye"]:#["bc1", "vanilla1", "with_eye_tracking1"]:
        models = sorted(glob(os.path.join(root, mode, "*")))
        for i in range(0, len(models), 2):
            model = models[i].split(os.sep)[-1]
            csv_f = models[i+1].split(os.sep)[-1]
            assert model in csv_f
            img_f = "rgb"
            trials = sorted(glob(os.path.join(root, mode, model, img_f, "*")))
            for trial in trials:
                if "rgb" in trial.split(os.sep)[-1]:
                    continue
                src_rgb_path = os.path.join(trial)
                #src_att_path = os.path.join(root, mode, model, "human_att", 
                #                            trial.split(os.sep)[-1]+"_heat")
                src_att_path = os.path.join(root, mode, model, "machine_att", 
                                            trial.split(os.sep)[-1]+"_heat")
                #dst_path = os.path.join(trial+"_rgb_human_heat")
                dst_path = os.path.join(trial+"_rgb_machine_heat")
                images_name = glob(os.path.join(src_rgb_path, "*.png"))
                assert len(images_name) > 0
                count += 1
                print(src_rgb_path)
                #print(src_att_path)
                #print(dst_path)
                for image_name in images_name:
                    image_name = image_name.split(os.sep)[-1]
                    alpha = 0.5  # Adjust the alpha value to change the blending ratio
                    blended_image = blend_images(src_rgb_path, 
                                                 src_att_path, 
                                                 dst_path, 
                                                 image_name, 
                                                 alpha)
    print(count)

if __name__=="__main__":
    main()

