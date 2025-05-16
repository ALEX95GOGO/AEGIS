import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def rgb_to_grayscale_segmentation(src_path, dst_path, image_name):
    # CITYSCAPES palette in RGB
    CITYSCAPES_PALETTE = np.array([
        [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
        [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
        [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
        [110, 190, 160], [170, 120, 50], [55, 90, 80], [45, 60, 150], [157, 234, 50],
        [81, 0, 81], [150, 100, 100], [230, 150, 140], [180, 165, 180]
    ], dtype=np.uint8)

    # Load the source image
    src_image = cv2.imread(os.path.join(src_path, image_name))
    #print(src_image)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    #src_image = (256-src_image).astype(np.uint8)
    #print(src_image)
    if src_image is None:
        raise FileNotFoundError(f"Source file '{src_path}' not found.")

    # Prepare an empty image for the grayscale segmentation mask
    grayscale_mask = np.zeros((src_image.shape[0], src_image.shape[1]), dtype=np.uint8)

    count = 0
    # Map each palette color to a grayscale value
    for index, color in enumerate(CITYSCAPES_PALETTE):
        # Find pixels matching this color
        matches = np.all(src_image == color, axis=-1)
        # Assign the index value (as grayscale value) to the matching pixels
        grayscale_mask[matches] = index
        count += np.sum(matches)
    #print(np.unique(src_image))
    #print(count, src_image.shape[0]*src_image.shape[1])
    assert count == src_image.shape[0]*src_image.shape[1]
    #print(np.unique(grayscale_mask))
    # Save the resulting grayscale image
    os.makedirs(dst_path, exist_ok=True)
    cv2.imwrite(os.path.join(dst_path, image_name), grayscale_mask)
    #print(f"Grayscale segmentation mask saved to '{dst_path}'.")

def main():
    count = 0
    #root = os.path.join("leftturn")
    #root = os.path.join("car_following")
    #root = os.path.join("leftturn_fixed_action")
    #root = os.path.join("car_following_fixed_action")
    root = os.path.join("leftturn_town5_3car1")
    for mode in ["bc1", "vanilla1", "with_eye_tracking1"]:
        models = sorted(glob(os.path.join(root, mode, "*")))
        for i in range(0, len(models), 2):
            model = models[i].split(os.sep)[-1]
            csv_f = models[i+1].split(os.sep)[-1]
            assert model in csv_f
            img_f = "semantic"
            trials = sorted(glob(os.path.join(root, mode, model, img_f, "*")))      
            for trial in trials:
                if "label" in trial.split(os.sep)[-1]:
                    continue
                src_path = os.path.join(trial)
                dst_path = os.path.join(trial+"_label")
                images_name = glob(os.path.join(src_path, "*.png"))
                assert len(images_name) > 0
                count += 1
                print(src_path)
                for image_name in images_name:
                    image_name = image_name.split(os.sep)[-1]
                    rgb_to_grayscale_segmentation(src_path, dst_path, image_name)
    print(count)

if __name__=="__main__":
    main()
