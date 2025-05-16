import cv2
import numpy as np
import os
from glob import glob

def calculate_entropy(values):
    """Calculate the entropy of an array of values."""
    # Normalize the values to a probability distribution
    probabilities = values / np.sum(values)
    # Filter out zero probabilities and compute entropy
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    # normalize
    if len(probabilities)==1:
        entropy = 0
    else:
        entropy /= np.log2(len(probabilities))
    return entropy

def process_attention_map(src_path, img_name):
    # Load the grayscale image
    image_path = os.path.join(src_path, img_name)
    attention_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if attention_map is None:
        raise FileNotFoundError(f"File '{image_path}' not found.")

    # Split the image into a 4x4 grid and compute the average attention for each grid
    h, w = attention_map.shape
    grid_height, grid_width = h // 4, w // 4
    average_attentions = []

    for i in range(4):
        for j in range(4):
            grid = attention_map[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
            # to binary
            grid = (grid>25).astype(int)
            average_attention = np.mean(grid)
            average_attentions.append(average_attention)

    # Compute the entropy across these 16 average attention values
    entropy = calculate_entropy(np.array(average_attentions))

    return average_attentions, entropy

def main():
    count = 0
    #root = os.path.join("leftturn")
    #root = os.path.join("car_following")
    #root = os.path.join("leftturn_fixed_action")
    #root = os.path.join("car_following_fixed_action")
    root = os.path.join("leftturn_town5_3car1")
    cat = set()
    for mode in ["bc1", "vanilla1", "with_eye_tracking1"]:
        models = sorted(glob(os.path.join(root, mode, "*")))
        for i in range(0, len(models), 2):
            print(models[i])
            model = models[i].split(os.sep)[-1]
            csv_f = models[i+1].split(os.sep)[-1]
            assert model in csv_f
            img_f = "machine_att"
            trials = sorted(glob(os.path.join(root, mode, model, img_f, "*")))      
            num_trials = len(trials)
            num_suc = 0
            num_fail = 0
            suc_trial_entropy = []
            fail_trial_entropy = []
            for trial in trials:
                if "heat" in trial:
                    continue
                src_path = os.path.join(trial)
                images_name = sorted(glob(os.path.join(src_path, "*.png")))[4:]
                assert len(images_name) > 0
                count += 1
                #print(src_path)
                suc_imgs_entropy = 0
                fail_imgs_entropy = 0
                for image_name in images_name:
                    #print(image_name)
                    num_imgs = len(images_name)
                    image_name = image_name.split(os.sep)[-1]
                    _, entropy = process_attention_map(src_path, image_name)
                    #print("Entropy:", entropy)
                    if "success" in trial:
                        suc_imgs_entropy += entropy/num_imgs
                    else:
                        fail_imgs_entropy += entropy/num_imgs
                if "success" in trial:
                    suc_trial_entropy.append(suc_imgs_entropy)
                    num_suc += 1
                else:
                    fail_trial_entropy.append(fail_imgs_entropy)
                    num_fail += 1
            print("Success", np.mean(suc_trial_entropy) if num_suc>0 else "None", num_suc)
            print("Fail", np.mean(fail_trial_entropy), num_fail)
        print()
        print()

if __name__=="__main__":
    main()
