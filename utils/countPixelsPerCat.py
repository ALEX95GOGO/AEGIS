import cv2
import numpy as np
import os
import csv
from glob import glob


def load_images_and_process(segmentation_mask_path, attention_map_path, image_name):
    # CITYSCAPES category names
    CATEGORY_NAMES = [
        "unlabeled", "road", "sidewalk", "building", "wall",
        "fence", "pole", "traffic light", "traffic sign", "vegetation",
        "terrain", "sky", "pedestrian", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
        "static", "dynamic", "other", "water", "road line",
        "ground", "bridge", "rail track", "guard rail"
    ]

    # Load the segmentation mask as a grayscale image
    segmentation_mask_path = os.path.join(segmentation_mask_path, image_name)
    segmentation_mask = cv2.imread(segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
    if segmentation_mask is None:
        raise FileNotFoundError(f"Segmentation mask file '{segmentation_mask_path}' not found.")
    
    # Load the attention map as a grayscale image
    attention_map_path = os.path.join(attention_map_path, image_name)
    attention_map = cv2.imread(attention_map_path, cv2.IMREAD_GRAYSCALE)
    if attention_map is None:
        raise FileNotFoundError(f"Attention map file '{attention_map_path}' not found.")
    
    h, w = attention_map.shape
    #print(np.unique(segmentation_mask))
    segmentation_mask = cv2.resize(segmentation_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    #print(np.unique(segmentation_mask))

    # Normalize the attention map to range [0, 1]
    attention_map = attention_map / 255.0
    
    # Convert the attention map to a binary map
    binary_map = (attention_map > 0.1).astype(np.uint8)
    
    # Count the number of '1's per category in the segmentation mask
    category_counts = {}
    count_pix = 0
    for category in range(len(CATEGORY_NAMES)):  # Use length of CATEGORY_NAMES for iteration
        # Create a mask for the current category
        category_mask = segmentation_mask == category
        
        # Count the '1's in the binary map where the category mask is True
        count = np.sum(binary_map[category_mask])
        category_counts[category] = count/np.sum(binary_map)
        count_pix += count
    assert count_pix == np.sum(binary_map)

    # Print the results
    #total = 0
    #for category, count in category_counts.items():
    #    total += count
    #    print(f"Category {category} ({CATEGORY_NAMES[category]}): {count} pixels with attention")
    #print(total)
    return category_counts


def append_to_csv(mean_cat_ratio, model, status, filename='output.csv', new=False):
    CATEGORY_NAMES = [
        "unlabeled", "road", "sidewalk", "building", "wall",
        "fence", "pole", "traffic light", "traffic sign", "vegetation",
        "terrain", "sky", "pedestrian", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
        "static", "dynamic", "other", "water", "road line",
        "ground", "bridge", "rail track", "guard rail"
    ]
    # Define the header with the specified keys
    headers = [i for i in range(len(CATEGORY_NAMES))] #[1, 2, 5, 9, 11, 14, 26, 27, 28]
    cat_headers = [CATEGORY_NAMES[h] for h in headers]
    cat_headers = [None, None] + cat_headers

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        
        # If the file does not exist, write the header first
        if not file_exists:
            writer.writerow(cat_headers)
        
        # Write the values corresponding to the headers in the mean_cat_ratio dictionary
        # If a key is missing in the dictionary, write 'None' or a default value
        row = [mean_cat_ratio.get(key, None) for key in headers]
        row = [model, status] + row
        writer.writerow(row)
        if new:
            writer.writerow([])
            writer.writerow([])
            writer.writerow([])
            writer.writerow([])
    
    #print(f"Data appended to {filename}.")

def main():
    CATEGORY_NAMES = [
        "unlabeled", "road", "sidewalk", "building", "wall",
        "fence", "pole", "traffic light", "traffic sign", "vegetation",
        "terrain", "sky", "pedestrian", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
        "static", "dynamic", "other", "water", "road line",
        "ground", "bridge", "rail track", "guard rail"
    ]
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
            img_f = "semantic"
            trials = sorted(glob(os.path.join(root, mode, model, img_f, "*_label")))      
            success_mean_cat_ratio = {}
            fail_mean_cat_ratio = {}
            num_trials = len(trials)
            num_suc = 0
            num_fail = 0
            for trial in trials:
                seg_path = os.path.join(trial)
                att_path = os.path.join(root, mode, model, "machine_att", 
                        trial.split(os.sep)[-1][:-6])
                images_name = sorted(glob(os.path.join(seg_path, "*.png")))[4:]
                assert len(images_name) > 0
                count += 1
                #print(seg_path)
                mean_cat_ratio = {}
                for image_name in images_name:
                    num_imgs = len(images_name)
                    image_name = image_name.split(os.sep)[-1]
                    cat_ratio = load_images_and_process(seg_path, att_path, image_name)
                    # get ratio per image
                    for category, count in cat_ratio.items():
                        if category not in mean_cat_ratio:
                            mean_cat_ratio[category] = cat_ratio[category]/num_imgs
                        else:
                            mean_cat_ratio[category] += cat_ratio[category]/num_imgs
                if "success" in trial:
                    num_suc += 1
                else:
                    num_fail += 1
                for category, count in mean_cat_ratio.items():
                    if "success" in trial:
                        if category not in success_mean_cat_ratio:
                            success_mean_cat_ratio[category] = mean_cat_ratio[category]
                        else:
                            success_mean_cat_ratio[category] += mean_cat_ratio[category]
                    else:
                        if category not in fail_mean_cat_ratio:
                            fail_mean_cat_ratio[category] = mean_cat_ratio[category]
                        else:
                            fail_mean_cat_ratio[category] += mean_cat_ratio[category]
            
            for category, count in success_mean_cat_ratio.items():
                success_mean_cat_ratio[category]/=num_suc
            for category, count in fail_mean_cat_ratio.items():
                fail_mean_cat_ratio[category]/=num_fail

            temp = []
            for category, count in success_mean_cat_ratio.items():
                if category not in [1, 2, 5, 9, 11, 14, 26, 27, 28]:
                    continue
                temp.append([count, CATEGORY_NAMES[category]])
                #print(f"Category {category} ({CATEGORY_NAMES[category]}): {count}")
            print("Success", temp, num_suc)
            """
            suc = sorted(temp[::-1][:3])
            print("Success", suc, num_suc)
            for t in suc:
                cat.add(t[1])
            """
            for category, count in fail_mean_cat_ratio.items():
                if category not in [1, 2, 5, 9, 11, 14, 26, 27, 28]:
                    continue
                temp.append([count, CATEGORY_NAMES[category]])
                #print(f"Category {category} ({CATEGORY_NAMES[category]}): {count}")
            print("Fail", temp, num_fail)
            """
            fail = sorted(temp)[::-1][:3]
            print("Fail", fail, num_fail)
            for t in fail:
                cat.add(t[1])
            """
            append_to_csv(success_mean_cat_ratio, model, "success", "ratio.csv", False)
            append_to_csv(fail_mean_cat_ratio, model, "fail", "ratio.csv", True)
            print()

if __name__=="__main__":
    main()
