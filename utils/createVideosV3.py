import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import numpy as np
from PIL import Image

# Folder paths
folder1 = "car_following/with_eye_tracking1/actor1707297182/rgb/success_1_rgb_machine_heat/"
folder2 = "car_following/vanilla1/actor1707294052/rgb/success_0_rgb_machine_heat/"
folder3 = "car_following/bc1/actor1708008767/rgb/success_0_rgb_machine_heat/" 
folder4 = "leftturn/with_eye_tracking1/actor1707879320/rgb/success_0_rgb_machine_heat/" 
folder5 = "leftturn/vanilla1/actor1707390422/rgb/success_9_rgb_machine_heat/"
folder6 = "leftturn/bc1/actor1708144103/rgb/success_0_rgb_machine_heat/" 
folders = [folder1, folder2, folder3, folder4, folder5, folder6]


# Prepare video writer
video_filename = './free_actions.mp4'
writer = imageio.get_writer(video_filename, fps=12)

# Find the maximum number of images in any folder
max_images = max(len(os.listdir(folder)) for folder in folders)

# Create video frames
for i in range(max_images):
    # Set up the figure and axes
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0, hspace=0)
    
    # Add images to each subplot
    for idx, folder in enumerate(folders):
        ax = fig.add_subplot(gs[idx])
        # Get image files and ensure the index is within bounds
        image_files = sorted(os.listdir(folder))[5:]
        if 'leftturn' in image_files[0]:
            new_i = i//2
            if i%2==1:
                image_file = image_files[new_i % len(image_files)]
            else:
                image_file = image_files[new_i % len(image_files)]
        else:    
            image_file = image_files[i % len(image_files)]
        #image_file = image_files[i % len(image_files)]
        image_path = os.path.join(folder, image_file)
        # Read and display the image
        img = Image.open(image_path)
        
        img = img.resize((256, 112))

        ax.imshow(img, aspect='auto')
        ax.axis('off')  # Hide axis
        #ax.set_aspect(aspect='equal')

    # Add text descriptions
    fig.text(0.03, 0.7, 'Car Following', va='center', rotation='vertical')
    fig.text(0.03, 0.3, 'Left Turn', va='center', rotation='vertical')
    # For column text, adjust the x-position accordingly
    fig.text(0.2, 0.9, 'HAG-RL (Ours)', ha='center')
    fig.text(0.5, 0.9, 'Vanilla', ha='center')
    fig.text(0.8, 0.9, 'BC', ha='center')

    plt.tight_layout(pad=5)
    #plt.show()
    #exit()
    # Convert the plot to an image
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Append frame to video
    writer.append_data(frame)
    
    plt.close(fig)

# Finalize the video
writer.close()

print('Video created successfully.')

