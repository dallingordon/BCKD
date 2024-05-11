import os
from PIL import Image
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('imagenet-1k')

# Base directory for dataset
base_dir = '/projectnb/textconv/distill/mdistiller/data/imagenet_fixed/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Function to save images in specified directory based on split
def save_images(data, directory):
    for index, example in enumerate(data):
        image = example['image']  # The image is already a Pillow Image object
        label = str(example['label'])
        #print(label, index)
        #break
        label_dir = os.path.join(directory, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        # Using index to create a unique file name
        image_path = os.path.join(label_dir, f"{index}.jpg")
        image.save(image_path)


# Create directories and save images
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#save_images(dataset['train'], train_dir) #do test first.
save_images(dataset['validation'], test_dir)
