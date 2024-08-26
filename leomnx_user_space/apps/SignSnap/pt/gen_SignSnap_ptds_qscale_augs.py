import os
import numpy as np
import argparse
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms

#------------------------------------------------------------------------

def process_images(dataset_dir, img_size):
    """ Process images and labels from the given directory. """
    img_paths = []
    labels = []
    print(os.listdir(dataset_dir))
    index = 0
    for dir_name in os.listdir(dataset_dir):
        if dir_name.startswith('.'):
            continue
        
        dir_path = os.path.join(dataset_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        for img_name in os.listdir(dir_path):
            if img_name.startswith('.'):
                continue
            
            img_path = os.path.join(dir_path, img_name)
            img_paths.append(img_path)
            labels.append(index)

        index += 1            
    # print("labels: ", labels)
    return img_paths, labels

def split_data(img_paths, labels, train_split, val_split, test_split):
    """ Split data into training, validation, and test sets. """
    num_imgs = len(img_paths)
    
    indices = np.arange(num_imgs)
    np.random.shuffle(indices)
    
    train_end = int(train_split * num_imgs)
    val_end = train_end + int(val_split * num_imgs)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_paths = [img_paths[i] for i in train_indices]
    val_paths = [img_paths[i] for i in val_indices]
    test_paths = [img_paths[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def create_tensor_dataset(img_paths, labels, img_size, augment=False, augment_factor=1):
    """ 
    Create a TensorDataset from image paths and labels, with optional augmentation.
    augment_factor determines how many augmented copies of each image to generate.
    """
    num_imgs = len(img_paths)
    ds_imgs = np.empty((num_imgs * augment_factor, 1, img_size, img_size), dtype=np.float32)
    ds_lbls = np.empty((num_imgs * augment_factor), int)
    
    # Base transformation
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert image to grayscale
        transforms.Resize((img_size, img_size)),  # Resize image
        transforms.ToTensor(),  # Convert image to tensor
    ])
    
    # Augmentation transformations (only applied if augment is True)
    augmentation_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),  # Rotate by a random angle between -10 and 10 degrees
        transforms.RandomAffine(translate=(0.1, 0.1), degrees=0),  # Random translation
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    ]) if augment else None
    
    for i, img_path in enumerate(img_paths):
        try:
            image = Image.open(img_path)
            for j in range(augment_factor):
                if augment and augmentation_transforms:
                    aug_image = augmentation_transforms(image)
                else:
                    aug_image = image
                aug_image = transform(aug_image)
                
                # Quantize scaling input images from 0 to 255                        
                ds_imgs_np = aug_image.numpy()
                scale_min = np.min(ds_imgs_np)
                scale_max = np.max(ds_imgs_np)
                ds_imgs_np = np.round((ds_imgs_np-scale_min)*(256/(scale_max-scale_min)))  
                ds_imgs_np = np.clip(ds_imgs_np, 0, 255)  # to be safe clamp at 0-255
                # End of quantize scaling
                
                ds_imgs[i * augment_factor + j] = ds_imgs_np  # image.numpy()
                ds_lbls[i * augment_factor + j] = labels[i]
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    
    ds_imgs = np.squeeze(ds_imgs, axis=1)
    pt_tensor_imgs = torch.tensor(ds_imgs)
    pt_tensor_lbls = torch.tensor(ds_lbls, dtype=torch.long)
    
    return TensorDataset(pt_tensor_imgs, pt_tensor_lbls)

def gen_pt_ds(dataset_dir, train_split, val_split, test_split, output_dir, augment_factor=1):
    print('Generating datasets...')
    
    img_size = 40 
    
    img_paths, labels = process_images(dataset_dir, img_size)
    
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_data(img_paths, labels, train_split, val_split, test_split)
    
    # Apply augmentation to training dataset only
    train_dataset = create_tensor_dataset(train_paths, train_labels, img_size, augment=True, augment_factor=augment_factor)
    val_dataset = create_tensor_dataset(val_paths, val_labels, img_size)
    test_dataset = create_tensor_dataset(test_paths, test_labels, img_size)

    label_names = [d for d in os.listdir(dataset_dir) if not d.startswith('.')]
    print(f'Label names: {label_names}')
    
    torch.save(dict(
        pt_ds=train_dataset,
        lnbi=label_names)
    , os.path.join(output_dir, 'sign_lang_train.pt'))
    
    torch.save(dict(
        pt_ds=val_dataset,
        lnbi=label_names)
    , os.path.join(output_dir, 'sign_lang_val.pt'))

    torch.save(dict(
        pt_ds=test_dataset,
        lnbi=label_names)
    , os.path.join(output_dir, 'sign_lang_test.pt'))
    
    print(f'Captured total {len(img_paths)} images')
    print(f'Split into {len(train_paths) * augment_factor} training, {len(val_paths)} validation, and {len(test_paths)} test images')
    print(f'Generated {len(train_paths) * augment_factor} images for training with augmentation factor {augment_factor}')

#------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generate PyTorch dataset from sign language images')   
    ap.add_argument('-d', metavar='<dataset_directory>', type=str, default="../data", help='Path to the root directory of the dataset')
    ap.add_argument('-o', metavar='<output_directory>', type=str, default="workspace", help='Directory where the output files will be saved')
    ap.add_argument('--train_split', type=float, default=0.7, help='Proportion of the data to use for training (default=0.7)')
    ap.add_argument('--val_split', type=float, default=0.15, help='Proportion of the data to use for validation (default=0.15)')
    ap.add_argument('--test_split', type=float, default=0.15, help='Proportion of the data to use for testing (default=0.15)')
    ap.add_argument('--augment_factor', type=int, default=1, help='Number of augmented versions of each image to generate (default=1)')
    args = ap.parse_args()
    
    dataset_dir = args.d
    output_dir = args.o
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split
    augment_factor = args.augment_factor
    
    assert train_split + val_split + test_split == 1.0, "The splits must sum to 1.0"
    
    print(f'Reading sign language dataset from {dataset_dir}')
    
    gen_pt_ds(dataset_dir, train_split, val_split, test_split, output_dir, augment_factor)
