import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }

def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            filename, label_str = line.strip().replace("'", "").replace('"', '').split(': ')
            
            label = int(label_str.strip().split(',')[-1].strip())
            
            labels[filename.strip()] = label
    return labels

def load_dataset(data_dir,label_file,device,is_train=True):
    image_labels = load_labels(label_file)
    
    images = []
    labels = []
    start_image_number = None

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if is_train:
                image = data_transforms['train'](image)  # Apply training transformations
            else:
                image = data_transforms['test'](image)  # Apply testing transformations
            images.append(image)
            labels.append(label)

            if start_image_number is None:
                start_image_number = int(filename.split('_')[-1].split('.')[0])

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1  # Use larger batch size for training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Loaded {len(images)} images.')

    return dataset, data_loader, start_image_number

def stuff_bits(binary_string):
    """
    Inserting '1' after every 5 consecutive '0's in the binary string.
    Args:
        binary_string (str): Binary string to be stuffed.
    Returns:
        str: Binary string after stuffing.
    """
    return binary_string

def print_image(img,n,pack):
    img = img.detach()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy format
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)  # Just in case

    plt.imshow(img, interpolation='none')  
    if n == 1:
        plt.title(f"Mask, Injection {pack})")
    elif n == 2:
        plt.title(f"Perturbed image, Injection{pack}")
    plt.axis('off')
    plt.show()

def print_bits_from_image(image,mask):
    for b in range(image.shape[0]):  # Iterate over batch dimension
        row = mask[b, 0].nonzero(as_tuple=True)[0]  
        if len(row) > 0:  # Check if any row was identified
            row = row[0].item()  # Get the first row index
            
            bits = image[b, :, row, :].flatten()  # Flatten the specific row across all channels
            
            binary_representation = ''.join(['1' if bit > 0.5 else '0' for bit in bits])
            print("length of binary representation:",len(binary_representation))
            print(f"Perturbed bits for batch {b}, row {row}: {binary_representation}")

def saving_image(img, name,output_path):
    os.makedirs(output_path, exist_ok=True)
    
    output_path = os.path.join(output_path, f'perturbed_image_{name}.png')
    
    save_image(img, output_path)

def calculate_crc(data):
    """
    Calculate CRC-15 checksum for the given data.
    """
    crc = 0x0000
    poly = 0x4599

    for bit in data:
        crc ^= (int(bit) & 0x01) << 14

        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1

        crc &= 0x7FFF
    return crc

def crc_remainder(input_bitstring, polynomial_bitstring, initial_filler):
    polynomial_bitstring = polynomial_bitstring.lstrip('0')
    len_input = len(input_bitstring)
    print("len_input",len_input)
    initial_padding = initial_filler * (len(polynomial_bitstring) - 1)
    input_padded_array = list(input_bitstring + initial_padding)
    
    while '1' in input_padded_array[:len_input]:
        cur_shift = input_padded_array.index('1')
        for i in range(len(polynomial_bitstring)):
            input_padded_array[cur_shift + i] = \
                str(int(polynomial_bitstring[i] != input_padded_array[cur_shift + i]))
                
    return ''.join(input_padded_array)[len_input:]

