import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import Tuple, List

def generate_synthetic_images(output_dir: str, num_images: int = 500, image_size: int = 256):
    os.makedirs(output_dir, exist_ok=True)
    
    existing = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    if existing >= num_images:
        return
    
    for i in range(existing, num_images):
        if i % 100 == 0:
            print(f"Generating synthetic image {i+1}/{num_images}")
        
        img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f"image_{i:05d}.png"))

class ImagePairDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        image_size: int = 256,
        split_type: str = "train"
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        self.split_type = split_type
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cover_idx = idx
        secret_idx = random.randint(0, len(self.image_paths) - 1)
        
        while secret_idx == cover_idx and len(self.image_paths) > 1:
            secret_idx = random.randint(0, len(self.image_paths) - 1)
        
        try:
            cover_path = self.image_paths[cover_idx]
            secret_path = self.image_paths[secret_idx]
            
            cover_image = Image.open(cover_path).convert('RGB')
            secret_image = Image.open(secret_path).convert('RGB')
            
            cover_tensor = self.transform(cover_image)
            secret_tensor = self.transform(secret_image)
            
            return cover_tensor, secret_tensor
        except:
            return self.__getitem__((idx + 1) % len(self.image_paths))

def load_images_from_folder(folder_path: str) -> List[str]:
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    if not os.path.exists(folder_path):
        return []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)

def create_dataloaders(
    config,
    data_folder: str = "data/tiny_imagenet"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    if config.generate_synthetic_dataset:
        total_needed = config.subset_size['train'] + config.subset_size['val'] + config.subset_size['test']
        generate_synthetic_images(config.synthetic_dataset_path, total_needed, config.image_size)
        data_folder = config.synthetic_dataset_path
    
    image_paths = load_images_from_folder(data_folder)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_folder}. Create {data_folder} or enable synthetic dataset generation.")
    
    random.seed(config.seed)
    random.shuffle(image_paths)
    
    train_count = min(config.subset_size['train'], int(len(image_paths) * config.dataset_split['train']))
    val_count = min(config.subset_size['val'], int(len(image_paths) * config.dataset_split['val']))
    test_count = min(config.subset_size['test'], int(len(image_paths) * config.dataset_split['test']))
    
    train_paths = image_paths[:train_count]
    val_paths = image_paths[train_count:train_count + val_count]
    test_paths = image_paths[train_count + val_count:train_count + val_count + test_count]
    
    print(f"\n{'='*60}")
    print(f"Dataset Composition:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    print(f"  Test:  {len(test_paths)} images")
    print(f"  Total: {len(train_paths) + len(val_paths) + len(test_paths)} images")
    print(f"{'='*60}\n")
    
    train_dataset = ImagePairDataset(
        train_paths,
        image_size=config.image_size,
        split_type="train"
    )
    
    val_dataset = ImagePairDataset(
        val_paths,
        image_size=config.image_size,
        split_type="val"
    )
    
    test_dataset = ImagePairDataset(
        test_paths,
        image_size=config.image_size,
        split_type="test"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader