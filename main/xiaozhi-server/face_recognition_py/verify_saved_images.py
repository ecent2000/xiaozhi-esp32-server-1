#!/usr/bin/env python3
"""
验证保存的图片文件是否有效
"""

import os
from PIL import Image

def verify_image_file(image_path):
    """验证图片文件是否有效"""
    try:
        with Image.open(image_path) as img:
            print(f"✓ {image_path}: {img.format} {img.size} {img.mode}")
            return True
    except Exception as e:
        print(f"✗ {image_path}: 无法打开 - {e}")
        return False

def verify_all_images_in_dataset():
    """验证数据库中的所有图片"""
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"数据库路径不存在: {dataset_path}")
        return
    
    total_images = 0
    valid_images = 0
    
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"\n检查 {person_name} 的图片:")
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_path = os.path.join(person_dir, filename)
                total_images += 1
                if verify_image_file(image_path):
                    valid_images += 1
    
    print(f"\n总结: {valid_images}/{total_images} 图片有效")

if __name__ == "__main__":
    print("=== 验证数据库中的图片文件 ===")
    verify_all_images_in_dataset() 