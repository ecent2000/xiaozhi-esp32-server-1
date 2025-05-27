#!/usr/bin/env python3
"""
测试人脸识别功能
"""

import os
import base64
import requests
import json

def test_face_recognition():
    """测试人脸识别"""
    
    # 使用同一张图片进行识别测试
    test_image_path = "test_images/test_image_01.jpg"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return False
    
    try:
        url = "http://localhost:8001/identify_face/"
        
        # 直接发送文件
        with open(test_image_path, 'rb') as img_file:
            files = {'image': ('test_image.jpg', img_file, 'image/jpeg')}
            data = {
                'model_name': 'VGG-Face',
                'distance_metric': 'cosine',
                'enforce_detection': True
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 人脸识别测试成功")
            print(f"识别结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"✗ 人脸识别测试失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 人脸识别测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 人脸识别功能测试 ===")
    test_face_recognition() 