#!/usr/bin/env python3
"""
测试脚本：验证base64图片的处理和保存功能
"""

import os
import base64
import requests
import json

def test_base64_image_processing():
    """测试base64图片处理"""
    
    # 读取测试图片并转换为base64
    test_image_path = "test_images/test_image_01.jpg"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return False
    
    # 读取图片并转换为base64
    with open(test_image_path, 'rb') as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"图片大小: {len(image_bytes)} bytes")
    print(f"Base64长度: {len(image_base64)} characters")
    print(f"文件头: {image_bytes[:12].hex()}")
    
    # 验证图片格式
    if image_bytes[:2] == b'\xff\xd8':
        print("✓ JPEG格式验证通过")
    else:
        print("✗ JPEG格式验证失败")
        return False
    
    # 测试base64解码
    try:
        decoded_bytes = base64.b64decode(image_base64)
        if decoded_bytes == image_bytes:
            print("✓ Base64编解码验证通过")
        else:
            print("✗ Base64编解码验证失败")
            return False
    except Exception as e:
        print(f"✗ Base64解码失败: {e}")
        return False
    
    # 模拟带头部的base64数据
    base64_with_header = f"data:image/jpeg;base64,{image_base64}"
    
    # 测试头部处理
    if ',' in base64_with_header:
        header, data_part = base64_with_header.split(',', 1)
        print(f"✓ 检测到头部: {header}")
        
        decoded_from_header = base64.b64decode(data_part)
        if decoded_from_header == image_bytes:
            print("✓ 带头部的base64处理验证通过")
        else:
            print("✗ 带头部的base64处理验证失败")
            return False
    
    return True

def test_microservice_integration():
    """测试微服务集成"""
    
    test_image_path = "test_images/test_image_01.jpg"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return False
    
    # 读取图片
    with open(test_image_path, 'rb') as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 模拟客户端发送的数据
    base64_with_header = f"data:image/jpeg;base64,{image_base64}"
    
    # 测试人脸注册
    try:
        url = "http://localhost:8001/add_face/"
        
        # 处理base64数据（模拟enroll_new_face_via_service的逻辑）
        if ',' in base64_with_header:
            header, clean_base64 = base64_with_header.split(',', 1)
            print(f"处理头部: {header}")
        else:
            clean_base64 = base64_with_header
        
        processed_bytes = base64.b64decode(clean_base64)
        
        # 创建文件对象发送请求
        import io
        image_file = io.BytesIO(processed_bytes)
        
        files = {'image': ('test_trump.jpg', image_file, 'image/jpeg')}
        data = {'person_name': 'test_trump'}
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            print("✓ 微服务注册测试成功")
            print(f"响应: {response.json()}")
            return True
        else:
            print(f"✗ 微服务注册测试失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 微服务测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Base64图片处理测试 ===")
    
    if test_base64_image_processing():
        print("\n=== 微服务集成测试 ===")
        test_microservice_integration()
    
    print("\n=== 测试完成 ===") 