#!/usr/bin/env python3
"""
通义千问视觉模型测试脚本
使用本地图片测试视觉分析功能
"""

import os
import sys
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("qwen-vision-test")

# 硬编码API密钥
QWEN_API_KEY = "sk-581b1d448f66412b8af5d242fcbc583b"

def encode_image_to_base64(image_path):
    """将图片编码为base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"图片编码失败: {e}")
        return None

def analyze_image(image_path, prompt, api_key):
    """分析单张图片"""
    logger.info(f"正在分析图片: {image_path}")
    
    # 检查图片文件
    if not os.path.exists(image_path):
        logger.error(f"文件不存在: {image_path}")
        return None
        
    # 检查文件格式
    supported_formats = ['.jpg', '.jpeg', '.png']
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in supported_formats:
        logger.error(f"不支持的图片格式: {file_ext}")
        return None
    
    # 编码图片
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=messages,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='通义千问视觉模型测试脚本')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--dir', type=str, help='包含测试图片的目录路径')
    parser.add_argument('--prompt', type=str, default='详细描述这张图片的内容', 
                        help='分析提示词（默认为"详细描述这张图片的内容"）')
    
    args = parser.parse_args()
    api_key = QWEN_API_KEY
    
    # 处理单张图片
    if args.image:
        result = analyze_image(args.image, args.prompt, api_key)
        if result:
            print("\n分析结果:")
            print("=" * 50)
            print(result)
            print("=" * 50)
    
    # 处理整个目录
    elif args.dir:
        if not os.path.isdir(args.dir):
            logger.error(f"目录不存在: {args.dir}")
            return
        
        logger.info(f"开始批量测试目录: {args.dir}")
        supported_formats = ['.jpg', '.jpeg', '.png']
        
        for file in os.listdir(args.dir):
            if Path(file).suffix.lower() in supported_formats:
                img_path = os.path.join(args.dir, file)
                result = analyze_image(img_path, args.prompt, api_key)
                
                if result:
                    print(f"\n图片: {file}")
                    print("-" * 50)
                    print(result)
                    print("=" * 50)
    
    else:
        logger.error("请指定 --image 或 --dir 参数")
        parser.print_help()

if __name__ == "__main__":
    main()
