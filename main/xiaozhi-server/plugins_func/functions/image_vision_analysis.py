import os
import base64
from openai import OpenAI
from pathlib import Path
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action

TAG = __name__
logger = setup_logging()

IMAGE_VISION_ANALYSIS_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "image_vision_analysis",
        "description": "分析test_images目录中的本地图片并生成描述，支持JPG、PNG、JPEG格式的图片文件",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string", 
                    "description": "对图片分析的具体要求，如'描述这张图片的内容'、'识别图片中的文字'等，默认为'详细描述这张图片的内容'",
                },
            },
            "required": [],
        },
    },
}


def encode_image_to_base64(image_path):
    """将图片编码为base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.bind(tag=TAG).error(f"图片编码失败: {e}")
        return None


def call_qwen_vision_api(image_base64, prompt):
    """调用通义千问视觉模型API"""
    try:
        client = OpenAI(
            api_key="sk-581b1d448f66412b8af5d242fcbc583b",
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
        logger.bind(tag=TAG).error(f"调用通义千问API失败: {e}")
        return None


@register_function("image_vision_analysis", IMAGE_VISION_ANALYSIS_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def image_vision_analysis(conn, prompt: str = "详细描述这张图片的内容"):
    """分析test_images目录中的本地图片并生成描述"""
    
    # 测试图片目录 - 使用相对路径
    test_image_dir = os.path.join(
        "face_recognition_py", 
        "test_images"
    )
    
    # 检查目录是否存在
    if not os.path.isdir(test_image_dir):
        return ActionResponse(
            Action.RESPONSE,
            "目录不存在", 
            f"测试图片目录不存在: {os.path.abspath(test_image_dir)}"
        )
    
    # 扫描支持的图片文件
    supported_formats = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for file in os.listdir(test_image_dir):
        if Path(file).suffix.lower() in supported_formats:
            image_files.append(file)
    
    if not image_files:
        return ActionResponse(
            Action.RESPONSE,
            "无图片文件",
            f"在目录 {test_image_dir} 中未找到支持的图片文件"
        )
    
    # 分析所有图片
    results = []
    for file in sorted(image_files):
        img_path = os.path.join(test_image_dir, file)
        logger.bind(tag=TAG).info(f"正在分析图片: {file}")
        
        # 编码图片
        image_base64 = encode_image_to_base64(img_path)
        if not image_base64:
            results.append(f"**{file}**: 编码失败")
            continue
        
        # 调用视觉模型API
        analysis_result = call_qwen_vision_api(image_base64, prompt)
        
        if analysis_result:
            results.append(f"**{file}**:\n{analysis_result}")
        else:
            results.append(f"**{file}**: 分析失败")
    
    if results:
        # 构建结构化的分析结果
        response_text = f"""我已完成对test_images目录中{len(results)}张图片的分析，以下是详细结果：

{chr(10).join([f"{i+1}. {result}" for i, result in enumerate(results)])}

请根据以上图片分析结果，为用户提供有用的信息和见解。"""
        
        return ActionResponse(Action.REQLLM, response_text, None)
    else:
        return ActionResponse(
            Action.RESPONSE,
            "分析失败",
            "所有图片分析都失败了，请检查网络连接"
        )
