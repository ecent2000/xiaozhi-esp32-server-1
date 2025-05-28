import os
import base64
import json
import asyncio
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
        "description": "触发客户端进行图片视觉分析，向客户端发送视觉识别命令",
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
        # 验证base64数据
        if not image_base64 or not image_base64.strip():
            logger.bind(tag=TAG).error("Base64图片数据为空")
            return None
            
        # 尝试解码base64验证数据有效性
        try:
            import base64
            image_bytes = base64.b64decode(image_base64)
            if len(image_bytes) == 0:
                logger.bind(tag=TAG).error("Base64解码后的图片数据为空")
                return None
                
            # 简单验证图片格式（检查文件头）
            if not (image_bytes[:2] == b'\xff\xd8' or  # JPEG
                   image_bytes[:8] == b'\x89PNG\r\n\x1a\n' or  # PNG
                   (image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP')):  # WebP
                logger.bind(tag=TAG).warning(f"未识别的图片格式，文件头: {image_bytes[:12].hex()}")
                
        except Exception as decode_error:
            logger.bind(tag=TAG).error(f"Base64数据验证失败: {decode_error}")
            return None
        
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
    """触发客户端进行图片视觉分析"""
    
    logger.bind(tag=TAG).info("image_vision_analysis: 启动视觉分析流程，请求客户端提供图片。")

    # 1. 构建发送给客户端的指令
    iot_message_data = {
        "type": "iot",
        "commands": ["vision_recognition"]
    }

    # 2. 发送消息给客户端
    if conn.websocket and conn.loop:
        async def send_message_async():
            try:
                await conn.websocket.send(json.dumps(iot_message_data))
                logger.bind(tag=TAG).info(f"已向客户端 {conn.client_ip} 发送视觉识别指令: {iot_message_data}")
            except Exception as e:
                logger.bind(tag=TAG).error(f"发送视觉识别指令给客户端失败: {e}")
                raise  # Propagate exception to be caught by future.result()

        future = asyncio.run_coroutine_threadsafe(send_message_async(), conn.loop)
        try:
            future.result(timeout=5) 
        except TimeoutError:
            logger.bind(tag=TAG).error("发送视觉识别指令给客户端超时")
            return ActionResponse(action=Action.RESPONSE, result="向客户端发送视觉识别指令超时。", response="向客户端发送视觉识别指令超时，请检查连接状态。")
        except Exception as e:
            logger.bind(tag=TAG).error(f"发送视觉识别指令时发生错误: {e}")
            return ActionResponse(action=Action.RESPONSE, result=f"向客户端发送视觉识别指令时发生错误: {str(e)}", response="向客户端发送视觉识别指令失败，请检查连接状态。")
    else:
        logger.bind(tag=TAG).error("image_vision_analysis: conn.websocket 或 conn.loop 不可用，无法发送消息。")
        return ActionResponse(action=Action.RESPONSE, result="系统内部错误，无法发送指令给客户端。", response="系统内部错误，无法发送指令给客户端。")

    # 3. 成功发送指令后的返回
    logger.bind(tag=TAG).info("image_vision_analysis: 视觉识别流程已启动。")
    return ActionResponse(
        action=Action.RESPONSE,
        result="已向客户端发送视觉识别指令，等待客户端上传图片进行分析。",
        response="已向客户端发送视觉识别指令，请配合客户端上传图片进行分析。"
    )
