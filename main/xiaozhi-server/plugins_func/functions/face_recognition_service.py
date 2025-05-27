import os
import sys
import logging
import requests # 新增：用于 HTTP 请求
import json # 新增：用于处理响应
import asyncio # 新增：用于 run_coroutine_threadsafe
import time # 新增：用于生成时间戳和临时文件名
import base64 # 新增：用于处理客户端上传的图片数据
import tempfile # 新增：虽然未使用，但通常与临时文件相关，这里用自定义路径
import io
from plugins_func.register import Action, ActionResponse

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 动态路径设置 --- 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
face_recognition_module_dir = os.path.join(current_script_dir, "face_recognition_py")

try:
    from plugins_func.register import register_function, ToolType, ActionResponse, Action 
    logging.info("成功导入 register 模块。")
except ImportError as e:
    logging.error(f"无法从 plugins_func.register 导入: {e}. 请确保项目结构和 PYTHONPATH 正确。")
    def register_function(name, desc, tool_type):
        def decorator(func):
            logging.warning(f"register_function 未能加载，函数 {func.__name__} 将不会被注册。")
            return func
        return decorator
    class ToolType: SYSTEM_CTL = "system_ctl"
    class Action: REQLLM = "REQLLM"; RESPONSE = "RESPONSE"
    class ActionResponse:
        def __init__(self, action, result, response):
            self.action = action; self.result = result; self.response = response

# --- 微服务配置 ---
DEEPFACE_MICROSERVICE_URL = "http://localhost:8001" # 微服务地址，后续可配置
ADD_FACE_ENDPOINT = f"{DEEPFACE_MICROSERVICE_URL}/add_face/"
IDENTIFY_FACE_ENDPOINT = f"{DEEPFACE_MICROSERVICE_URL}/identify_face/"

recognize_face_desc = {
    "type": "function",
    "function": {
        "name": "recognize_face_in_image",
        "description": "启动完整的人脸识别流程，包括请求客户端App拍照并进行身份识别。",
        "parameters": {
            "type": "object",
            "properties": {}, # 无需外部参数来启动此流程
            "required": [] 
        }
    }
}

@register_function("recognize_face_in_image", recognize_face_desc, ToolType.SYSTEM_CTL)
def recognize_face_in_image(conn) -> ActionResponse | None: # image_path 参数在此流程中通常不被使用
    logging.info("recognize_face_in_image: 启动人脸识别流程，请求客户端提供照片。")

    # 1. 构建发送给客户端的指令
    iot_message_data = {
        "type": "iot",
        "commands": ["face_recognition"]
    }

    # 2. 发送消息给客户端
    if conn.websocket and conn.loop:
        async def send_message_async():
            try:
                await conn.websocket.send(json.dumps(iot_message_data))
                logging.info(f"已向客户端 {conn.client_ip} 发送人脸识别指令: {iot_message_data}")
            except Exception as e:
                logging.error(f"发送人脸识别指令给客户端失败: {e}")
                raise # Propagate exception to be caught by future.result()

        future = asyncio.run_coroutine_threadsafe(send_message_async(), conn.loop)
        try:
            future.result(timeout=5) 
        except TimeoutError:
            logging.error("发送人脸识别指令给客户端超时")
            # 返回给用户的消息应该是友好的，result给LLM
            return ActionResponse(action=Action.REQLLM, result="向客户端发送人脸识别指令超时。", response=None)
        except Exception as e:
            logging.error(f"发送人脸识别指令时发生错误: {e}")
            return ActionResponse(action=Action.REQLLM, result=f"向客户端发送人脸识别指令时发生错误: {str(e)}", response=None)
    else:
        logging.error("recognize_face_in_image: conn.websocket 或 conn.loop 不可用，无法发送消息。")
        return ActionResponse(action=Action.REQLLM, result="系统内部错误，无法发送指令给客户端。", response=None)

    # 3. 成功发送指令后，不再返回 ActionResponse
    llm_facing_result = f"已向客户端发送拍照指令: {iot_message_data['commands']}。"
    logging.info(f"recognize_face_in_image: 流程已启动。LLM result: '{llm_facing_result}'")
    return None


def _save_uploaded_image(image_data_base64: str, conn_session_id: str) -> str | None:
    """辅助函数：保存上传的 base64 图片到临时文件，返回文件路径"""
    try:
        # 移除可能的 base64 头部 (e.g., "data:image/jpeg;base64,")
        image_format = "jpg"  # 默认格式
        if ',' in image_data_base64:
            header, image_data_base64 = image_data_base64.split(',', 1)
            # 从header中提取图片格式
            if "image/jpeg" in header.lower() or "image/jpg" in header.lower():
                image_format = "jpg"
            elif "image/png" in header.lower():
                image_format = "png"
            elif "image/webp" in header.lower():
                image_format = "webp"
            else:
                logging.warning(f"未知图片格式: {header}，使用默认jpg格式")
        
        # 验证base64数据不为空
        if not image_data_base64.strip():
            logging.error("Base64图片数据为空")
            return None
            
        image_bytes = base64.b64decode(image_data_base64)
        
        # 验证解码后的数据不为空
        if len(image_bytes) == 0:
            logging.error("Base64解码后的图片数据为空")
            return None
            
        # 简单验证图片格式（检查文件头）
        if image_bytes[:2] == b'\xff\xd8':
            # JPEG 文件头 (更通用的检测)
            image_format = "jpg"
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG 文件头
            image_format = "png"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            # WebP 文件头
            image_format = "webp"
        else:
            logging.warning(f"未识别的图片格式，文件头: {image_bytes[:12].hex()}，使用默认jpg格式")
        
        temp_dir = os.path.join(current_script_dir, "tmp_face_images")
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d%H%M%S")
        temp_file_name = f"face_upload_{conn_session_id}_{timestamp}.{image_format}"
        temp_file_path = os.path.join(temp_dir, temp_file_name)
        
        with open(temp_file_path, 'wb') as f:
            f.write(image_bytes)
        
        # 验证保存的文件是否有效
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            logging.error(f"保存的图片文件无效: {temp_file_path}")
            return None
            
        logging.info(f"客户端上传的人脸图片已保存到: {temp_file_path} (格式: {image_format}, 大小: {len(image_bytes)} bytes)")
        return temp_file_path
    except base64.binascii.Error as b64_error:
        logging.error(f"Base64解码失败: {b64_error}. 输入数据 (前100字符): {image_data_base64[:100]}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"保存上传的图片失败: {e}", exc_info=True)
        return None

def process_uploaded_face_image(conn, image_data_base64: str) -> ActionResponse:
    """
    处理客户端上传的用于人脸识别的图片。
    这个函数应该在 ConnectionHandler 收到客户端图片后被调用。
    """
    logging.info(f"process_uploaded_face_image: 开始处理客户端上传的图片数据 (session: {conn.session_id}).")

    temp_image_path = _save_uploaded_image(image_data_base64, conn.session_id)

    if not temp_image_path:
        return ActionResponse(action=Action.REQLLM, result="处理上传图片失败，图片无法保存。", response=None)

    logging.info(f"开始调用人脸识别微服务 (客户端图片): 图片='{temp_image_path}'")
    
    try:
        with open(temp_image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(temp_image_path), img_file)} 
            data = {'enforce_detection': True} 
            
            response = requests.post(IDENTIFY_FACE_ENDPOINT, files=files, data=data, timeout=60)
            response.raise_for_status()
            response_data = response.json()

        identification_output = response_data["results"]
        
        if not identification_output: 
            result_summary = f"在您提供的照片中未检测到有效人脸，或者检测到的人脸在我们的数据库中没有匹配项。"
            logging.info(f"识别人脸 (客户端图片): {result_summary}")
            return ActionResponse(action=Action.REQLLM, result=result_summary, response=None)
        
        num_faces = len(identification_output)
        confirmed_persons = []
        for face_data in identification_output:
            if face_data.get("confirmed") and face_data.get("identified_person_name") != "未知身份":
                confirmed_persons.append(face_data["identified_person_name"])
        
        if confirmed_persons:
            unique_confirmed_persons = list(set(confirmed_persons))
            result_summary = f"根据您提供的照片，识别出 {len(unique_confirmed_persons)} 位已确认身份的人: {', '.join(unique_confirmed_persons)}。"
            if len(unique_confirmed_persons) < num_faces:
                 result_summary += f" (照片中共检测到 {num_faces} 张人脸区域，部分未确认身份或为同一人多次出现)"
        else:
            result_summary = f"在您提供的照片中检测到 {num_faces} 张人脸区域，但未能确认任何已在我们数据库中的身份。"
        
        logging.info(f"人脸识别成功 (客户端图片): {result_summary}")
        return ActionResponse(action=Action.REQLLM, result=result_summary, response=None)

    except Exception as e:
        # 捕获所有类型的异常，包括requests.exceptions.*, json.JSONDecodeError, KeyError, TypeError等
        logging.error(f"处理人脸识别过程中发生统一捕获的错误 (客户端图片): {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result="处理人脸识别过程中发生错误，请检查服务日志获取详情。", response=None)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            logging.info(f"图片文件已保留在: {temp_image_path}")
        elif temp_image_path: # 文件未成功保存但路径已生成
            logging.info(f"临时图片路径已生成但文件不存在: {temp_image_path}")

TAG_FRS = "face_recognition_service" # Define a TAG for logging if not already present

def enroll_new_face_via_service(conn, person_name: str, image_data_base64: str) -> ActionResponse:
    """
    Calls the face recognition microservice to enroll a new face.

    Args:
        conn: The connection object, used for logging and accessing config.
        person_name: The name of the person to enroll.
        image_data_base64: Base64 encoded string of the image data.

    Returns:
        ActionResponse indicating the outcome of the enrollment.
    """
    logger = getattr(conn, 'logger', logging.getLogger(TAG_FRS))
    try:
        # 处理base64数据，移除可能的头部信息
        if ',' in image_data_base64:
            # 如果包含数据URI头部，移除它
            header, image_data_base64 = image_data_base64.split(',', 1)
            logger.info(f"检测到base64头部: {header}")
        
        # 验证base64数据不为空
        if not image_data_base64.strip():
            logger.error("Base64图片数据为空")
            return ActionResponse(action=Action.ERROR, result="图片数据为空", response=None)
            
        image_bytes = base64.b64decode(image_data_base64)
        
        # 验证解码后的数据不为空
        if len(image_bytes) == 0:
            logger.error("Base64解码后的图片数据为空")
            return ActionResponse(action=Action.ERROR, result="图片解码后数据为空", response=None)
            
        # 验证图片格式（检查文件头）
        is_valid_image = False
        image_format = "jpg"
        
        if image_bytes[:2] == b'\xff\xd8':
            # JPEG 文件头 (更通用的检测)
            is_valid_image = True
            image_format = "jpg"
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG 文件头
            is_valid_image = True
            image_format = "png"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            # WebP 文件头
            is_valid_image = True
            image_format = "webp"
        
        if not is_valid_image:
            logger.error(f"无效的图片格式，文件头: {image_bytes[:12].hex()}")
            return ActionResponse(action=Action.ERROR, result="无效的图片格式", response=None)
            
        logger.info(f"图片格式验证通过: {image_format}, 大小: {len(image_bytes)} bytes")
        
        # Create a file-like object for requests
        image_file_in_memory = io.BytesIO(image_bytes)
        
        # 使用正确的文件名和格式
        files = {'image': (f'{person_name}.{image_format}', image_file_in_memory, f'image/{image_format}')}
        data = {'person_name': person_name}

        add_url = f"{DEEPFACE_MICROSERVICE_URL}/add_face/"
        logger.info(f"正在为 '{person_name}' 注册人脸到 {add_url}")

        # 参考测试文件，使用更长的超时时间
        response = requests.post(add_url, files=files, data=data, timeout=30)
        response.raise_for_status()

        response_json = response.json()
        logger.info(f"人脸注册服务返回响应: {response_json}")

        # 检查响应中的 message 字段
        if response.status_code == 200:
            message = response_json.get("message", f"成功为 {person_name} 注册人脸。")
            return ActionResponse(action=Action.RESPONSE, response=message, result=message)
        else:
            error_msg = response_json.get("message", "注册失败，微服务返回未知错误。")
            logger.error(f"人脸注册服务返回错误: {error_msg}")
            return ActionResponse(action=Action.ERROR, result=f"人脸注册失败: {error_msg}", response=None)

    except requests.exceptions.Timeout:
        logger.error(f"调用人脸注册服务超时: {person_name}")
        return ActionResponse(action=Action.ERROR, result=f"为 {person_name} 注册人脸超时", response=None)
    except requests.exceptions.RequestException as e:
        logger.error(f"人脸注册HTTP请求失败: {e}", exc_info=True)
        return ActionResponse(action=Action.ERROR, result=f"人脸注册服务连接失败: {str(e)}", response=None)
    except base64.binascii.Error as e:
        logger.error(f"Base64解码失败: {e}", exc_info=True)
        return ActionResponse(action=Action.ERROR, result=f"提供的图像数据无效: {str(e)}", response=None)
    except Exception as e:
        logger.error(f"人脸注册过程中发生未知错误: {e}", exc_info=True)
        return ActionResponse(action=Action.ERROR, result=f"人脸注册时发生内部系统错误: {str(e)}", response=None)


