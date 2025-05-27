import os
import sys
import logging
import requests # 新增：用于 HTTP 请求
import json # 新增：用于处理响应
import asyncio # 新增：用于 run_coroutine_threadsafe
import time # 新增：用于生成时间戳和临时文件名
import base64 # 新增：用于处理客户端上传的图片数据
import tempfile # 新增：虽然未使用，但通常与临时文件相关，这里用自定义路径

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

# 定义默认测试图片路径
# DEFAULT_RECOGNITION_IMAGE_PATH 不再是此流程的主要输入
# DEFAULT_RECOGNITION_IMAGE_PATH = r"F:\\xiaozhi-esp32-server-1\\main\\xiaozhi-server\\plugins_func\\functions\\test_images\\test_image_01.jpg"

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
def recognize_face_in_image(conn, image_path: str = None) -> ActionResponse: # image_path 参数在此流程中通常不被使用
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

    # 3. 返回 ActionResponse
    # user_facing_response = "我已经向您的App发送了拍照请求，请您按照App上的提示完成人脸识别操作。" # 根据用户要求，考虑移除或修改
    llm_facing_result = f"已向客户端发送拍照指令: {iot_message_data['commands']}。" # 修改 llm_facing_result，使其更简洁
    
    logging.info(f"recognize_face_in_image: 流程已启动，返回 ActionResponse。LLM result: '{llm_facing_result}'")
    
    return ActionResponse(action=Action.REQLLM, result="", response=None)


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
        if image_bytes[:4] == b'\xff\xd8\xff\xe0' or image_bytes[:4] == b'\xff\xd8\xff\xe1':
            # JPEG 文件头
            image_format = "jpg"
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG 文件头
            image_format = "png"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            # WebP 文件头
            image_format = "webp"
        
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
        return ActionResponse(action=Action.REQLLM, result="处理上传图片失败，无法正确保存图片数据。", response=None)

    logging.info(f"开始调用人脸识别微服务 (客户端图片): 图片='{temp_image_path}'")
    
    try:
        with open(temp_image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(temp_image_path), img_file, 'image/jpeg')} # 假设jpeg
            data = {'enforce_detection': True} # 和原逻辑一致
            
            # 使用 IDENTIFY_FACE_ENDPOINT
            response = requests.post(IDENTIFY_FACE_ENDPOINT, files=files, data=data, timeout=60)
            response.raise_for_status()
            response_data = response.json()

        if response_data and "results" in response_data:
            identification_output = response_data["results"]
            if isinstance(identification_output, dict) and "error" in identification_output:
                error_msg = identification_output['error']
                logging.error(f"人脸识别微服务报告错误 (客户端图片): {error_msg}")
                return ActionResponse(action=Action.REQLLM, result=f"人脸识别失败: {error_msg}", response=None)
            elif isinstance(identification_output, list):
                if not identification_output: 
                    result_summary = f"在您提供的照片中未检测到有效人脸，或者检测到的人脸在我们的数据库中没有匹配项。"
                    logging.info(f"识别人脸 (客户端图片): {result_summary}")
                    return ActionResponse(action=Action.REQLLM, result=result_summary, response=None)
                
                num_faces = len(identification_output)
                confirmed_persons = []
                for face_data in identification_output:
                    # "identity" field in deepface is usually a list of paths, we need "identified_person_name" from our wrapper
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
            else: 
                raw_output_str = f"Type: {type(identification_output)}, Value: {str(identification_output)[:200]}"
                logging.error(f"人脸识别微服务返回了意外的 'results' 格式 (客户端图片): {raw_output_str}")
                return ActionResponse(action=Action.REQLLM, result=f"人脸识别服务返回了意外的数据格式。", response=None)
        elif response_data and "error" in response_data: # FastAPI detail field
            error_msg = response_data.get('detail', response_data['error'])
            logging.error(f"人脸识别微服务调用失败 (客户端图片, 来自响应体): {error_msg}")
            return ActionResponse(action=Action.REQLLM, result=f"人脸识别失败: {error_msg}", response=None)
        else:
            logging.error(f"人脸识别微服务返回未知格式响应 (客户端图片): HTTP {response.status_code}, Body: {response.text[:200]}")
            return ActionResponse(action=Action.REQLLM, result="人脸识别服务通讯或响应格式错误。", response=None)

    except requests.exceptions.RequestException as e:
        logging.error(f"调用人脸识别微服务失败 (客户端图片): {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"无法连接到人脸识别服务: {str(e).splitlines()[-1]}", response=None)
    except json.JSONDecodeError as e:
        logging.error(f"解析人脸识别微服务响应失败 (客户端图片): {e}. Last response text (if any): {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result="解析人脸识别服务响应时出错。", response=None)
    except Exception as e:
        logging.error(f"处理人脸识别时发生未知错误 (客户端图片): {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"处理人脸识别时发生意外错误: {str(e)}", response=None)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            # 如果需要永久保留图片，注释掉下面的os.remove和相关的日志
            # try:
            #     os.remove(temp_image_path)
            #     logging.info(f"已清理临时图片文件: {temp_image_path}")
            # except Exception as e_remove:
            #     logging.error(f"清理临时图片文件失败 '{temp_image_path}': {e_remove})
            
            # 如果选择保留，可以记录文件已保留
            logging.info(f"图片文件已保留在: {temp_image_path}")
        elif temp_image_path: # 文件未成功保存但路径已生成
            logging.info(f"临时图片路径已生成但文件不存在: {temp_image_path}")


add_face_desc = {
    "type": "function",
    "function": {
        "name": "add_face_for_recognition",
        "description": "将指定人物的人脸图片通过微服务添加到人脸识别数据库中。图片路径和人物姓名是必需的。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string", 
                    "description": "包含人脸的本地图片路径。例如 D:/images/photo.jpg 或 ./data/input/person.png"
                },
                "person_name": {
                    "type": "string",
                    "description": "图片中人物的姓名。这将用于在数据库中创建子文件夹。"
                }
            },
            "required": ["image_path", "person_name"]
        }
    }
}

@register_function("add_face_for_recognition", add_face_desc, ToolType.SYSTEM_CTL)
def add_face_for_recognition(conn, image_path: str, person_name: str) -> ActionResponse:
    if not isinstance(image_path, str) or not image_path.strip():
        return ActionResponse(action=Action.RESPONSE, result="参数错误", response="图片路径不能为空。")
    if not isinstance(person_name, str) or not person_name.strip():
        return ActionResponse(action=Action.RESPONSE, result="参数错误", response="人物姓名不能为空。")

    if not os.path.isabs(image_path):
        logging.info(f"提供的图片路径 '{image_path}' 是相对路径，将尝试直接使用。确保路径相对于服务运行位置正确。")

    if not os.path.exists(image_path):
        return ActionResponse(action=Action.RESPONSE, result="文件未找到", response=f"图片文件 '{os.path.basename(image_path)}' 未找到。")

    # 数据库目录创建由微服务处理
    logging.info(f"开始调用微服务添加人脸: 图片='{image_path}', 姓名='{person_name}'")
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(image_path), img_file, 'image/jpeg')}
            data = {'person_name': person_name}
            
            response = requests.post(ADD_FACE_ENDPOINT, files=files, data=data, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()

        # dm_add_face 的原始逻辑是直接操作文件系统并打印日志，微服务抽象了这一点
        # 微服务的响应现在是主要依据
        if response_data and "message" in response_data:
            message = response_data["message"]
            logging.info(f"添加人脸微服务响应: {message}")
            # 可以根据 message 内容判断是否真正成功，或者依赖 HTTP 状态码
            # 假设 200 OK 并且有 message 就代表操作已提交给微服务
            # 原来的 os.path.exists(destination_path) 检查现在不适用，因为文件在微服务那边
            return ActionResponse(action=Action.REQLLM, result=f"向微服务提交添加人脸请求成功: {message}", response=None)
        elif response_data and "error" in response_data: # 例如微服务返回的业务逻辑错误
            error_msg = response_data.get('detail', response_data['error'])
            logging.warning(f"添加人脸微服务报告错误: {error_msg}")
            return ActionResponse(action=Action.REQLLM, result=f"添加人脸失败: {error_msg}", response=None)
        else:
            logging.error(f"添加人脸微服务返回未知格式响应: HTTP {response.status_code}, Body: {response.text[:200]}")
            return ActionResponse(action=Action.REQLLM, result="添加人脸服务通讯或响应格式错误。", response=None)

    except requests.exceptions.RequestException as e:
        logging.error(f"调用添加人脸微服务失败: {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"无法连接到人脸识别服务以添加人脸: {e}", response=None)
    except json.JSONDecodeError as e:
        logging.error(f"解析添加人脸微服务响应失败: {e}. Response text: {response.text[:200]}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result="解析添加人脸服务响应时出错。", response=None)
    except Exception as e: # 包括 dm_add_face 可能抛出的 RuntimeError 等
        logging.error(f"添加人脸时发生未知错误: {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"添加人脸时发生意外错误: {str(e)}", response=None)
