import os
import sys
import logging
import requests # 新增：用于 HTTP 请求
import json # 新增：用于处理响应

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 动态路径设置 --- 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
face_recognition_module_dir = os.path.join(current_script_dir, "face_recognition_py")

# 移除 deepface_manager 的直接导入
# if face_recognition_module_dir not in sys.path:
#     sys.path.append(face_recognition_module_dir)
# try:
#     from deepface_manager import identify_faces_in_image, DEFAULT_DB_PATH, add_face_to_database as dm_add_face
#     logging.info("成功导入 deepface_manager 模块。")
# except ImportError as e:
#     logging.error(f"无法导入 deepface_manager: {e}. 请确保其在 '{face_recognition_module_dir}' 下。")
#     # Stubs for deepface_manager functions if import fails
#     def identify_faces_in_image(*args, **kwargs):
#         return {"error": "deepface_manager 模块导入失败"} 
#     def dm_add_face(*args, **kwargs):
#         raise RuntimeError("deepface_manager not loaded, cannot add face.")
#     DEFAULT_DB_PATH = "dataset" # 仍然保留，但意义改变

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

# DB_PATH 的概念改变：现在它更多地是指向微服务管理的数据库，而不是本地路径
# 不过，为了保持函数签名和原有逻辑的兼容性（例如检查数据库是否为空），
# 我们可以保留 DB_PATH，但它的含义是抽象的。
# 实际的数据库文件管理由微服务完成。
# DEFAULT_DB_PATH 仍然可以来自 deepface_manager (如果需要引用其原始值)
# 或者在这里定义一个象征性的值。由于 dm_add_face 和 identify_faces_in_image
# 不再直接使用，DEFAULT_DB_PATH 的直接本地文件系统意义减弱。
# 我们将不再需要检查本地 DB_PATH 是否存在或有数据。

# 定义默认测试图片路径
DEFAULT_RECOGNITION_IMAGE_PATH = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\plugins_func\functions\face_recognition_py\test_images\test_image_01.jpg" # 保持路径分隔符的一致性

recognize_face_desc = {
    "type": "function",
    "function": {
        "name": "recognize_face_in_image",
        "description": "使用人脸识别微服务识别指定图片中的人脸。如果未提供图片路径，则会尝试识别一个预设的测试图片。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string", 
                    "description": "(可选) 需要进行人脸识别的本地图片路径。例如 D:/images/photo.jpg。如果留空，将使用默认测试图片。"
                }
            },
            "required": [] 
        }
    }
}

@register_function("recognize_face_in_image", recognize_face_desc, ToolType.SYSTEM_CTL)
def recognize_face_in_image(conn, image_path: str = None) -> ActionResponse:
    actual_image_path = image_path
    if not actual_image_path:
        logging.info(f"未提供 image_path，将使用默认测试图片路径: {DEFAULT_RECOGNITION_IMAGE_PATH}")
        actual_image_path = DEFAULT_RECOGNITION_IMAGE_PATH
    else:
        logging.info(f"使用提供的图片路径进行识别: {actual_image_path}")

    if not isinstance(actual_image_path, str) or not actual_image_path.strip():
        logging.error("recognize_face_in_image: 最终的 image_path 无效或为空。")
        return ActionResponse(action=Action.RESPONSE, result="参数错误", response="最终解析的图片路径无效。")

    if not os.path.isabs(actual_image_path):
        logging.info(f"提供的图片路径 '{actual_image_path}' 是相对路径，将尝试直接使用。确保路径相对于服务运行位置正确。")

    if not os.path.exists(actual_image_path):
        logging.error(f"recognize_face_in_image: 图片文件未找到 '{actual_image_path}'")
        user_message = f"图片文件 '{os.path.basename(actual_image_path)}' 未找到。"
        if actual_image_path == DEFAULT_RECOGNITION_IMAGE_PATH:
            user_message += " (这是预设的测试图片路径，请检查该文件是否存在)"
        return ActionResponse(action=Action.RESPONSE, result="文件未找到", response=user_message)
    
    # 数据库存在和是否为空的检查现在由微服务处理，此处不再检查本地DB_PATH
    logging.info(f"开始调用人脸识别微服务: 图片='{actual_image_path}'")
    
    try:
        with open(actual_image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(actual_image_path), img_file, 'image/jpeg')} # 假设jpeg,可改进
            data = {'enforce_detection': True}
            
            response = requests.post(IDENTIFY_FACE_ENDPOINT, files=files, data=data, timeout=60) # 增加超时
            response.raise_for_status() # 如果HTTP错误 (4xx or 5xx), 会抛出异常
            
            response_data = response.json()

        # 处理微服务返回的结果
        if response_data and "results" in response_data:
            identification_output = response_data["results"]
            # 后续处理与之前类似，但基于微服务返回的 identification_output
            if isinstance(identification_output, dict) and "error" in identification_output: # 微服务内部逻辑错误
                error_msg = identification_output['error']
                logging.error(f"人脸识别微服务报告错误: {error_msg}")
                return ActionResponse(action=Action.REQLLM, result=f"人脸识别失败: {error_msg}", response=None)
            elif isinstance(identification_output, list):
                if not identification_output: 
                    result_summary = f"在图片 '{os.path.basename(actual_image_path)}' 中未检测到人脸，或检测到的人脸在数据库中没有匹配项。"
                    # 数据库是否为空的提示可以由微服务返回，或在这里根据特定响应码添加
                    logging.info(f"识别人脸：{result_summary}")
                    return ActionResponse(action=Action.REQLLM, result=result_summary, response=None)
                
                num_faces = len(identification_output)
                confirmed_persons = []
                for face_data in identification_output:
                    if face_data.get("confirmed") and face_data.get("identified_person_name") != "未知身份":
                        confirmed_persons.append(face_data["identified_person_name"])
                
                if confirmed_persons:
                    result_summary = f"在图片 '{os.path.basename(actual_image_path)}' 中识别出 {len(confirmed_persons)} 位已确认身份的人: {', '.join(list(set(confirmed_persons)))}。"
                    if len(confirmed_persons) < num_faces:
                         result_summary += f" (图片中共检测到 {num_faces} 张人脸区域)"
                else:
                    result_summary = f"在图片 '{os.path.basename(actual_image_path)}' 中检测到 {num_faces} 张人脸区域，但未能确认任何已知身份。"

                logging.info(f"人脸识别成功: {result_summary}")
                return ActionResponse(action=Action.REQLLM, result=result_summary, response=None)
            else: # 意外的 "results" 内容
                raw_output_str = f"Type: {type(identification_output)}, Value: {str(identification_output)[:200]}"
                logging.error(f"人脸识别微服务返回了意外的 'results' 格式: {raw_output_str}")
                return ActionResponse(action=Action.REQLLM, result=f"人脸识别服务返回了意外的数据格式。", response=None)
        elif response_data and "error" in response_data: # 微服务直接返回错误（例如，HTTP 400时FastAPI的detail）
            error_msg = response_data.get('detail', response_data['error'])
            logging.error(f"人脸识别微服务调用失败 (来自响应体): {error_msg}")
            return ActionResponse(action=Action.REQLLM, result=f"人脸识别失败: {error_msg}", response=None)
        else: # 响应体不符合预期
            logging.error(f"人脸识别微服务返回未知格式响应: HTTP {response.status_code}, Body: {response.text[:200]}")
            return ActionResponse(action=Action.REQLLM, result="人脸识别服务通讯或响应格式错误。", response=None)

    except requests.exceptions.RequestException as e:
        logging.error(f"调用人脸识别微服务失败: {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"无法连接到人脸识别服务: {e}", response=None)
    except json.JSONDecodeError as e:
        logging.error(f"解析人脸识别微服务响应失败: {e}. Response text: {response.text[:200]}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result="解析人脸识别服务响应时出错。", response=None)
    except Exception as e:
        logging.error(f"处理人脸识别时发生未知错误: {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"处理人脸识别时发生意外错误: {str(e)}", response=None)


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


if __name__ == '__main__':
    class MockConn:
        def __init__(self):
            self.config = {}
            self.client_ip = "127.0.0.1"
            self.logger = logging.getLogger("MockConn") # Use standard logging
            # self.logger.setLevel(logging.INFO) # Already configured by basicConfig
            # if not self.logger.handlers:
            #     self.logger.addHandler(logging.StreamHandler())
    mock_conn = MockConn()

    logging.info(f"测试模式: 当前脚本目录: {current_script_dir}")
    # logging.info(f"测试模式: face_recognition_module_dir: {face_recognition_module_dir}") # Still relevant
    # logging.info(f"测试模式: DB_PATH (数据库路径): {DB_PATH}") # DB_PATH is no longer a local FS path

    # 确保微服务已启动并监听在 DEEPFACE_MICROSERVICE_URL (e.g., http://localhost:8001)
    logging.info(f"--- 前提: 请确保 DeepFace 微服务正在运行于 {DEEPFACE_MICROSERVICE_URL} ---")

    # 用户指定的测试图片路径
    USER_SPECIFIED_TEST_IMAGE_PATH_ADD = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\plugins_func\functions\face_recognition_py\test_images\test_image_01.jpg" # 用于添加
    USER_SPECIFIED_TEST_IMAGE_PATH_RECOGNIZE = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\plugins_func\functions\face_recognition_py\test_images\test_image_02.jpg" # 用于识别

    # 1. 测试添加人脸
    test_person_name = "ServiceTestPersonViaHttp"
    if os.path.exists(USER_SPECIFIED_TEST_IMAGE_PATH_ADD):
        logging.info(f"\n--- 测试添加人脸 (通过HTTP): {test_person_name} using {USER_SPECIFIED_TEST_IMAGE_PATH_ADD} ---")
        add_response = add_face_for_recognition(mock_conn, USER_SPECIFIED_TEST_IMAGE_PATH_ADD, test_person_name)
        logging.info(f"添加人脸 ActionResponse: action={add_response.action}, result='{add_response.result}', response='{add_response.response}'")
    else:
        logging.warning(f"用于添加的测试图片 {USER_SPECIFIED_TEST_IMAGE_PATH_ADD} 不存在，跳过添加测试。")

    # 2. 测试识别人脸 (使用默认测试图片 - 函数内部逻辑)
    # logging.info(f"\n--- 测试识别人脸 (使用默认内置图片路径，通过HTTP) ---")
    # rec_response_default = recognize_face_in_image(mock_conn) # No path provided
    # logging.info(f"默认图片识别 ActionResponse: action={rec_response_default.action}, result='{rec_response_default.result}', response='{rec_response_default.response}'")
    
    # 3. 测试识别人脸 (使用用户指定的图片)
    if os.path.exists(USER_SPECIFIED_TEST_IMAGE_PATH_RECOGNIZE):
        logging.info(f"\n--- 测试识别人脸 (使用用户指定图片 {USER_SPECIFIED_TEST_IMAGE_PATH_RECOGNIZE}，通过HTTP) ---")
        rec_response_user_img = recognize_face_in_image(mock_conn, USER_SPECIFIED_TEST_IMAGE_PATH_RECOGNIZE)
        logging.info(f"用户指定图片识别 ActionResponse: action={rec_response_user_img.action}, result='{rec_response_user_img.result}', response='{rec_response_user_img.response}'")
    else:
        logging.warning(f"用于识别的测试图片 {USER_SPECIFIED_TEST_IMAGE_PATH_RECOGNIZE} 不存在，跳过此项识别测试。")

    # 4. 测试识别不存在的图片 (本地文件检查会先失败)
    logging.info(f"\n--- 测试识别人脸 (使用不存在的图片路径) ---")
    rec_response_nonexist = recognize_face_in_image(mock_conn, "non_existent_image.jpg")
    logging.info(f"识别不存在图片 ActionResponse: action={rec_response_nonexist.action}, result='{rec_response_nonexist.result}', response='{rec_response_nonexist.response}'")

    # 清理逻辑不再适用，因为数据库和文件在微服务中管理
    # 确保 pillow 相关代码被移除或注释掉，因为它用于生成本地测试图片，现在依赖微服务
    # try:
    #     from PIL import Image, ImageDraw ...
    # except ImportError:
    #     ...