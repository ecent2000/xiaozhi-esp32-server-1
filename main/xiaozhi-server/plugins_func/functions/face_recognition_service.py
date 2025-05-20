import os
import sys
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 动态路径设置，尝试导入 deepface_manager 和 register --- 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
face_recognition_module_dir = os.path.join(current_script_dir, "face_recognition_py")
# functions_dir = current_script_dir # functions 目录本身
# plugins_func_dir = os.path.dirname(functions_dir) # plugins_func 目录

# 添加 face_recognition_py 到 sys.path 以导入 deepface_manager
if face_recognition_module_dir not in sys.path:
    sys.path.append(face_recognition_module_dir)

# # 添加 plugins_func 到 sys.path 以导入 register -- REMOVED THIS BLOCK
# if plugins_func_dir not in sys.path:
#     sys.path.append(plugins_func_dir)

try:
    from deepface_manager import identify_faces_in_image, DEFAULT_DB_PATH, add_face_to_database as dm_add_face
    logging.info("成功导入 deepface_manager 模块。")
except ImportError as e:
    logging.error(f"无法导入 deepface_manager: {e}. 请确保其在 '{face_recognition_module_dir}' 下。")
    # Stubs for deepface_manager functions if import fails
    def identify_faces_in_image(*args, **kwargs):
        # Return a structure that the main function expects for error reporting
        return {"error": "deepface_manager 模块导入失败"} 
    def dm_add_face(*args, **kwargs):
        raise RuntimeError("deepface_manager not loaded, cannot add face.")
    DEFAULT_DB_PATH = "dataset"

try:
    # MODIFIED IMPORT to be like other plugins
    from plugins_func.register import register_function, ToolType, ActionResponse, Action 
    logging.info("成功导入 register 模块。")
except ImportError as e:
    logging.error(f"无法从 plugins_func.register 导入: {e}. 请确保项目结构和 PYTHONPATH 正确。")
    # Stubs for register components if import fails
    def register_function(name, desc, tool_type):
        def decorator(func):
            logging.warning(f"register_function 未能加载，函数 {func.__name__} 将不会被注册。")
            return func
        return decorator
    class ToolType: SYSTEM_CTL = "system_ctl" # Dummy value
    class Action: 
        REQLLM = "REQLLM" # Dummy
        RESPONSE = "RESPONSE" # Dummy
    class ActionResponse: # Dummy
        def __init__(self, action, result, response):
            self.action = action; self.result = result; self.response = response

# --- 服务函数定义 --- 
DB_PATH = os.path.join(face_recognition_module_dir, DEFAULT_DB_PATH)
# 定义默认测试图片路径
DEFAULT_RECOGNITION_IMAGE_PATH = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\plugins_func\functions\face_recognition_py\test_images\test_image_01.jpg"

recognize_face_desc = {
    "type": "function",
    "function": {
        "name": "recognize_face_in_image",
        "description": "使用人脸识别功能识别指定图片中的人脸。如果未提供图片路径，则会尝试识别一个预设的测试图片。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string", 
                    "description": "(可选) 需要进行人脸识别的本地图片路径。例如 D:/images/photo.jpg。如果留空，将使用默认测试图片。"
                }
            },
            "required": [] # image_path is now optional
        }
    }
}

@register_function("recognize_face_in_image", recognize_face_desc, ToolType.SYSTEM_CTL)
def recognize_face_in_image(conn, image_path: str = None) -> ActionResponse: # image_path is now optional
    actual_image_path = image_path
    if not actual_image_path: # Handles None or empty string
        logging.info(f"未提供 image_path，将使用默认测试图片路径: {DEFAULT_RECOGNITION_IMAGE_PATH}")
        actual_image_path = DEFAULT_RECOGNITION_IMAGE_PATH
    else:
        logging.info(f"使用提供的图片路径进行识别: {actual_image_path}")

    if not isinstance(actual_image_path, str) or not actual_image_path.strip():
        # This case should ideally not be hit if default path is valid and logic is correct
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

    if not os.path.exists(DB_PATH):
        try:
            os.makedirs(DB_PATH, exist_ok=True)
            logging.info(f"人脸数据库目录已创建: {DB_PATH}。")
        except OSError as e:
            logging.error(f"创建人脸数据库目录失败 '{DB_PATH}': {e}")
            return ActionResponse(action=Action.RESPONSE, result="数据库错误", response=f"创建人脸数据库目录失败: {DB_PATH}. {e}")
    
    db_has_data = False
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        if any(os.path.isdir(os.path.join(DB_PATH, item)) for item in os.listdir(DB_PATH)):
            db_has_data = True

    if not db_has_data:
        message = f"人脸数据库 '{DB_PATH}' 为空或不包含任何人物数据。请先使用 add_face_for_recognition 添加人脸数据才能进行有效识别。"
        logging.warning(f"recognize_face_in_image: {message}")
        # 即使数据库为空，也让 deepface_manager 处理，它会返回空列表, LLM可以解释此情况

    logging.info(f"开始人脸识别: 图片='{actual_image_path}', 数据库='{DB_PATH}'")
    
    identification_output = identify_faces_in_image(
        image_to_check_path=actual_image_path,
        database_path=DB_PATH
    )

    if isinstance(identification_output, dict) and "error" in identification_output:
        error_msg = identification_output['error']
        logging.error(f"人脸识别内部失败: {error_msg}")
        return ActionResponse(action=Action.REQLLM, result=f"人脸识别失败: {error_msg}", response=None)
    elif isinstance(identification_output, list):
        if not identification_output: 
            result_summary = f"在图片 '{os.path.basename(actual_image_path)}' 中未检测到人脸，或检测到的人脸在数据库中没有匹配项。"
            if not db_has_data:
                result_summary += " (提示: 当前人脸数据库为空)"
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
        if not db_has_data:
            result_summary += " (提示: 当前人脸数据库为空，可能影响识别效果)"

        logging.info(f"人脸识别成功: {result_summary}")
        return ActionResponse(action=Action.REQLLM, result=result_summary, response=None)
    else:
        raw_output_str = f"Type: {type(identification_output)}, Value: {str(identification_output)[:200]}"
        logging.error(f"人脸识别返回未知格式: {raw_output_str}")
        return ActionResponse(action=Action.REQLLM, result=f"人脸识别服务返回了意外的数据格式。", response=None)


add_face_desc = {
    "type": "function",
    "function": {
        "name": "add_face_for_recognition",
        "description": "将指定人物的人脸图片添加到人脸识别数据库中。图片路径是必需的。",
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

    if not os.path.exists(DB_PATH):
        try:
            os.makedirs(DB_PATH, exist_ok=True)
            logging.info(f"人脸数据库目录已创建: {DB_PATH}")
        except OSError as e:
            logging.error(f"创建人脸数据库目录失败 '{DB_PATH}': {e}")
            return ActionResponse(action=Action.RESPONSE, result="数据库错误", response=f"创建人脸数据库目录失败: {DB_PATH}. {e}")

    logging.info(f"开始添加人脸: 图片='{image_path}', 姓名='{person_name}', 数据库='{DB_PATH}'")
    
    try:
        dm_add_face(
            image_path=image_path,
            person_name=person_name,
            database_path=DB_PATH
        )
        person_dir = os.path.join(DB_PATH, person_name)
        image_filename = os.path.basename(image_path)
        destination_path = os.path.join(person_dir, image_filename)

        if os.path.exists(destination_path):
            message = f"已成功将 '{person_name}' 的人脸图像 '{image_filename}' 添加到数据库。"
            logging.info(message)
            return ActionResponse(action=Action.REQLLM, result=message, response=None)
        else:
            message = f"尝试添加人脸 '{person_name}'，但操作后未在预期位置 '{destination_path}' 找到文件。请检查日志。"
            logging.warning(message)
            return ActionResponse(action=Action.REQLLM, result=message, response=None) 

    except RuntimeError as e: 
        logging.error(f"添加人脸因模块导入失败而中止: {e}")
        return ActionResponse(action=Action.REQLLM, result=f"添加人脸操作失败，因为内部模块未能正确加载。", response=None)
    except Exception as e:
        logging.error(f"添加人脸时发生未知错误: {e}", exc_info=True)
        return ActionResponse(action=Action.REQLLM, result=f"添加人脸时发生意外错误: {str(e)}", response=None)


if __name__ == '__main__':
    # 本地测试部分，需要模拟 conn 对象或移除对它的依赖进行简单测试
    class MockConn:
        def __init__(self):
            self.config = {}
            self.client_ip = "127.0.0.1" # 示例 IP
            self.logger = logging.getLogger("MockConn")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())

    mock_conn = MockConn()

    logging.info(f"测试模式: 当前脚本目录: {current_script_dir}")
    logging.info(f"测试模式: face_recognition_module_dir: {face_recognition_module_dir}")
    logging.info(f"测试模式: DB_PATH (数据库路径): {DB_PATH}")

    # 确保测试数据库目录存在
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH, exist_ok=True)
        logging.info(f"测试: 已创建数据库目录 {DB_PATH}")

    # 创建一个 dummy test image (用于添加和基本识别测试)
    service_generated_test_image_filename = "service_test_image.jpg"
    service_generated_test_image_path = os.path.join(current_script_dir, service_generated_test_image_filename)

    # 用户指定的测试图片路径
    USER_SPECIFIED_TEST_IMAGE_PATH = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\plugins_func\functions\face_recognition_py\test_image.jpg"

    try:
        from PIL import Image, ImageDraw
        # 创建一个简单的图像，希望 deepface 能至少检测到里面有个"脸"
        img = Image.new('RGB', (200, 250), color = 'lightgrey')
        draw = ImageDraw.Draw(img)
        draw.ellipse((50, 50, 150, 150), fill='yellow', outline='black') # Face
        draw.ellipse((70, 80, 90, 100), fill='black')  # Left eye
        draw.ellipse((110, 80, 130, 100), fill='black') # Right eye
        draw.line((70, 125, 130, 125), fill='black', width=3) # Mouth
        img.save(service_generated_test_image_path)
        logging.info(f"测试: 已创建或覆盖服务生成的测试图片: {service_generated_test_image_path}")
        
        test_person_name = "ServiceTestPersonX"
        logging.info(f"\n--- 测试添加人脸 (使用服务生成的图片): {test_person_name} ---")
        add_response = add_face_for_recognition(mock_conn, service_generated_test_image_path, test_person_name)
        logging.info(f"添加人脸 ActionResponse: action={add_response.action}, result='{add_response.result}', response='{add_response.response}'")

        if add_response.action == Action.REQLLM and "成功" in add_response.result:
            logging.info(f"\n--- 测试识别人脸 (使用服务生成的、刚添加的图片) ---")
            rec_response = recognize_face_in_image(mock_conn, service_generated_test_image_path)
            logging.info(f"服务生成图片识别 ActionResponse: action={rec_response.action}, result='{rec_response.result}', response='{rec_response.response}'")
        else:
            logging.error("使用服务生成的图片添加测试人脸未按预期成功，相关识别测试可能不准确。")

        # 测试识别用户指定的图片
        logging.info(f"\n--- 测试识别人脸 (使用用户指定的特定图片: {USER_SPECIFIED_TEST_IMAGE_PATH}) ---")
        if os.path.exists(USER_SPECIFIED_TEST_IMAGE_PATH):
            rec_response_user_img = recognize_face_in_image(mock_conn, USER_SPECIFIED_TEST_IMAGE_PATH)
            logging.info(f"用户指定图片识别 ActionResponse: action={rec_response_user_img.action}, result='{rec_response_user_img.result}', response='{rec_response_user_img.response}'")
        else:
            logging.warning(f"用户指定的测试图片 {USER_SPECIFIED_TEST_IMAGE_PATH} 不存在，跳过此项识别测试。请确保图片存在于该路径。")

        # 测试识别不存在的图片
        logging.info(f"\n--- 测试识别人脸 (使用不存在的图片路径) ---")
        rec_response_nonexist = recognize_face_in_image(mock_conn, "non_existent_image.jpg")
        logging.info(f"识别不存在图片 ActionResponse: action={rec_response_nonexist.action}, result='{rec_response_nonexist.result}', response='{rec_response_nonexist.response}'")

    except ImportError:
        logging.warning("Pillow 未安装，无法创建服务生成的测试图片。请手动准备或安装 Pillow (pip install Pillow)。用户指定的图片测试仍可进行（如果存在）。")
        # 即使Pillow未安装，仍然尝试测试用户指定的图片（如果路径有效）
        if os.path.exists(USER_SPECIFIED_TEST_IMAGE_PATH):
            logging.info(f"\n--- 测试识别人脸 (Pillow未装, 使用用户指定的特定图片: {USER_SPECIFIED_TEST_IMAGE_PATH}) ---")
            rec_response_user_img_no_pillow = recognize_face_in_image(mock_conn, USER_SPECIFIED_TEST_IMAGE_PATH)
            logging.info(f"用户指定图片识别 (Pillow未装) ActionResponse: action={rec_response_user_img_no_pillow.action}, result='{rec_response_user_img_no_pillow.result}', response='{rec_response_user_img_no_pillow.response}'")
        else:
            logging.warning(f"Pillow未安装，且用户指定的测试图片 {USER_SPECIFIED_TEST_IMAGE_PATH} 也不存在。跳过图片识别测试。")
    except RuntimeError as e:
        if "deepface_manager not loaded" in str(e):
            logging.error(f"测试中止: deepface_manager 模块未能加载。请检查之前的错误日志。 Error: {e}")
        else:
            logging.error(f"测试期间发生运行时错误: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"测试主程序块发生一般错误: {e}", exc_info=True)

    # 可选：清理测试生成的文件
    # if os.path.exists(service_generated_test_image_path):
    #     os.remove(service_generated_test_image_path)
    # test_person_folder_in_db = os.path.join(DB_PATH, test_person_name) # test_person_name 在 try 块中定义
    # if 'test_person_name' in locals() and os.path.exists(test_person_folder_in_db):
    #     import shutil
    #     shutil.rmtree(test_person_folder_in_db) 
    #     logging.info(f"测试: 已清理测试人物文件夹 {test_person_folder_in_db}") 