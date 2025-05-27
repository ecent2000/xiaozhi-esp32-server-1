import requests
import os

# 定义常量
DM_DEFAULT_DB_PATH = "dataset"

# FastAPI 服务运行的地址和端口
BASE_URL = "http://localhost:8001"
ADD_FACE_URL = f"{BASE_URL}/add_face/"
IDENTIFY_FACE_URL = f"{BASE_URL}/identify_face/"

# --- 配置测试参数 ---
# 请将 'sample_image.jpg' 替换为你想要用于测试的图片文件路径
# 这张图片应该包含清晰的人脸，并放在与此测试脚本相同的目录下，或者提供完整路径。
TEST_IMAGE_FILENAME = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\face_recognition_py\face_upload_e49464fc-7403-4730-8365-29c240b3d539_20250527104056.jpg"
TEST_PERSON_NAME = "trump" # 用于添加到数据库的人名
# --- --------------- ---

def check_image_exists(filename):
    """
    检查指定的图片文件是否存在。
    """
    if not os.path.exists(filename):
        print(f"错误: 测试图片 '{filename}' 未找到。")
        print(f"请确保图片路径正确且文件存在。")
        return False
    print(f"找到测试图片: {filename}")
    return True


def test_add_face(person_name: str, image_path: str):
    print(f"\n--- 步骤 1: 测试添加人脸 ---")
    print(f"人物姓名: {person_name}")
    print(f"图片路径: {image_path}")

    if not os.path.exists(image_path):
        print(f"错误: 测试图片 '{image_path}' 未找到。请确保图片存在或占位符图片已成功创建。")
        return False

    try:
        with open(image_path, "rb") as image_file:
            # Content-Type 会由 requests 根据文件名自动推断，通常是正确的
            files = {"image": (os.path.basename(image_path), image_file)}
            payload = {"person_name": person_name}
            
            print(f"向 {ADD_FACE_URL} 发送添加人脸请求...")
            response = requests.post(ADD_FACE_URL, files=files, data=payload, timeout=30) # 30秒超时
            
            print(f"服务器响应状态码: {response.status_code}")
            response_json = response.json()
            print("服务器响应内容:")
            import json
            print(json.dumps(response_json, indent=2, ensure_ascii=False))

            if response.status_code == 200:
                print("人脸添加请求似乎已成功提交。")
                return True
            else:
                print(f"人脸添加失败。服务器消息: {response_json.get('detail') or response_json.get('error', '无详细错误信息')}")
                return False

    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到服务器 {BASE_URL}。请确保微服务正在运行。")
        return False
    except requests.exceptions.Timeout:
        print(f"错误: 请求超时。服务器 {BASE_URL} 可能处理时间过长或未响应。")
        return False
    except requests.exceptions.RequestException as e:
        print(f"添加人脸时发生请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"详细错误: {e.response.json()}")
            except ValueError:
                print(f"详细错误 (非JSON): {e.response.text}")
        return False
    except Exception as e:
        print(f"添加人脸过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_identify_face(image_path: str, model_name: str = "VGG-Face", distance_metric: str = "cosine"):
    print(f"\n--- 步骤 2: 测试识别人脸 ---")
    print(f"待识别图片路径: {image_path}")
    print(f"使用模型: {model_name}, 距离度量: {distance_metric}")

    if not os.path.exists(image_path):
        print(f"错误: 测试图片 '{image_path}' 未找到。")
        return

    try:
        with open(image_path, "rb") as image_file:
            files = {"image": (os.path.basename(image_path), image_file)}
            payload = {
                "model_name": model_name,
                "distance_metric": distance_metric,
                "enforce_detection": True # 通常希望强制检测
            }
            
            print(f"向 {IDENTIFY_FACE_URL} 发送识别人脸请求...")
            response = requests.post(IDENTIFY_FACE_URL, files=files, data=payload, timeout=60) # 识别可能耗时更长

            print(f"服务器响应状态码: {response.status_code}")
            response_json = response.json()
            print("服务器响应内容:")
            import json
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            
            if response.status_code == 200:
                results = response_json.get("results", [])
                if isinstance(results, list) and results:
                    print(f"识别成功！找到 {len(results)} 个可能匹配的人脸。")
                    # 检查是否识别到了我们添加的人
                    identified_names = []
                    for result in results: # results 是一个包含识别结果字典的列表
                        # 新的返回格式包含 'identified_person_name' 字段
                        person_name = result.get('identified_person_name')
                        if person_name:
                            identified_names.append(person_name)
                        else:
                            # 备用解析方式，从 identity_path 中提取
                            identity_path = result.get('identity_path', '')
                            if identity_path:
                                name_part = os.path.basename(os.path.dirname(identity_path))
                                identified_names.append(name_part)
                    
                    if TEST_PERSON_NAME in identified_names:
                        print(f"太棒了！在识别结果中找到了我们添加的人物 '{TEST_PERSON_NAME}'。")
                    else:
                        print(f"注意: 在识别结果中 未直接匹配到 '{TEST_PERSON_NAME}'。请检查识别结果详情和图片质量。")
                        print(f"识别出的人员可能为: {list(set(identified_names))}")

                elif isinstance(results, dict) and "error" in results:
                     print(f"人脸识别服务返回错误: {results['error']}")
                else:
                    print("识别服务返回了成功状态码，但结果为空或格式未知。请检查响应详情。")
            else:
                print(f"识别人脸失败。服务器消息: {response_json.get('detail') or response_json.get('error', '无详细错误信息')}")


    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到服务器 {BASE_URL}。请确保微服务正在运行。")
    except requests.exceptions.Timeout:
        print(f"错误: 请求超时。服务器 {BASE_URL} 可能处理时间过长或未响应。")
    except requests.exceptions.RequestException as e:
        print(f"识别人脸时发生请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"详细错误: {e.response.json()}")
            except ValueError:
                print(f"详细错误 (非JSON): {e.response.text}")
    except Exception as e:
        print(f"识别人脸过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("========================================")
    print(" DeepFace 微服务客户端测试脚本")
    print("========================================")
    print(f" 本脚本将尝试: ")
    print(f" 1. 添加名为 '{TEST_PERSON_NAME}' 的人脸，使用图片 '{TEST_IMAGE_FILENAME}'。")
    print(f" 2. 使用同一图片 '{TEST_IMAGE_FILENAME}' 识别人脸。")
    print("----------------------------------------")
    print(f" 请确保 DeepFace 微服务 (deepface_microservice.py) 正在 {BASE_URL} 运行。")
    print("----------------------------------------")

    # 准备测试图片
    if not check_image_exists(TEST_IMAGE_FILENAME):
        print("\n错误: 找不到指定的测试图片。")
        print(f"请确保图片路径正确: {TEST_IMAGE_FILENAME}")
        print("测试中止。")
    else:
        # 1. 测试添加人脸
        add_success = test_add_face(person_name=TEST_PERSON_NAME, image_path=TEST_IMAGE_FILENAME)

        if add_success:
            print("\n等待几秒钟，确保人脸数据已处理完毕...")
            test_identify_face(image_path=TEST_IMAGE_FILENAME)
        else:
            print("\n由于添加人脸步骤失败或未成功，跳过识别人脸测试。")
