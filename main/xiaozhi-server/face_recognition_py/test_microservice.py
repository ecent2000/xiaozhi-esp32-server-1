import requests
import os

# 微服务识别人脸的端点
IDENTIFY_FACE_ENDPOINT = "http://localhost:8001/identify_face/"

# 本地图片路径
image_path = r"F:\xiaozhi-esp32-server-1\main\xiaozhi-server\plugins_func\functions\tmp_face_images\face_upload_e49464fc-7403-4730-8365-29c240b3d539_20250527104056.jpg"

# 检查图片文件是否存在
if not os.path.exists(image_path):
    print(f"错误：图片文件未找到 - {image_path}")
else:
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(image_path), img_file, 'image/jpeg')} # 假设是jpeg, 根据实际情况调整
            # 可以添加其他表单参数，如 model_name, distance_metric, enforce_detection
            data = {
                'model_name': 'VGG-Face', # 或者其他模型
                'distance_metric': 'cosine',
                'enforce_detection': True
            }
            
            print(f"正在向 {IDENTIFY_FACE_ENDPOINT} 发送图片 {os.path.basename(image_path)} 进行识别...")
            response = requests.post(IDENTIFY_FACE_ENDPOINT, files=files, data=data, timeout=60)
            
            # 检查响应状态
            response.raise_for_status() # 如果发生HTTP错误 (4xx 或 5xx)，则抛出异常
            
            response_data = response.json()
            
            print("\n识别成功！")
            print("服务返回结果:")
            import json
            print(json.dumps(response_data, indent=4, ensure_ascii=False))

    except requests.exceptions.ConnectionError as e:
        print(f"\n错误：无法连接到人脸识别微服务 ({IDENTIFY_FACE_ENDPOINT})。")
        print(f"请确保服务正在运行。详细信息: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"\n错误：人脸识别微服务返回HTTP错误。")
        print(f"状态码: {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
    except requests.exceptions.Timeout:
        print(f"\n错误：请求超时。微服务处理时间过长。")
    except Exception as e:
        print(f"\n处理过程中发生未知错误: {e}")
