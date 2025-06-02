import os
import shutil
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Optional
import tempfile
import pandas as pd
from deepface import DeepFace
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepFace Microservice")

# 默认的人脸数据库路径
DEFAULT_DB_PATH = "dataset"

# 微服务内部的数据库路径
MICROSERVICE_DB_PATH = os.path.join(os.getcwd(), DEFAULT_DB_PATH) 
# 确保数据库目录存在
os.makedirs(MICROSERVICE_DB_PATH, exist_ok=True)
logger.info(f"人脸数据库路径 (微服务内部，使用DEFAULT_DB_PATH): {MICROSERVICE_DB_PATH}")

# ------------------------------------------------------------------------------
# 核心人脸识别功能函数 (原deepface_manager.py内容)
# ------------------------------------------------------------------------------

def add_face_to_database(image_path: str, person_name: str, database_path: str):
    """
    将指定图片添加至人脸数据库中，以人物姓名作为子文件夹进行组织。
    图片将保存为 database_path/person_name/image_filename.ext。

    参数:
        image_path (str): 待添加的人脸图像的本地路径。
        person_name (str): 图像中人物的姓名。
        database_path (str): 人脸数据库的根目录路径。
    """
    if not os.path.exists(image_path):
        print(f"错误: 图像文件 '{image_path}' 未找到。")
        return

    if not person_name.strip():
        print("错误: 人物姓名不能为空。")
        return
    
    person_dir = os.path.join(database_path, person_name) # 使用原始 person_name
    
    try:
        os.makedirs(person_dir, exist_ok=True)
    except OSError as e:
        print(f"错误: 创建人物目录 '{person_dir}' 失败: {e}")
        return

    image_filename = os.path.basename(image_path)
    # 生成更清晰的文件名，使用person_name和时间戳
    timestamp = int(time.time())
    file_extension = os.path.splitext(image_filename)[1] or '.jpg'
    clean_filename = f"{person_name}_{timestamp}{file_extension}"
    destination_path = os.path.join(person_dir, clean_filename)

    # 检查目标文件是否已存在
    if os.path.exists(destination_path):
        print(f"提示: 文件 '{destination_path}' 已存在。如果您想更新，请先删除旧文件或使用不同文件名。")
        return 

    try:
        shutil.copy(image_path, destination_path)
        print(f"成功: 图像 '{clean_filename}' 已添加至数据库，身份为 '{person_name}' (路径: {destination_path})。")

        deleted_pkl = False
        for file_in_db in os.listdir(database_path):
            if file_in_db.startswith("representations_") and file_in_db.endswith(".pkl"):
                try:
                    pkl_path_to_remove = os.path.join(database_path, file_in_db)
                    os.remove(pkl_path_to_remove)
                    print(f"提示: 已删除旧的特征文件 '{pkl_path_to_remove}'，下次识别时将重新分析整个数据库。")
                    deleted_pkl = True
                except OSError as e_remove:
                    print(f"警告: 无法删除旧的特征文件 '{pkl_path_to_remove}': {e_remove}")
        if not deleted_pkl:
            print("提示: 未找到预计算的特征文件进行删除。DeepFace将在需要时处理数据库更新。")

    except shutil.Error as e_copy:
        print(f"错误: 复制文件时出错: {e_copy}")
    except Exception as e_general:
        print(f"错误: 添加人脸时发生未知错误: {e_general}")
        traceback.print_exc()


def identify_faces_in_image(image_to_check_path: str, database_path: str, 
                              model_name: str = "VGG-Face", distance_metric: str = "cosine", 
                              enforce_detection: bool = True, benchmark: bool = False):
    """
    在给定图像中识别人脸，并与人脸数据库进行比对。
    脚本会尝试识别图像中检测到的每一张人脸。

    参数:
        image_to_check_path (str): 待识别的人脸图像的本地路径。
        database_path (str): 人脸数据库的根目录路径。
        model_name (str): 使用的人脸识别模型 (例如: "VGG-Face", "Facenet", "ArcFace")。
        distance_metric (str): 用于计算相似度的距离度量 (例如: "cosine", "euclidean")。
        enforce_detection (bool): 是否强制执行人脸检测。若为False，则假定输入图像已是裁剪好的人脸。
        benchmark (bool): 是否显示性能测试信息。
    返回:
        list | dict: 成功时返回包含识别结果字典的列表，失败时返回包含错误信息的字典。
    """
    if not os.path.exists(image_to_check_path):
        print(f"错误: 待识别的图片路径 '{image_to_check_path}' 不存在。")
        return {"error": f"待识别的图片路径 '{image_to_check_path}' 不存在。"}

    # 修改后的数据库检查逻辑
    db_is_effectively_empty = True
    if os.path.exists(database_path):
        # 检查数据库目录是否包含任何子目录 (代表人物)
        if any(os.path.isdir(os.path.join(database_path, item)) for item in os.listdir(database_path)):
            db_is_effectively_empty = False
        elif not os.listdir(database_path): # 如果目录完全为空
            db_is_effectively_empty = True
    
    if db_is_effectively_empty:
        print(f"提示: 人脸数据库 '{database_path}' 为空或不包含已注册的人物数据。识别结果将为空。")
        return [] # 返回空列表，表示未识别到任何人脸

    recognition_results = [] # 用于存储所有识别结果

    try:
        print(f"正在使用模型 '{model_name}' 和距离度量 '{distance_metric}' 进行人脸识别...")
        print(f"待识别图片: {image_to_check_path}")
        print(f"人脸数据库: {database_path}")
        print(f"强制检测人脸: {enforce_detection}")

        if benchmark:
            print("\n开始性能测试...")
            # 预热模型，确保模型已加载
            DeepFace.find(
                img_path=image_to_check_path,
                db_path=database_path,
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=enforce_detection,
                detector_backend='ssd',
                silent=True
            )
            print("模型预热完成，开始计时...")

        # 开始计时
        start_time = time.time()
        
        # silent=False 会显示 DeepFace 内部的进度条和部分日志，有助于了解过程
        dfs = DeepFace.find(
            img_path=image_to_check_path,
            db_path=database_path,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            detector_backend='ssd',
            silent=False 
        )

        if benchmark:
            inference_time = time.time() - start_time
            print(f"\n模型推理耗时: {inference_time:.2f} 秒")

        # DeepFace.find 返回一个列表 (dfs)，列表中的每个元素是一个 DataFrame。
        # 每个 DataFrame 对应输入图像中检测到的一个人脸区域。
        # DataFrame 内的行是数据库中与该区域匹配的候选人，按相似度排序。

        if not isinstance(dfs, list):
            print(f"错误: DeepFace.find 返回了意外的格式 (期望列表，得到 {type(dfs)})。")
            return {"error": f"DeepFace.find 返回了意外的格式 (期望列表，得到 {type(dfs)})。"}
            
        if not dfs: # 列表为空
            print("在输入图片中没有检测到人脸，或者 DeepFace.find 返回为空列表。")
            # 根据具体情况，这里可以返回空列表表示未检测到，或者一个特定的消息
            return [] # 或者 {"message": "未检测到人脸或无匹配项"}

        processed_faces_count = 0
        for i, df_region in enumerate(dfs):
            if df_region.empty or len(df_region) == 0:
                if enforce_detection: # 只有在检测模式下，区分"未检测到脸"和"检测到但无匹配"才有意义
                    # 对于空的DataFrame，我们无法获取人脸区域坐标信息
                    print(f"图片中的第 {i+1} 张检测到的人脸区域为空或在数据库中没有找到匹配项。")
                else: # 如果 enforce_detection is False, 意味着输入图片本身就是一张脸，但没匹配
                    print(f"输入的人脸图片在数据库中没有找到匹配项 (df_region 为空或无行)。")
                continue
            
            processed_faces_count += 1
            print(f"\n--- 正在分析图片中检测到的第 {i+1} 张人脸 ---")
            
            # df_region 已由 DeepFace.find 按相似度（距离从小到大）排序
            most_similar_match = df_region.iloc[0] # 取最相似的那个 (第一行)
            
            identity_path = most_similar_match.get('identity', "未知路径")
            distance = most_similar_match.get('distance', float('inf'))
            # threshold = most_similar_match.get('threshold', 0.4) # 阈值也由DeepFace提供，随模型和度量变化
            # 在新版本的deepface中，DataFrame直接包含threshold列，对应每个匹配项
            threshold = most_similar_match.get(f'{model_name}_{distance_metric}_threshold', 0.4)

            # 从 'identity' 路径中提取人物姓名
            # 假设数据库结构是 database_path/Person_Name/image.jpg
            identified_person_name = "未知身份"
            if identity_path != "未知路径" and os.path.exists(identity_path):
                try:
                    abs_db_path = os.path.abspath(database_path)
                    # DeepFace 返回的 identity_path 通常是绝对路径
                    if not os.path.isabs(identity_path): # 以防万一不是绝对路径
                        identity_path_abs = os.path.abspath(os.path.join(database_path, identity_path)) # 尝试基于db_path组合
                        if not os.path.exists(identity_path_abs): # 如果组合后仍不存在，则使用原始路径
                           identity_path_abs = os.path.abspath(identity_path)
                    else:
                        identity_path_abs = identity_path

                    relative_to_db = os.path.relpath(identity_path_abs, start=abs_db_path)
                    # relative_to_db 应该是 "Person_Name/image.jpg" 的形式
                    potential_name_dir = os.path.dirname(relative_to_db)
                    if potential_name_dir and potential_name_dir != "." and potential_name_dir not in ["..", os.path.pardir]:
                        identified_person_name = potential_name_dir.split(os.sep)[0]
                    else: # 可能是图片直接在 db_path 根目录，或者路径解析有问题
                        identified_person_name = f"来自数据库根目录 ({os.path.basename(identity_path_abs)})"
                except ValueError: # 例如，identity_path 不在 database_path 之下
                     identified_person_name = f"数据库外路径 ({os.path.basename(identity_path)})"
                except Exception as e_parse:
                    identified_person_name = f"解析名称出错 ({e_parse})"
            
            confirmed = distance <= threshold
            
            recognition_results.append({
                "face_area_index": i + 1,
                "identity_path": identity_path,
                "identified_person_name": identified_person_name,
                "distance": round(float(distance), 4),
                "threshold": round(float(threshold), 4),
                "confirmed": bool(confirmed)
            })

        if processed_faces_count == 0 :
            if not any(not df.empty for df in dfs): # 所有dataframe都为空
                 print("图片中检测到人脸，但在数据库中没有找到任何匹配项。")
        
        print(f"\n--- 分析完毕 ---")
        print(f"在输入图片中共处理了 {processed_faces_count} 张有效人脸区域。")

        if benchmark:
            print(f"\n性能测试结果:")
            print(f"模型推理耗时: {inference_time:.2f} 秒")
            if processed_faces_count > 0:
                print(f"平均每张人脸推理时间: {inference_time/processed_faces_count:.2f} 秒")
        
        return recognition_results

    except FileNotFoundError: # 应该由脚本开头的检查捕获
        print(f"错误: 图片路径 '{image_to_check_path}' 或数据库路径 '{database_path}' 未找到。")
        return {"error": f"图片路径 '{image_to_check_path}' 或数据库路径 '{database_path}' 未找到。"}
    except ValueError as ve: 
        error_message = ""
        if "Face detector" in str(ve) and "could not find any face" in str(ve):
             error_message = f"错误: 使用 '{model_name}' 的人脸检测器未能在输入图片 '{image_to_check_path}' 中找到人脸。"
             print(error_message)
             print("建议: 尝试不同检测后端(detector_backend), 或确保图片中人脸清晰可见。")
        elif "No image found in" in str(ve) or "cannot be read" in str(ve): # 来自DeepFace的错误
             error_message = f"错误: 在数据库 '{database_path}' 中没有找到有效的图片，或者图片无法读取。"
             print(error_message)
             print("建议: 检查数据库路径和内容，确保包含有效的图片文件。")
        else:
            error_message = f"处理图片时发生值错误: {ve}"
            print(error_message)
            traceback.print_exc()
        return {"error": error_message, "details": str(ve)}
    except Exception as e:
        print(f"人脸识别过程中发生未知错误: {e}")
        traceback.print_exc()
        return {"error": f"人脸识别过程中发生未知错误: {e}", "details": str(e)}

# ------------------------------------------------------------------------------
# 服务启动时预加载模型
# ------------------------------------------------------------------------------
MODELS_TO_PRELOAD = ["VGG-Face"] # 可以扩展此列表

@app.on_event("startup")
async def startup_event():
    logger.info("服务启动，开始预加载人脸识别模型...")
    for model_name in MODELS_TO_PRELOAD:
        try:
            DeepFace.build_model(model_name)
            logger.info(f"模型 '{model_name}' 已成功加载。")
        except Exception as e:
            logger.error(f"加载模型 '{model_name}' 失败: {e}")
            # 根据需要，这里可以决定是否因模型加载失败而阻止服务启动
    logger.info("模型预加载完成。")

# ------------------------------------------------------------------------------
# API 端点
# ------------------------------------------------------------------------------

@app.post("/add_face/", summary="向数据库添加人脸")
async def add_face_endpoint(
    person_name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    将指定人物的人脸图片添加到人脸识别数据库中。
    - **person_name**: 图片中人物的姓名。
    - **image**: 包含人脸的图片文件。
    """
    logger.info(f"收到添加人脸请求: 姓名='{person_name}', 图片='{image.filename}'")

    # FastAPI 的 UploadFile 需要保存到临时文件才能被 deepface 处理
    try:
        # 使用更清晰的临时文件名，包含person_name信息
        file_extension = os.path.splitext(image.filename)[1] if image.filename else '.jpg'
        if not file_extension:
            file_extension = '.jpg'  # 默认使用jpg扩展名
            
        timestamp = int(time.time())
        temp_filename = f"{person_name}_{timestamp}{file_extension}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, prefix=f"{person_name}_") as tmp_image_file:
            shutil.copyfileobj(image.file, tmp_image_file)
            tmp_image_path = tmp_image_file.name
        logger.info(f"临时图片已保存到: {tmp_image_path} (原始文件名: {image.filename})")

        # 调用内部函数
        add_face_to_database(
            image_path=tmp_image_path,
            person_name=person_name,
            database_path=MICROSERVICE_DB_PATH # 使用微服务管理的数据库路径
        )
        
        # 清理临时文件
        os.remove(tmp_image_path)
        logger.info(f"临时图片已删除: {tmp_image_path}")

        return JSONResponse(
            status_code=200,
            content={
                "message": f"人脸图像 '{image.filename}' 已为 '{person_name}' 提交处理。",
                "person_name": person_name,
                "database_path_info": f"使用微服务数据库: {MICROSERVICE_DB_PATH}"
            }
        )
    except Exception as e:
        logger.error(f"添加人脸时发生错误: {e}", exc_info=True)
        # 如果临时文件已创建，尝试删除
        if 'tmp_image_path' in locals() and os.path.exists(tmp_image_path):
            try:
                os.remove(tmp_image_path)
                logger.info(f"错误处理：临时图片已删除: {tmp_image_path}")
            except Exception as e_del:
                logger.error(f"错误处理：删除临时图片失败: {e_del}")
        
        raise HTTPException(status_code=500, detail=f"处理图像时发生内部错误: {str(e)}")
    finally:
        await image.close()


@app.post("/identify_face/", summary="识别人脸")
async def identify_face_endpoint(
    image: UploadFile = File(...),
    model_name: Optional[str] = Form("VGG-Face"),
    distance_metric: Optional[str] = Form("cosine"),
    enforce_detection: Optional[bool] = Form(True)
):
    """
    在给定图像中识别人脸，并与人脸数据库进行比对。
    - **image**: 待识别的图片文件。
    - **model_name**: 使用的人脸识别模型 (例如: "VGG-Face", "Facenet")。
    - **distance_metric**: 用于计算相似度的距离度量 (例如: "cosine", "euclidean")。
    - **enforce_detection**: 是否强制执行人脸检测。
    """
    logger.info(f"收到识别人脸请求: 图片='{image.filename}', 模型='{model_name}', 度量='{distance_metric}'")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp_image_file:
            shutil.copyfileobj(image.file, tmp_image_file)
            tmp_image_path = tmp_image_file.name
        logger.info(f"临时图片已保存到: {tmp_image_path}")

        # 调用内部函数
        results = identify_faces_in_image(
            image_to_check_path=tmp_image_path,
            database_path=MICROSERVICE_DB_PATH, # 使用微服务管理的数据库路径
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            benchmark=False # 微服务中一般不直接开启 benchmark，除非特定调试需求
        )

        # 清理临时文件
        os.remove(tmp_image_path)
        logger.info(f"临时图片已删除: {tmp_image_path}")

        # identify_faces_in_image 返回列表或带错误的字典
        if isinstance(results, dict) and "error" in results:
            logger.warning(f"人脸识别返回错误: {results['error']}")
            # 可以选择将此作为HTTP 4xx/5xx 错误或在200响应中包含错误信息
            return JSONResponse(
                status_code=400, # 或 500，取决于错误类型
                content={"error": results['error'], "details": results.get("details")}
            )
        
        logger.info(f"人脸识别成功，找到 {len(results) if isinstance(results, list) else 'N/A'} 个结果。")
        return JSONResponse(
            status_code=200,
            content={"results": results}
        )

    except Exception as e:
        logger.error(f"识别人脸时发生错误: {e}", exc_info=True)
        if 'tmp_image_path' in locals() and os.path.exists(tmp_image_path):
            try:
                os.remove(tmp_image_path)
                logger.info(f"错误处理：临时图片已删除: {tmp_image_path}")
            except Exception as e_del:
                logger.error(f"错误处理：删除临时图片失败: {e_del}")
        raise HTTPException(status_code=500, detail=f"处理图像时发生内部错误: {str(e)}")
    finally:
        await image.close()

# ------------------------------------------------------------------------------
# 运行微服务 (用于本地测试)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("启动 DeepFace 微服务 (用于本地测试)...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    # 可以在命令行使用: uvicorn deepface_microservice:app --host 0.0.0.0 --port 8001 --reload 