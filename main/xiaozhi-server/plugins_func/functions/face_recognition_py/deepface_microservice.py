import os
import shutil
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Optional
import tempfile

# 导入 deepface_manager 中的函数和常量
from deepface_manager import (
    add_face_to_database as dm_add_face,
    identify_faces_in_image as dm_identify_faces,
    DEFAULT_DB_PATH as DM_DEFAULT_DB_PATH
)
from deepface import DeepFace # 用于预加载模型

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepFace Microservice")

# 微服务内部的数据库路径，现在参考 deepface_manager.py 中的默认设置
# DM_DEFAULT_DB_PATH 的值是 "dataset"
MICROSERVICE_DB_PATH = os.path.join(os.getcwd(), DM_DEFAULT_DB_PATH) 
# 确保数据库目录存在
os.makedirs(MICROSERVICE_DB_PATH, exist_ok=True)
logger.info(f"人脸数据库路径 (微服务内部，使用DM_DEFAULT_DB_PATH): {MICROSERVICE_DB_PATH}")

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp_image_file:
            shutil.copyfileobj(image.file, tmp_image_file)
            tmp_image_path = tmp_image_file.name
        logger.info(f"临时图片已保存到: {tmp_image_path}")

        # 调用 deepface_manager 中的函数
        # 注意：deepface_manager.add_face_to_database 内部有打印和错误处理
        # 但我们在这里可以捕获更上层的错误或自定义响应
        dm_add_face(
            image_path=tmp_image_path,
            person_name=person_name,
            database_path=MICROSERVICE_DB_PATH # 使用微服务管理的数据库路径
        )
        
        # 检查文件是否真的被添加 (可选，取决于 dm_add_face 的可靠性)
        # 此处简化，假设 dm_add_face 成功则文件已添加
        # dm_add_face 会打印成功信息

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

        # 调用 deepface_manager 中的函数
        results = dm_identify_faces(
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

        # dm_identify_faces 返回列表或带错误的字典
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
    # 获取脚本所在的目录，以便正确设置 MICROSERVICE_DB_PATH (如果 cwd 不是期望的)
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # MICROSERVICE_DB_PATH = os.path.join(current_dir, "microservice_dataset_run")
    # os.makedirs(MICROSERVICE_DB_PATH, exist_ok=True)
    # logger.info(f"本地运行时，数据库路径更新为: {MICROSERVICE_DB_PATH}")
    
    logger.info("启动 DeepFace 微服务 (用于本地测试)...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    # 可以在命令行使用: uvicorn deepface_microservice:app --host 0.0.0.0 --port 8001 --reload 