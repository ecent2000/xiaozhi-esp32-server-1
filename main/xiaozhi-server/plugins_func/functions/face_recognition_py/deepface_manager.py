import argparse
import os
import shutil
import pandas as pd
from deepface import DeepFace
import traceback
import time  # 添加time模块用于计时

# 默认的人脸数据库路径 (可以根据需要修改)
DEFAULT_DB_PATH = "dataset" 
# 首次运行模型时，deepface会下载模型文件，请确保网络通畅

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

    # 替换或移除不适合作为文件夹名称的字符 (可选，但推荐)
    # person_name_safe = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in person_name).strip()
    # if not person_name_safe:
    #     print(f"错误: 处理后的任务姓名 '{person_name}' 无效。")
    #     return
    
    person_dir = os.path.join(database_path, person_name) # 使用原始 person_name
    
    try:
        os.makedirs(person_dir, exist_ok=True)
    except OSError as e:
        print(f"错误: 创建人物目录 '{person_dir}' 失败: {e}")
        return

    image_filename = os.path.basename(image_path)
    destination_path = os.path.join(person_dir, image_filename)

    # 检查目标文件是否已存在
    if os.path.exists(destination_path):
        print(f"提示: 文件 '{destination_path}' 已存在。如果您想更新，请先删除旧文件或使用不同文件名。")
        # 可以选择是否覆盖: shutil.copy(image_path, destination_path) 并打印覆盖信息
        # 或者提示用户，此处选择不覆盖并返回
        return 

    try:
        shutil.copy(image_path, destination_path)
        print(f"成功: 图像 '{image_filename}' 已添加至数据库，身份为 '{person_name}' (路径: {destination_path})。")

        # DeepFace.find() 在找不到预计算的特征文件 (如 representations_vgg-face.pkl) 或文件过时时，
        # 会自动重新扫描数据库图片并计算特征。
        # 为确保新添加的图片被立即用于识别，可以考虑删除旧的特征 .pkl 文件。
        # 这会导致下次调用 find 时强制重新生成特征库，对于大型数据库可能耗时。
        # 这是一个可选步骤，取决于具体需求。
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
    """
    if not os.path.exists(image_to_check_path):
        print(f"错误: 待识别的图片路径 '{image_to_check_path}' 不存在。")
        return

    if not os.path.exists(database_path) or not os.listdir(database_path):
        print(f"错误: 人脸数据库路径 '{database_path}' 不存在或为空。请先使用 'add' 命令添加人脸。")
        return

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
            return
            
        if not dfs: # 列表为空
            print("在输入图片中没有检测到人脸，或者 DeepFace.find 返回为空列表。")
            return

        processed_faces_count = 0
        for i, df_region in enumerate(dfs):
            if df_region.empty:
                if enforce_detection: # 只有在检测模式下，区分"未检测到脸"和"检测到但无匹配"才有意义
                    face_coords_info = ""
                    if 'source_x' in df_region.columns : # 早期版本可能没有这些列如果find直接返回空df
                         face_coords_info = f"(区域: x={df_region.iloc[0].get('source_x', '?')}, y={df_region.iloc[0].get('source_y', '?')}, w={df_region.iloc[0].get('source_w', '?')}, h={df_region.iloc[0].get('source_h', '?')})"
                    print(f"图片中的第 {i+1} 张检测到的人脸 {face_coords_info} 在数据库中没有找到匹配项。")
                else: # 如果 enforce_detection is False, 意味着输入图片本身就是一张脸，但没匹配
                    print(f"输入的人脸图片在数据库中没有找到匹配项。")
                continue
            
            processed_faces_count += 1
            print(f"\n--- 正在分析图片中检测到的第 {i+1} 张人脸 ---")
            
            # df_region 已由 DeepFace.find 按相似度（距离从小到大）排序
            most_similar_match = df_region.iloc[0] # 取最相似的那个 (第一行)
            
            identity_path = most_similar_match.get('identity', "未知路径")
            distance = most_similar_match.get('distance', float('inf'))
            threshold = most_similar_match.get('threshold', 0.4) # 阈值也由DeepFace提供，随模型和度量变化

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
            
            print(f"  最佳匹配图片: {identity_path}")
            print(f"  识别出的人物 (可能): {identified_person_name}")
            print(f"  与数据库记录的距离: {distance:.4f} (模型阈值: {threshold:.4f})")

            if distance <= threshold:
                print(f"  结论: 确认身份为 '{identified_person_name}' (置信度高)")
            else:
                print(f"  结论: 未能确认身份 (或与 '{identified_person_name}' 相似度不足)")
        
        if processed_faces_count == 0 :
            if not any(not df.empty for df in dfs): # 所有dataframe都为空
                 print("图片中检测到人脸，但在数据库中没有找到任何匹配项。")
            # 如果dfs本身就空，在前面已经处理了
        
        print(f"\n--- 分析完毕 ---")
        print(f"在输入图片中共处理了 {processed_faces_count} 张有效人脸区域。")

        if benchmark:
            print(f"\n性能测试结果:")
            print(f"模型推理耗时: {inference_time:.2f} 秒")
            if processed_faces_count > 0:
                print(f"平均每张人脸推理时间: {inference_time/processed_faces_count:.2f} 秒")

    except FileNotFoundError: # 应该由脚本开头的检查捕获
        print(f"错误: 图片路径 '{image_to_check_path}' 或数据库路径 '{database_path}' 未找到。")
    except ValueError as ve: 
        if "Face detector" in str(ve) and "could not find any face" in str(ve):
             print(f"错误: 使用 '{model_name}' 的人脸检测器未能在输入图片 '{image_to_check_path}' 中找到人脸。")
             print("建议: 尝试不同检测后端(detector_backend), 或确保图片中人脸清晰可见。")
        elif "No image found in" in str(ve) or "cannot be read" in str(ve): # 来自DeepFace的错误
             print(f"错误: 在数据库 '{database_path}' 中没有找到有效的图片，或者图片无法读取。")
             print("建议: 检查数据库路径和内容，确保包含有效的图片文件。")
        else:
            print(f"处理图片时发生值错误: {ve}")
            traceback.print_exc()
    except Exception as e:
        print(f"人脸识别过程中发生未知错误: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="DeepFace 人脸识别与数据库管理脚本。",
        formatter_class=argparse.RawTextHelpFormatter # 更好地显示多行帮助信息
    )
    parser.add_argument(
        "--db", 
        type=str, 
        default=DEFAULT_DB_PATH, 
        help=f"""人脸数据库的根目录路径。
默认为: "{DEFAULT_DB_PATH}" (在脚本同级目录下创建)。"""
    )

    # 子命令解析器
    subparsers = parser.add_subparsers(dest="command", required=True, title="可用命令",
                                       help="选择要执行的操作: 'add' 或 'identify'")

    # --- 'add' 命令 ---
    parser_add = subparsers.add_parser(
        "add", 
        help="向数据库中添加新的人脸图像。",
        description="将一张图片复制到人脸数据库中，并以指定的人物姓名创建子文件夹进行归类。"
    )
    parser_add.add_argument(
        "-i", "--image", 
        type=str, 
        required=True, 
        help="待添加的人脸图像的本地文件路径。"
    )
    parser_add.add_argument(
        "-n", "--name", 
        type=str, 
        required=True, 
        help="图像中人物的姓名。这将作为数据库中存放该人物图片的子文件夹名称。"
    )

    # --- 'identify' 命令 ---
    parser_identify = subparsers.add_parser(
        "identify", 
        help="在数据库中识别给定图像中的人脸。",
        description="检测输入图像中的人脸，并与数据库中的已知人脸进行比对，输出识别结果。"
    )
    parser_identify.add_argument(
        "-i", "--image", 
        type=str, 
        required=True, 
        help="待识别的人脸图像的本地文件路径。"
    )
    parser_identify.add_argument(
        "--model", 
        type=str, 
        default="VGG-Face", 
        help="""使用的人脸识别模型。
可选: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace".
默认为: "VGG-Face"."""
    )
    parser_identify.add_argument(
        "--metric", 
        type=str, 
        default="cosine", 
        help="""用于计算相似度的距离度量。
常用: "cosine", "euclidean", "euclidean_l2".
默认为: "cosine"."""
    )
    parser_identify.add_argument(
        "--no-detect", 
        action="store_false", 
        dest="enforce_detection", 
        help="""禁用人脸检测步骤。
如果使用此选项，脚本将假定输入图像本身已经是一张裁剪好的人脸图片。
默认不启用此项 (即默认执行人脸检测)。"""
    )
    parser_identify.add_argument(
        "--benchmark",
        action="store_true",
        help="显示性能测试信息，包括总耗时和每张人脸的处理时间。"
    )
    parser_identify.set_defaults(enforce_detection=True) # 默认强制执行人脸检测

    args = parser.parse_args()

    # 确保主数据库目录存在，如果不存在则创建
    try:
        os.makedirs(args.db, exist_ok=True)
        # print(f"使用人脸数据库目录: {os.path.abspath(args.db)}")
    except OSError as e:
        print(f"错误: 无法创建或访问数据库目录 '{args.db}': {e}")
        return

    if args.command == "add":
        print(f"\n执行操作: 添加人脸")
        add_face_to_database(args.image, args.name, args.db)
    elif args.command == "identify":
        print(f"\n执行操作: 识别人脸")
        identify_faces_in_image(args.image, args.db, args.model, args.metric, args.enforce_detection, args.benchmark)

if __name__ == "__main__":
    main() 