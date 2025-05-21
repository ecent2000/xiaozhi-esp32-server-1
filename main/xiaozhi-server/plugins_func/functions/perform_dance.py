from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
import asyncio
import json
import os
import random
from pathlib import Path
from core.utils import p3
from core.handle.sendAudioHandle import send_stt_message
from core.utils.dialogue import Message

TAG = __name__
logger = setup_logging()

# 从 play_music.py 借鉴的音乐缓存和文件查找逻辑的简化版本
MUSIC_CACHE_PERFORM_DANCE = {}

perform_dance_function_desc = {
    "type": "function",
    "function": {
        "name": "perform_dance",
        "description": "模拟执行一个舞蹈动作。",
        "parameters": {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "description": "要执行的舞蹈的名称。例如：'街舞', '芭蕾'。",
                }
            },
            "required": ["dance_name"],
        },
    },
}

def _get_random_play_prompt_for_dance(song_name):
    """生成随机播放引导语"""
    clean_name = os.path.splitext(song_name)[0]
    prompts = [
        f"正在为您的舞蹈 {clean_name} 配乐。",
        f"请欣赏舞蹈 {clean_name} 的音乐。",
        f"即将为舞蹈 {clean_name} 播放音乐。",
        f"为您带来 {clean_name} 的伴奏。",
    ]
    return random.choice(prompts)

async def _send_music_to_client_async(conn, dance_name: str):
    """
    异步查找并发送与舞蹈名称相关的音乐给客户端。
    """
    try:
        # 初始化音乐相关配置 (简化版，实际应从 conn.config 获取)
        if not MUSIC_CACHE_PERFORM_DANCE:
            music_dir_config = conn.config.get("plugins", {}).get("play_music", {}).get("music_dir", "./music")
            MUSIC_CACHE_PERFORM_DANCE["music_dir"] = os.path.abspath(music_dir_config)
            MUSIC_CACHE_PERFORM_DANCE["music_ext"] = (".mp3", ".wav", ".p3")
            MUSIC_CACHE_PERFORM_DANCE["music_files"] = []
            if os.path.exists(MUSIC_CACHE_PERFORM_DANCE["music_dir"]):
                for file in Path(MUSIC_CACHE_PERFORM_DANCE["music_dir"]).rglob("*"):
                    if file.is_file() and file.suffix.lower() in MUSIC_CACHE_PERFORM_DANCE["music_ext"]:
                        MUSIC_CACHE_PERFORM_DANCE["music_files"].append(str(file.relative_to(MUSIC_CACHE_PERFORM_DANCE["music_dir"])))

        if not MUSIC_CACHE_PERFORM_DANCE.get("music_files"):
            logger.bind(tag=TAG).warning(f"音乐目录 {MUSIC_CACHE_PERFORM_DANCE.get('music_dir', 'N/A')} 为空或未找到音乐文件。")
            return

        # 尝试根据 dance_name 查找音乐 (简单匹配文件名包含 dance_name)
        # 注意：这里的匹配逻辑非常简单，实际应用中可能需要更复杂的匹配算法
        # 例如，使用 difflib 或其他模糊匹配库，或者期望音乐文件名与 dance_name 精确对应。
        # 为了演示，这里仅作了简化处理。
        selected_music_file = None
        for music_file in MUSIC_CACHE_PERFORM_DANCE["music_files"]:
            if dance_name.lower() in music_file.lower():
                selected_music_file = music_file
                break
        
        if not selected_music_file:
            logger.bind(tag=TAG).info(f"未找到与舞蹈 '{dance_name}' 直接相关的音乐，尝试随机播放一首。")
            selected_music_file = random.choice(MUSIC_CACHE_PERFORM_DANCE["music_files"])


        if selected_music_file:
            music_path = os.path.join(MUSIC_CACHE_PERFORM_DANCE["music_dir"], selected_music_file)
            if not os.path.exists(music_path):
                logger.bind(tag=TAG).error(f"选定的音乐文件不存在: {music_path}")
                return

            # 发送引导语
            prompt_text = _get_random_play_prompt_for_dance(selected_music_file)
            await send_stt_message(conn, prompt_text) # 假设 conn.send_stt_message 是异步的或者可以安全调用
            conn.dialogue.put(Message(role="assistant", content=prompt_text))
            
            # 重置TTS索引，确保引导语优先播放
            # conn.tts_first_text_index = 0 # 移除这两行，因为play_music.py中的conn.tts_first_text_index = 0是在主线程中设置的，这里是异步的，可能会导致冲突
            # conn.tts_last_text_index = 0

            tts_file = await asyncio.to_thread(conn.tts.to_tts, prompt_text) # TTS转换
            if tts_file and os.path.exists(tts_file):
                # conn.tts_last_text_index += 1 # 移除这一行，因为play_music.py中的conn.tts_first_text_index = 0是在主线程中设置的，这里是异步的，可能会导致冲突
                opus_packets_prompt, _ = conn.tts.audio_to_opus_data(tts_file)
                conn.audio_play_queue.put((opus_packets_prompt, None, 0, None)) # 引导语 Opus，索引设为0确保优先
                os.remove(tts_file)

            # 播放音乐
            if music_path.endswith(".p3"):
                opus_packets_music, _ = p3.decode_opus_from_file(music_path)
            else:
                opus_packets_music, _ = conn.tts.audio_to_opus_data(music_path) # 对于非.p3，也用audio_to_opus_data转换
            
            # 确保音乐在引导语之后播放，这里的索引需要小心处理
            # 理想情况下，应该有一个机制来管理音频队列的播放顺序和索引
            # 为简单起见，我们假设引导语播放后，下一个索引是1 (或更大，取决于是否有其他音频在队列中)
            # 这里我们用一个较大的数字，或者依赖于conn.tts_last_text_index的正确管理（如果在主线程中更新）
            # conn.audio_play_queue.put((opus_packets_music, None, conn.tts_last_text_index, None))
            # 暂时使用固定索引1，表示在引导语（索引0）之后。这可能需要根据实际的音频队列管理进行调整。
            conn.audio_play_queue.put((opus_packets_music, None, 1, None))


            logger.bind(tag=TAG).info(f"已将音乐 '{selected_music_file}' 添加到播放队列。")
        else:
            logger.bind(tag=TAG).info(f"未找到与舞蹈 '{dance_name}' 相关的音乐，也无音乐可随机播放。")

    except Exception as e_music:
        logger.bind(tag=TAG).error(f"发送舞蹈相关音乐时异步出错: {e_music} traceback: {json.dumps(asyncio.traceback.format_exc())}")


@register_function("perform_dance", perform_dance_function_desc, ToolType.SYSTEM_CTL)
def perform_dance(conn, dance_name: str):
    """
    模拟执行跳舞动作的函数。
    实际场景中，这里可能会调用客户端接口执行相应的动作。
    """
    try:
        # 构建并发送 LLM 格式的消息给客户端
        if hasattr(conn, 'loop') and conn.loop.is_running() and hasattr(conn, 'websocket') and hasattr(conn, 'session_id'):
            async def _send_dance_feedback_to_client(current_conn, current_dance_name: str):
                try:
                    session_id = current_conn.session_id
                    llm_message_data = {
                        "type": "llm",
                        "text": "💃",  # 跳舞的表情符号
                        "emotion": "happy",
                        "session_id": session_id,
                        "motion_data": {
                            "motion": "跳舞",
                            "dance_name": current_dance_name,
                            "expression": "happy" 
                        }
                    }
                    message_json = json.dumps(llm_message_data, ensure_ascii=False)
                    logger.bind(tag=TAG).info(f"发送舞蹈LLM消息到客户端: {message_json}")
                    await current_conn.websocket.send(message_json)

                    # 在发送舞蹈动作反馈后，异步尝试发送音乐
                    await _send_music_to_client_async(current_conn, current_dance_name)

                except Exception as e_async:
                    logger.bind(tag=TAG).error(f"发送舞蹈LLM消息或音乐时异步出错: {e_async}")

            # 在事件循环中安全地运行异步任务
            asyncio.run_coroutine_threadsafe(
                _send_dance_feedback_to_client(conn, dance_name), 
                conn.loop
            )
        else:
            logger.bind(tag=TAG).warning("无法发送舞蹈LLM消息：conn 对象缺少 loop, websocket 或 session_id 属性，或者 loop 未运行。")

        message = f"已完成 {dance_name} 舞蹈"
        logger.bind(tag=TAG).info(message)
        
        # 可以在这里通过 conn 对象与客户端交互，如果需要的话
        # 例如: conn.send_to_client({"action": "perform_dance", "dance": dance_name})
        
        # 返回一个简单的响应给LLM或调用者
        return ActionResponse(
            action=Action.RESPONSE, 
            result="success", 
            response=f"好的，已经开始跳 {dance_name} 了！"
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"执行舞蹈 '{dance_name}' 时出错: {e}")
        return ActionResponse(
            action=Action.RESPONSE, 
            result="error", 
            response=f"抱歉，尝试跳 {dance_name} 时出错了。"
        )

# 确保 __init__.py 能够发现这个模块中的函数
# 如果 plugins_func/functions/__init__.py 是手动导入各个功能模块的，
# 可能需要在那边添加 from . import perform_dance

# 为了简单起见，这里假设插件系统会自动扫描并注册。
# 如果不是，你可能需要在 `plugins_func/functions/__init__.py` 中添加:
# from .perform_dance import perform_dance
# 并在 `FunctionRegistry` 初始化时确保这些模块被加载。 