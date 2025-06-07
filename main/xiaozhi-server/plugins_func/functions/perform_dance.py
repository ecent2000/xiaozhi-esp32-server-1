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

# 音乐缓存
MUSIC_CACHE_PERFORM_DANCE = {}

perform_dance_function_desc = {
    "type": "function",
    "function": {
        "name": "perform_dance",
        "description": "执行一个带有配乐的舞蹈动作。",
        "parameters": {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "description": "要执行的舞蹈的名称。例如：'街舞', '芭蕾'。将根据名称匹配背景音乐。",
                }
            },
            "required": ["dance_name"],
        },
    },
}

def _get_random_play_prompt_for_dance(song_name):
    """为舞蹈配乐生成随机引导语"""
    clean_name = os.path.splitext(song_name)[0]
    prompts = [
        f"好的，为您带来舞蹈表演，配乐是《{clean_name}》。",
        f"请欣赏我的舞蹈，音乐是《{clean_name}》。",
        f"接下来，我将随着《{clean_name}》的旋律起舞。",
    ]
    return random.choice(prompts)

@register_function("perform_dance", perform_dance_function_desc, ToolType.SYSTEM_CTL)
def perform_dance(conn, dance_name: str):
    """
    执行跳舞动作的函数。
    会向客户端发送一个LLM消息，并异步查找、播放音乐。
    """
    try:
        if hasattr(conn, 'loop') and conn.loop.is_running() and hasattr(conn, 'websocket') and hasattr(conn, 'session_id'):
            async def _send_dance_feedback_to_client(current_conn, current_dance_name: str):
                try:
                    # 1. 初始化音乐缓存和查找音乐文件
                    if not MUSIC_CACHE_PERFORM_DANCE:
                        music_dir_config = current_conn.config.get("plugins", {}).get("play_music", {}).get("music_dir", "./music")
                        MUSIC_CACHE_PERFORM_DANCE["music_dir"] = os.path.abspath(music_dir_config)
                        MUSIC_CACHE_PERFORM_DANCE["music_ext"] = (".mp3", ".wav", ".p3")
                        MUSIC_CACHE_PERFORM_DANCE["music_files"] = []
                        if os.path.exists(MUSIC_CACHE_PERFORM_DANCE["music_dir"]):
                            for file in Path(MUSIC_CACHE_PERFORM_DANCE["music_dir"]).rglob("*"):
                                if file.is_file() and file.suffix.lower() in MUSIC_CACHE_PERFORM_DANCE["music_ext"]:
                                    MUSIC_CACHE_PERFORM_DANCE["music_files"].append(str(file.relative_to(MUSIC_CACHE_PERFORM_DANCE["music_dir"])))
                    
                    selected_music_file = None
                    if MUSIC_CACHE_PERFORM_DANCE.get("music_files"):
                        for music_file in MUSIC_CACHE_PERFORM_DANCE["music_files"]:
                            if current_dance_name.lower() in music_file.lower():
                                selected_music_file = music_file
                                break
                        if not selected_music_file:
                            logger.bind(tag=TAG).info(f"未找到与舞蹈 '{current_dance_name}' 相关的音乐，随机选择一首。")
                            selected_music_file = random.choice(MUSIC_CACHE_PERFORM_DANCE["music_files"])
                    
                    # 2. 如果没有音乐，只发送跳舞动作
                    if not selected_music_file:
                        logger.bind(tag=TAG).warning("无可用音乐，仅执行舞蹈动作。")
                        session_id = current_conn.session_id
                        llm_message_data = {
                            "type": "llm", "text": "💃", "emotion": "happy", "session_id": session_id,
                            "motion_data": { "motion": "跳舞", "dance_name": current_dance_name, "expression": "happy", "duration": 0 }
                        }
                        await current_conn.websocket.send(json.dumps(llm_message_data, ensure_ascii=False))
                        return

                    # 3. 处理音乐文件，获取时长和Opus数据
                    music_path = os.path.join(MUSIC_CACHE_PERFORM_DANCE["music_dir"], selected_music_file)
                    opus_packets_music, duration = (None, 0)
                    if os.path.exists(music_path):
                        if music_path.endswith(".p3"):
                            opus_packets_music, duration = p3.decode_opus_from_file(music_path)
                        else:
                            opus_packets_music, duration = current_conn.tts.audio_to_opus_data(music_path)
                    
                    if not opus_packets_music:
                        logger.bind(tag=TAG).error(f"音乐文件处理失败: {music_path}")
                        return # or send TTS error message

                    # 4. 发送包含音乐时长的LLM消息
                    session_id = current_conn.session_id
                    llm_message_data = {
                        "type": "llm", "text": "💃", "emotion": "happy", "session_id": session_id,
                        "motion_data": {
                            "motion": "跳舞", "dance_name": current_dance_name, "expression": "happy", "duration": duration
                        }
                    }
                    message_json = json.dumps(llm_message_data, ensure_ascii=False)
                    logger.bind(tag=TAG).info(f"发送舞蹈LLM消息到客户端: {message_json}")
                    await current_conn.websocket.send(message_json)

                    # 5. 生成并播放引导语
                    prompt_text = _get_random_play_prompt_for_dance(selected_music_file)
                    await send_stt_message(current_conn, prompt_text)
                    current_conn.dialogue.put(Message(role="assistant", content=prompt_text))
                    
                    tts_file = await asyncio.to_thread(current_conn.tts.to_tts, prompt_text)
                    if tts_file and os.path.exists(tts_file):
                        opus_packets_prompt, _ = current_conn.tts.audio_to_opus_data(tts_file)
                        if opus_packets_prompt:
                            current_conn.audio_play_queue.put((opus_packets_prompt, None, 0, None))
                        os.remove(tts_file)

                    # 6. 播放音乐
                    current_conn.audio_play_queue.put((opus_packets_music, None, 1, None))
                    logger.bind(tag=TAG).info(f"已将音乐 '{selected_music_file}' 添加到播放队列。")
                    current_conn.llm_finish_task = True

                except Exception as e_async:
                    logger.bind(tag=TAG).error(f"发送舞蹈反馈时异步出错: {e_async}")

            asyncio.run_coroutine_threadsafe(
                _send_dance_feedback_to_client(conn, dance_name), 
                conn.loop
            )
        else:
            logger.bind(tag=TAG).warning("无法发送舞蹈LLM消息：conn 对象缺少 loop, websocket 或 session_id 属性。")

        return ActionResponse(
            action=Action.NONE, 
            result="指令已接收", 
            response=f"好的，这就为您表演舞蹈《{dance_name}》。"
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