import asyncio
import json
import os
import random
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
from core.utils import p3
from core.utils.dialogue import Message

TAG = __name__
logger = setup_logging()

perform_sing_function_desc = {
    "type": "function",
    "function": {
        "name": "perform_sing",
        "description": "执行唱歌动作，并向客户端发送状态。",
        "parameters": {
            "type": "object",
            "properties": {
                "song_name": {
                    "type": "string",
                    "description": "要演唱的歌曲的名称。例如：'小星星', '生日快乐歌'。",
                }
            },
            "required": ["song_name"],
        },
    },
}

def _get_sing_intro_prompt(song_name: str) -> str:
    """生成演唱歌曲的随机引导语"""
    prompts = [
        f"好的，接下来我为大家演唱一首《{song_name}》。",
        f"请欣赏我带来的歌曲，《{song_name}》。",
        f"下面，我来唱《{song_name}》。",
        f"为你献上歌曲《{song_name}》。",
    ]
    return random.choice(prompts)

@register_function("perform_sing", perform_sing_function_desc, ToolType.SYSTEM_CTL)
def perform_sing(conn, song_name: str):
    """
    执行唱歌动作的函数。
    会向客户端发送一个LLM消息，表明正在唱歌，并通过audio_play_queue播放歌曲。
    """
    try:
        if hasattr(conn, 'loop') and conn.loop.is_running() and hasattr(conn, 'websocket') and hasattr(conn, 'session_id'):
            async def _send_sing_feedback_to_client(current_conn, current_song_name: str):
                try:
                    session_id = current_conn.session_id
                    emotion = "happy"
                    emoji = "🎤"

                    # 1. 查找并处理歌曲文件
                    music_root_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "music"))
                    song_file_path = None
                    possible_extensions = [".mp3", ".wav", ".p3"]
                    for ext in possible_extensions:
                        path_try = os.path.join(music_root_dir, f"{current_song_name}{ext}")
                        if os.path.exists(path_try):
                            song_file_path = path_try
                            logger.bind(tag=TAG).info(f"找到歌曲文件: {song_file_path}")
                            break
                    
                    opus_packets_song = None
                    duration = 0
                    if song_file_path:
                        if song_file_path.endswith(".p3"):
                            opus_packets_song, duration = p3.decode_opus_from_file(song_file_path)
                        else:
                            opus_packets_song, duration = current_conn.tts.audio_to_opus_data(song_file_path)
                        
                        if not opus_packets_song:
                            logger.bind(tag=TAG).error(f"歌曲《{current_song_name}》音频转换失败: {song_file_path}")
                            song_file_path = None # Treat as not found

                    # 2. 如果找不到歌曲，则发送错误提示并退出
                    if not song_file_path:
                        logger.bind(tag=TAG).error(f"找不到歌曲文件或处理失败: {current_song_name}")
                        error_text = f"抱歉，我没有找到歌曲《{current_song_name}》。"
                        current_conn.dialogue.put(Message(role="assistant", content=error_text))
                        tts_file_err = await asyncio.to_thread(current_conn.tts.to_tts, error_text)
                        if tts_file_err and os.path.exists(tts_file_err):
                            opus_packets_err, _ = current_conn.tts.audio_to_opus_data(tts_file_err)
                            if opus_packets_err:
                                current_conn.audio_play_queue.put((opus_packets_err, None, 0, None))
                            os.remove(tts_file_err)
                        current_conn.llm_finish_task = True
                        return # 结束函数

                    # 3. 发送包含时长的唱歌LLM消息
                    llm_message_data = {
                        "type": "llm",
                        "text": emoji,
                        "emotion": emotion,
                        "session_id": session_id,
                        "motion_data": {
                            "motion": "唱歌",
                            "song_name": current_song_name,
                            "expression": "happy",
                            "duration": duration, # 添加时长
                        }
                    }
                    message_json = json.dumps(llm_message_data, ensure_ascii=False)
                    logger.bind(tag=TAG).info(f"发送唱歌LLM消息到客户端: {message_json}")
                    await current_conn.websocket.send(message_json)
                    logger.bind(tag=TAG).info(f"已发送唱歌LLM消息到客户端: {message_json}")

                    # 4. 设置语音和服务状态
                    current_conn.tts_first_text_index = 0
                    current_conn.tts_last_text_index = 0

                    # 5. 生成并播放引导语
                    intro_text = _get_sing_intro_prompt(current_song_name)
                    current_conn.dialogue.put(Message(role="assistant", content=intro_text))
                    tts_file_intro = await asyncio.to_thread(current_conn.tts.to_tts, intro_text)
                    if tts_file_intro and os.path.exists(tts_file_intro):
                        opus_packets_intro, _ = current_conn.tts.audio_to_opus_data(tts_file_intro)
                        if opus_packets_intro:
                            current_conn.audio_play_queue.put((opus_packets_intro, None, 0, None))
                            current_conn.tts_last_text_index = 1
                        os.remove(tts_file_intro)
                    
                    # 6. 播放歌曲文件
                    current_conn.audio_play_queue.put((opus_packets_song, None, current_conn.tts_last_text_index, None))
                    logger.bind(tag=TAG).info(f"已将歌曲《{current_song_name}》的Opus数据放入播放队列")

                    current_conn.llm_finish_task = True # 标记LLM任务完成

                    # 7. 发送结束唱歌的消息
                    await current_conn.websocket.send(json.dumps({
                        "type": "tts",
                        "state": "stop",
                        "session_id": session_id
                    }))

                except Exception as e_async:
                    logger.bind(tag=TAG).error(f"发送唱歌LLM消息时异步出错: {e_async}")


            asyncio.run_coroutine_threadsafe(
                _send_sing_feedback_to_client(conn, song_name), 
                conn.loop
            )
        else:
            logger.bind(tag=TAG).warning("无法发送唱歌LLM消息：conn 对象缺少 loop, websocket 或 session_id 属性，或者 loop 未运行。")

        logger.bind(tag=TAG).info(f"准备演唱: {song_name}")
        # 返回 Action.NONE 表示指令已接收，异步处理播放
        return ActionResponse(
            action=Action.NONE, 
            result="指令已接收", 
            response=f"好的，这就为您演唱歌曲《{song_name}》。" # 此文本通常由LLM处理，不直接TTS
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"执行唱歌 '{song_name}' 时出错: {e}")
        return ActionResponse(
            action=Action.RESPONSE, 
            result="error", 
            response=f"演唱歌曲《{song_name}》时发生错误。"
        )

# 确保 __init__.py 能够发现这个模块中的函数
# 如果 plugins_func/functions/__init__.py 是手动导入各个功能模块的，
# 可能需要在那边添加 from . import perform_sing
# 例如:
# from .perform_sing import perform_sing
# 并在 FunctionRegistry 初始化时确保这些模块被加载。 