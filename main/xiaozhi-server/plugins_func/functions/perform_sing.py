import asyncio
import json
import os
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
from core.utils.util import audio_to_data

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

@register_function("perform_sing", perform_sing_function_desc, ToolType.SYSTEM_CTL)
def perform_sing(conn, song_name: str):
    """
    执行唱歌动作的函数。
    会向客户端发送一个LLM消息，表明正在唱歌。
    """
    try:
        if hasattr(conn, 'loop') and conn.loop.is_running() and hasattr(conn, 'websocket') and hasattr(conn, 'session_id'):
            async def _send_sing_feedback_to_client(current_conn, current_song_name: str):
                try:
                    session_id = current_conn.session_id
                    # 假设一个默认的愉快表情，可以根据歌曲情感调整
                    emotion = "happy" 
                    emoji = "🎤" # 唱歌的表情符号

                    # 发送开始唱歌的消息
                    llm_message_data = {
                        "type": "llm",
                        "text": emoji,
                        "emotion": emotion,
                        "session_id": session_id,
                        "motion_data": {
                            "motion": "唱歌",
                            "song_name": current_song_name,
                            "expression": "happy" # 默认开心，可以根据实际情况调整
                        }
                    }
                    message_json = json.dumps(llm_message_data, ensure_ascii=False)
                    logger.bind(tag=TAG).info(f"发送唱歌LLM消息到客户端: {message_json}")
                    await current_conn.websocket.send(message_json)

                    # 设置语音输入状态，禁用语音输入
                    current_conn.tts_first_text_index = 0
                    current_conn.tts_last_text_index = 1
                    current_conn.asr_server_receive = False

                    # 处理音频文件
                    music_file = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "music", f"{current_song_name}.mp3"))
                    logger.bind(tag=TAG).info(f"尝试加载音乐文件: {music_file}")
                    if os.path.exists(music_file):
                        # 转换音频为opus格式
                        audio_datas, duration = audio_to_data(music_file)
                        if audio_datas:
                            # 发送音频数据
                            for audio_data in audio_datas:
                                await current_conn.websocket.send(audio_data)
                            logger.bind(tag=TAG).info(f"已发送歌曲《{current_song_name}》的音频数据")
                        else:
                            logger.bind(tag=TAG).error(f"音频转换失败: {music_file}")
                    else:
                        logger.bind(tag=TAG).error(f"找不到歌曲文件: {music_file}")

                    # 发送结束唱歌的消息
                    await current_conn.websocket.send(json.dumps({
                        "type": "tts",
                        "state": "stop",
                        "session_id": session_id
                    }))

                    # 恢复语音输入状态
                    current_conn.tts_first_text_index = -1
                    current_conn.tts_last_text_index = -1
                    current_conn.asr_server_receive = True

                except Exception as e_async:
                    logger.bind(tag=TAG).error(f"发送唱歌LLM消息时异步出错: {e_async}")
                    # 确保在出错时也恢复语音输入状态
                    current_conn.tts_first_text_index = -1
                    current_conn.tts_last_text_index = -1
                    current_conn.asr_server_receive = True

            # 在事件循环中安全地运行异步任务
            asyncio.run_coroutine_threadsafe(
                _send_sing_feedback_to_client(conn, song_name), 
                conn.loop
            )
        else:
            logger.bind(tag=TAG).warning("无法发送唱歌LLM消息：conn 对象缺少 loop, websocket 或 session_id 属性，或者 loop 未运行。")

        logger.bind(tag=TAG).info(f"准备演唱: {song_name}")
        return ActionResponse(
            action=Action.RESPONSE, 
            result="success", 
            response=""
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"执行唱歌 '{song_name}' 时出错: {e}")
        return ActionResponse(
            action=Action.RESPONSE, 
            result="error", 
            response=""
        )

# 确保 __init__.py 能够发现这个模块中的函数
# 如果 plugins_func/functions/__init__.py 是手动导入各个功能模块的，
# 可能需要在那边添加 from . import perform_sing
# 例如:
# from .perform_sing import perform_sing
# 并在 FunctionRegistry 初始化时确保这些模块被加载。 