import asyncio
import json
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

perform_sing_function_desc = {
    "type": "function",
    "function": {
        "name": "perform_sing",
        "description": "模拟执行唱歌动作，并向客户端发送状态。",
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
    模拟执行唱歌动作的函数。
    会向客户端发送一个LLM消息，表明正在唱歌。
    """
    try:
        # 构建并发送 LLM 格式的消息给客户端
        if hasattr(conn, 'loop') and conn.loop.is_running() and hasattr(conn, 'websocket') and hasattr(conn, 'session_id'):
            async def _send_sing_feedback_to_client(current_conn, current_song_name: str):
                try:
                    session_id = current_conn.session_id
                    # 假设一个默认的愉快表情，可以根据歌曲情感调整
                    emotion = "happy" 
                    emoji = "🎤" # 唱歌的表情符号

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
                except Exception as e_async:
                    logger.bind(tag=TAG).error(f"发送唱歌LLM消息时异步出错: {e_async}")

            # 在事件循环中安全地运行异步任务
            asyncio.run_coroutine_threadsafe(
                _send_sing_feedback_to_client(conn, song_name), 
                conn.loop
            )
        else:
            logger.bind(tag=TAG).warning("无法发送唱歌LLM消息：conn 对象缺少 loop, websocket 或 session_id 属性，或者 loop 未运行。")

        response_message = f"好的，我来为你唱《{song_name}》！"
        logger.bind(tag=TAG).info(f"准备演唱: {song_name}")
        
        return ActionResponse(
            action=Action.RESPONSE, 
            result="success", 
            response=response_message
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"执行唱歌 '{song_name}' 时出错: {e}")
        return ActionResponse(
            action=Action.RESPONSE, 
            result="error", 
            response=f"抱歉，尝试唱《{song_name}》时出错了。"
        )

# 确保 __init__.py 能够发现这个模块中的函数
# 如果 plugins_func/functions/__init__.py 是手动导入各个功能模块的，
# 可能需要在那边添加 from . import perform_sing
# 例如:
# from .perform_sing import perform_sing
# 并在 FunctionRegistry 初始化时确保这些模块被加载。 