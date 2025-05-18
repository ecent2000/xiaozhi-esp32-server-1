from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
import asyncio
import json

TAG = __name__
logger = setup_logging()

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
                except Exception as e_async:
                    logger.bind(tag=TAG).error(f"发送舞蹈LLM消息时异步出错: {e_async}")

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