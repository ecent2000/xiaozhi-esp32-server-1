from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

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