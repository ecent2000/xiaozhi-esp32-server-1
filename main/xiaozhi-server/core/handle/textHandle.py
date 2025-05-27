import json
from core.handle.abortHandle import handleAbortMessage
from core.handle.helloHandle import handleHelloMessage
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import startToChat, handleAudioMessage
from core.handle.sendAudioHandle import send_stt_message, send_tts_message
from core.handle.iotHandle import handleIotDescriptors, handleIotStatus
from core.handle.ttsReportHandle import enqueue_tts_report
import asyncio

# Added imports
from plugins_func.functions.face_recognition_service import process_uploaded_face_image
from plugins_func.register import Action

TAG = __name__


async def handleTextMessage(conn, message):
    """处理文本消息"""
    conn.logger.bind(tag=TAG).info(f"收到文本消息：{message}")
    try:
        msg_json = json.loads(message)
        if isinstance(msg_json, int):
            await conn.websocket.send(message)
            return
        if msg_json["type"] == "hello":
            await handleHelloMessage(conn, msg_json)
        elif msg_json["type"] == "abort":
            await handleAbortMessage(conn)
        elif msg_json["type"] == "listen":
            if "mode" in msg_json:
                conn.client_listen_mode = msg_json["mode"]
                conn.logger.bind(tag=TAG).debug(
                    f"客户端拾音模式：{conn.client_listen_mode}"
                )
            if msg_json["state"] == "start":
                conn.client_have_voice = True
                conn.client_voice_stop = False
            elif msg_json["state"] == "stop":
                conn.client_have_voice = True
                conn.client_voice_stop = True
                if len(conn.asr_audio) > 0:
                    await handleAudioMessage(conn, b"")
            elif msg_json["state"] == "detect":
                conn.asr_server_receive = False
                conn.client_have_voice = False
                conn.asr_audio.clear()
                if "text" in msg_json:
                    text = msg_json["text"]
                    _, text = remove_punctuation_and_length(text)

                    # 识别是否是唤醒词
                    is_wakeup_words = text in conn.config.get("wakeup_words")
                    # 是否开启唤醒词回复
                    enable_greeting = conn.config.get("enable_greeting", True)

                    if is_wakeup_words and not enable_greeting:
                        # 如果是唤醒词，且关闭了唤醒词回复，就不用回答
                        await send_stt_message(conn, text)
                        await send_tts_message(conn, "stop", None)
                    elif is_wakeup_words:
                        # 上报纯文字数据（复用ASR上报功能，但不提供音频数据）
                        enqueue_tts_report(conn, 1, "嘿，你好呀", [])
                        await startToChat(conn, "嘿，你好呀")
                    else:
                        # 上报纯文字数据（复用ASR上报功能，但不提供音频数据）
                        enqueue_tts_report(conn, 1, text, [])
                        # 否则需要LLM对文字内容进行答复
                        await startToChat(conn, text)
        elif msg_json["type"] == "iot":
            if "descriptors" in msg_json:
                asyncio.create_task(handleIotDescriptors(conn, msg_json["descriptors"]))
            if "states" in msg_json:
                asyncio.create_task(handleIotStatus(conn, msg_json["states"]))
            
            # New logic for handling iot messages with image_data
            if "image_data" in msg_json:
                image_data_base64 = msg_json["image_data"]
                conn.logger.bind(tag=TAG).info(f"收到包含 image_data 的 IoT 消息，准备进行人脸识别。")

                action_response = None 
                try:
                    # Run the blocking face recognition function in a thread pool
                    action_response = await conn.loop.run_in_executor(
                        conn.executor,
                        process_uploaded_face_image, 
                        conn,                      
                        image_data_base64          
                    )
                    
                    conn.logger.bind(tag=TAG).info(f"人脸识别服务返回: action={action_response.action}, result='{action_response.result}', response='{action_response.response}'")

                    text_for_llm = None
                    if action_response.action == Action.REQLLM:
                        text_for_llm = action_response.result
                        if not text_for_llm: 
                            conn.logger.bind(tag=TAG).warning("人脸识别 REQLLM 但 result 为空。")
                            text_for_llm = "人脸识别已尝试处理，但没有具体结果可报告。"
                    # process_uploaded_face_image in face_recognition_service.py primarily returns REQLLM.
                    # It wraps errors within REQLLM, setting result to the error message.
                    # Explicit checks for Action.RESPONSE or Action.ERROR are for robustness against
                    # potential future changes in that service, though not strictly needed with its current version.
                    elif action_response.action == Action.RESPONSE: 
                        text_for_llm = action_response.response or action_response.result
                        if not text_for_llm:
                            conn.logger.bind(tag=TAG).warning("人脸识别 RESPONSE 但 response 和 result 都为空。")
                            text_for_llm = "人脸识别操作已完成。"
                    elif action_response.action == Action.ERROR: 
                        error_detail = action_response.result or "未知详情"
                        text_for_llm = f"抱歉，人脸识别过程中似乎出了点问题：{error_detail}"
                    else: # Handles any unexpected action type
                        conn.logger.bind(tag=TAG).error(f"未知或未处理的人脸识别 Action: {action_response.action}")
                        text_for_llm = "人脸识别操作完成，但返回了意外的状态。"

                    if text_for_llm:
                        await startToChat(conn, text_for_llm)

                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"调用或处理人脸识别服务时发生异常: {e}", exc_info=True)
                    await startToChat(conn, f"抱歉，尝试处理人脸识别请求时系统遇到内部错误。")

        elif msg_json["type"] == "server":
            # 如果配置是从API读取的，则需要验证secret
            if not conn.read_config_from_api:
                return
            # 获取post请求的secret
            post_secret = msg_json.get("content", {}).get("secret", "")
            secret = conn.config["manager-api"].get("secret", "")
            # 如果secret不匹配，则返回
            if post_secret != secret:
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "server",
                            "status": "error",
                            "message": "服务器密钥验证失败",
                        }
                    )
                )
                return
            # 动态更新配置
            if msg_json["action"] == "update_config":
                try:
                    # 更新WebSocketServer的配置
                    if not conn.server:
                        await conn.websocket.send(
                            json.dumps(
                                {
                                    "type": "config_update_response",
                                    "status": "error",
                                    "message": "无法获取服务器实例",
                                }
                            )
                        )
                        return

                    if not await conn.server.update_config():
                        await conn.websocket.send(
                            json.dumps(
                                {
                                    "type": "config_update_response",
                                    "status": "error",
                                    "message": "更新服务器配置失败",
                                }
                            )
                        )
                        return

                    # 发送成功响应
                    await conn.websocket.send(
                        json.dumps(
                            {
                                "type": "config_update_response",
                                "status": "success",
                                "message": "配置更新成功",
                            }
                        )
                    )
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"更新配置失败: {str(e)}")
                    await conn.websocket.send(
                        json.dumps(
                            {
                                "type": "config_update_response",
                                "status": "error",
                                "message": f"更新配置失败: {str(e)}",
                            }
                        )
                    )
            # 重启服务器
            elif msg_json["action"] == "restart":
                await conn.handle_restart(msg_json)
    except json.JSONDecodeError:
        await conn.websocket.send(message)
    except Exception as e: 
        conn.logger.bind(tag=TAG).error(f"处理文本消息时发生根级别意外错误: {e}", exc_info=True)
        try:
            await conn.websocket.send(json.dumps({
                "type": "error",
                "message": "处理您的请求时发生服务器内部错误。"
            }))
        except Exception as send_e:
            conn.logger.bind(tag=TAG).error(f"发送根级别错误消息到客户端失败: {send_e}")
