#!/usr/bin/env python3
"""
xiaozhi-server 兼容性测试客户端
针对不同版本的websockets库优化
"""

import asyncio
import websockets
import json
import os
import random
import uuid
import subprocess # 用于调用ffmpeg
import time

def generate_mac_address():
    """生成一个随机的MAC地址"""
    # return "00:1A:2B:%02X:%02X:%02X" % (
    #     random.randint(0, 255),
    #     random.randint(0, 255),
    #     random.randint(0, 255),
    # )
    return "3E:8A:F1:6C:2D:B5" # 使用用户指定的MAC地址

def check_ffmpeg():
    """检查ffmpeg是否安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ FFmpeg 已安装")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg 未安装或未在PATH中找到。音频将仅保存为 .opus 格式。")
        print("   请访问 https://ffmpeg.org/download.html 安装 FFmpeg 并将其添加到系统 PATH。")
        return False

async def convert_opus_to_mp3(opus_path, mp3_path):
    """使用ffmpeg将opus文件转换为mp3"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", opus_path, "-acodec", "libmp3lame", "-q:a", "2", mp3_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            print(f"💾 音频已成功转换为 MP3: {mp3_path}")
            return True
        else:
            print(f"❌ FFmpeg 转换失败 (返回码: {process.returncode}):")
            if stdout:
                print(f"   [FFmpeg STDOUT]:\n{stdout.decode(errors='ignore')}")
            if stderr:
                print(f"   [FFmpeg STDERR]:\n{stderr.decode(errors='ignore')}")
            return False
    except FileNotFoundError:
        # ffmpeg 未找到的错误已在 check_ffmpeg 中处理，这里主要捕获其他可能的错误
        print(f"❌ 执行 ffmpeg 失败。请确保 FFmpeg 已安装并配置在系统 PATH 中。")
        return False
    except Exception as e:
        print(f"❌ FFmpeg 转换过程中发生未知错误: {e}")
        return False

async def compatible_test():
    """兼容性测试"""
    
    # 确保tmp目录存在
    os.makedirs("tmp", exist_ok=True)
    ffmpeg_available = check_ffmpeg()
    
    device_id = generate_mac_address()
    client_id = str(uuid.uuid4())
    
    try:
        print(f"🔄 尝试连接到服务器 (Device ID: {device_id}, Client ID: {client_id})...")
        
        headers = {
            "device-id": device_id,
            "client-id": client_id,
            "protocol-version": "1",
            "Authorization": "Bearer test-token"
        }
        
        websocket_connection = None
        try:
            websocket_connection = await websockets.connect("ws://127.0.0.1:8000/xiaozhi/v1/", extra_headers=headers)
        except TypeError:
            websocket_connection = await websockets.connect("ws://127.0.0.1:8000/xiaozhi/v1/", additional_headers=headers)

        async with websocket_connection as websocket:
            print("✅ 连接成功")
            
            hello_message = {
                "type": "hello",
                "version": 1,
                "transport": "websocket", 
                "audio_params": {
                    "format": "opus",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration": 60
                }
            }
            
            print("📤 发送hello消息")
            await websocket.send(json.dumps(hello_message))
            
            hello_response_raw = await websocket.recv()
            print(f"🤝 服务器Hello已确认:")
            try:
                hello_response = json.loads(hello_response_raw)
                print(json.dumps(hello_response, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(hello_response_raw)
            
            # --- 开始对话测试 ---
            session_id_for_test = "test_session_compatible_" + str(uuid.uuid4())[:8]
            text_to_send = "你好小明，给我讲个故事吧"
            
            print(f"\n=== 开始对话 (会话ID: {session_id_for_test}) ===")
            test_message = {
                "session_id": session_id_for_test,
                "type": "listen",
                "state": "detect",
                "text": text_to_send
            }
            
            print(f"💬 发送消息: {text_to_send}")
            await websocket.send(json.dumps(test_message))
            
            print("\n⏳ 开始接收服务器回复...")
            
            # 用于收集当前对话的所有opus数据
            current_conversation_opus_data = bytearray()
            message_count = 0
            tts_completed = False
            
            while not tts_completed:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30)
                    message_count += 1
                    
                    if isinstance(message, bytes):
                        print(f"🎵 [{message_count}] 收到 Opus 音频包: {len(message)} 字节")
                        current_conversation_opus_data.extend(message) # 追加到缓冲区
                        
                    else:
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type", "unknown")
                            session_id_recv = data.get("session_id")

                            print(f"💬 [{message_count}] 收到消息 (类型: {msg_type}, 会话ID: {session_id_recv}):")
                            print(json.dumps(data, indent=2, ensure_ascii=False))
                            
                            if msg_type == "tts" and data.get("state") == "stop":
                                print("🔇 [{message_count}] TTS完成 - 对话结束")
                                tts_completed = True # 标记TTS完成，跳出循环
                                
                        except json.JSONDecodeError:
                            print(f"⚠️  [{message_count}] 非JSON消息: {message}")
                            
                except asyncio.TimeoutError:
                    print("⏰ 等待消息超时")
                    break 
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"🔌 连接已关闭: {e}")
                    tts_completed = True # 标记完成以退出循环
                    break
                except Exception as e:
                    print(f"💥 处理消息时出错: {e}")
                    tts_completed = True # 标记完成以退出循环
                    break
            
            # --- 对话结束后的处理 ---
            if len(current_conversation_opus_data) > 0:
                # 生成文件名
                base_filename = f"conversation_{session_id_for_test}_{int(time.time())}"
                merged_opus_path = os.path.join("tmp", f"{base_filename}.opus")
                final_mp3_path = os.path.join("tmp", f"{base_filename}.mp3")
                
                print(f"\n🔊 合并当前对话的 Opus 数据 ({len(current_conversation_opus_data)} 字节)...")
                with open(merged_opus_path, "wb") as f_opus:
                    f_opus.write(current_conversation_opus_data)
                print(f"💾 已保存合并的 Opus 文件: {merged_opus_path}")
                
                if ffmpeg_available:
                    print(f"🔄 尝试将 {merged_opus_path} 转换为 MP3...")
                    conversion_success = await convert_opus_to_mp3(merged_opus_path, final_mp3_path)
                    if conversion_success:
                        try:
                            os.remove(merged_opus_path) # 删除临时的opus文件
                            print(f"🗑️ 已删除临时 Opus 文件: {merged_opus_path}")
                        except OSError as e_remove:
                            print(f"⚠️ 无法删除临时 Opus 文件 {merged_opus_path}: {e_remove}")
                    else:
                        print(f"💔 MP3 转换失败。保留 Opus 文件: {merged_opus_path}")
                else:
                    print(f"ℹ️ FFmpeg 不可用，仅保存为 Opus 文件: {merged_opus_path}")
            else:
                print("🤷 当前对话未收到任何音频数据。")
                
            print(f"\n🎯 对话测试完成! (会话ID: {session_id_for_test})")
            print(f"📊 总共收到 {message_count} 条消息 (包括文本和音频包)")
            
    except ConnectionRefusedError:
        print("❌ 连接被拒绝，请确保xiaozhi-server正在运行")
    except Exception as e:
        print(f"❌ 测试主流程发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("💡 建议:")
        print("   1. 确保xiaozhi-server正在运行: python app.py")
        print("   2. 检查服务器地址和端口是否正确")
        print("   3. 检查websockets库版本: pip show websockets")

if __name__ == "__main__":
    print("xiaozhi-server 兼容性测试客户端")
    print("=" * 50)
    print("将合并每个对话的音频并尝试转换为MP3 (如果FFmpeg可用)")
    print("使用固定MAC地址: 3E:8A:F1:6C:2D:B5")
    print("=" * 50)
    
    asyncio.run(compatible_test()) 