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
import subprocess
import time
import numpy as np
from pydub import AudioSegment

def generate_mac_address():
    """生成一个随机的MAC地址"""
    return "3E:8A:F1:6C:2D:B5" # 使用用户指定的MAC地址

def check_ffmpeg():
    """检查ffmpeg是否安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ FFmpeg 已安装")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg 未安装或未在PATH中找到。音频将仅保存为原始格式。")
        print("   请访问 https://ffmpeg.org/download.html 安装 FFmpeg 并将其添加到系统 PATH。")
        return False

def check_opuslib():
    """检查opuslib_next是否可用"""
    try:
        import opuslib_next
        print("✅ opuslib_next 已安装")
        return True
    except ImportError:
        print("⚠️ opuslib_next 未安装。请使用: pip install opuslib-next")
        return False

def opus_frames_to_wav(opus_frames_data, output_wav_path):
    """将Opus帧数据转换为WAV文件"""
    try:
        import opuslib_next
        
        # 初始化Opus解码器（16kHz, 单声道）
        decoder = opuslib_next.Decoder(16000, 1)
        
        # 解码所有Opus帧
        decoded_samples = []
        
        for frame_data in opus_frames_data:
            if len(frame_data) > 0:  # 跳过空帧
                try:
                    # 解码当前帧
                    decoded_frame = decoder.decode(frame_data, 960)  # 60ms @ 16kHz = 960 samples
                    decoded_samples.append(decoded_frame)
                except Exception as e:
                    print(f"⚠️ 跳过无效的Opus帧: {e}")
                    continue
        
        if not decoded_samples:
            print("❌ 没有有效的Opus帧数据")
            return False
        
        # 合并所有解码的PCM数据
        all_samples = b''.join(decoded_samples)
        
        # 转换为numpy数组
        audio_array = np.frombuffer(all_samples, dtype=np.int16)
        
        # 使用pydub创建AudioSegment
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=16000,
            sample_width=2,  # 16位 = 2字节
            channels=1
        )
        
        # 导出为WAV
        audio_segment.export(output_wav_path, format="wav")
        print(f"💾 已成功将Opus帧转换为WAV: {output_wav_path}")
        return True
        
    except ImportError:
        print("❌ opuslib_next未安装，无法解码Opus帧")
        return False
    except Exception as e:
        print(f"❌ Opus解码失败: {e}")
        return False

async def convert_wav_to_mp3(wav_path, mp3_path):
    """使用ffmpeg将WAV转换为MP3"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", wav_path, "-acodec", "libmp3lame", "-q:a", "2", mp3_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"💾 WAV已成功转换为MP3: {mp3_path}")
            return True
        else:
            print(f"❌ WAV转MP3失败 (返回码: {process.returncode})")
            if stderr:
                print(f"   [STDERR]: {stderr.decode(errors='ignore')[:200]}...")
            return False
    except Exception as e:
        print(f"❌ WAV转MP3过程出错: {e}")
        return False

async def compatible_test():
    """兼容性测试"""
    
    # 确保tmp目录存在
    os.makedirs("tmp", exist_ok=True)
    ffmpeg_available = check_ffmpeg()
    opuslib_available = check_opuslib()
    
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
                "text": text_to_send,
                "source": "text"
            }
            
            print(f"💬 发送消息: {text_to_send}")
            await websocket.send(json.dumps(test_message))
            
            print("\n⏳ 开始接收服务器回复...")
            
            # 收集Opus帧数据（bytes列表）
            opus_frames = []
            message_count = 0
            tts_completed = False
            
            while not tts_completed:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=45)  # 增加超时时间
                    message_count += 1
                    
                    if isinstance(message, bytes):
                        print(f"🎵 [{message_count}] 收到 Opus 音频帧: {len(message)} 字节")
                        opus_frames.append(message)  # 保存为独立的帧
                        
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
            if len(opus_frames) > 0:
                base_filename = f"conversation_{session_id_for_test}_{int(time.time())}"
                
                print(f"\n🔊 处理音频数据 ({len(opus_frames)} 个Opus帧)...")
                
                if opuslib_available:
                    # 方法1: 使用opuslib解码Opus帧
                    wav_path = os.path.join("tmp", f"{base_filename}.wav")
                    if opus_frames_to_wav(opus_frames, wav_path):
                        # WAV转换成功
                        if ffmpeg_available:
                            # 转换为MP3
                            mp3_path = os.path.join("tmp", f"{base_filename}.mp3")
                            if await convert_wav_to_mp3(wav_path, mp3_path):
                                # MP3转换成功，删除临时WAV
                                try:
                                    os.remove(wav_path)
                                    print(f"🗑️ 已删除临时WAV文件: {wav_path}")
                                except OSError:
                                    pass
                            else:
                                print(f"💔 MP3转换失败，保留WAV文件: {wav_path}")
                        else:
                            print(f"ℹ️ FFmpeg不可用，仅保存WAV文件: {wav_path}")
                    else:
                        print("💔 Opus解码失败")
                else:
                    # 方法2: 保存原始数据作为后备
                    raw_path = os.path.join("tmp", f"{base_filename}_raw.dat")
                    with open(raw_path, "wb") as f:
                        for frame in opus_frames:
                            f.write(frame)
                    print(f"💾 已保存原始Opus帧数据: {raw_path}")
                    print("ℹ️ 安装 opuslib-next 以启用音频解码功能")
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
    print("将合并每个对话的音频并转换为MP3")
    print("使用固定MAC地址: 3E:8A:F1:6C:2D:B5")
    print("=" * 50)
    
    asyncio.run(compatible_test()) 