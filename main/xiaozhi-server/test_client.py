#!/usr/bin/env python3
"""
xiaozhi-server 测试客户端
功能：向xiaozhi-server发送指定格式的消息并接收音频回复
将合并每个对话的音频并转换为MP3
"""

import asyncio
import websockets
import json
import uuid
import time
import os
import random
import subprocess
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

class XiaozhiTestClient:
    def __init__(self, server_url="ws://127.0.0.1:8000/xiaozhi/v1/"):
        self.server_url = server_url
        self.websocket = None
        self.session_id = str(uuid.uuid4()) 
        self.device_id = generate_mac_address()
        self.client_id = str(uuid.uuid4())
        self.ffmpeg_available = check_ffmpeg()
        self.opuslib_available = check_opuslib()
        
    def is_connected(self):
        """检查WebSocket连接是否处于打开状态（兼容不同版本）"""
        if self.websocket is None:
            return False
        
        # 尝试不同的连接状态检查方法，按兼容性优先级
        try:
            # 方法1: 检查 state 属性 (新版本)
            if hasattr(self.websocket, 'state'):
                from websockets.connection import State
                return self.websocket.state == State.OPEN
        except (AttributeError, ImportError):
            pass
            
        try:
            # 方法2: 检查 open 属性 (legacy 版本)
            if hasattr(self.websocket, 'open'):
                return self.websocket.open
        except AttributeError:
            pass
            
        try:
            # 方法3: 检查 protocol.state (某些版本)
            if hasattr(self.websocket, 'protocol') and hasattr(self.websocket.protocol, 'state'):
                from websockets.connection import State  
                return self.websocket.protocol.state == State.OPEN
        except (AttributeError, ImportError):
            pass
            
        # 如果所有方法都失败，返回 True 并依赖异常处理
        return True

    async def connect(self):
        """连接到websocket服务器"""
        try:
            headers = {
                "device-id": self.device_id,
                "client-id": self.client_id, 
                "protocol-version": "1",
                "Authorization": "Bearer test-token"
            }
            print(f"ℹ️  使用请求头: {headers}")
            
            try:
                self.websocket = await websockets.connect(self.server_url, extra_headers=headers)
            except TypeError:
                self.websocket = await websockets.connect(self.server_url, additional_headers=headers)
            
            print(f"✅ 已连接到服务器: {self.server_url}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
            
    async def send_hello(self):
        """发送hello消息"""
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
        try:
            await self.websocket.send(json.dumps(hello_message))
            print("📤 已发送hello消息")
        except websockets.exceptions.ConnectionClosed:
            print("‼️ 发送hello时连接已关闭。")
            raise
        
    async def send_text_message(self, session_id, text):
        """发送文本消息"""
        message = {
            "session_id": session_id, 
            "type": "listen",
            "state": "detect",
            "text": text,
            "source": "text"
        }
        try:
            await self.websocket.send(json.dumps(message))
            print(f"📤 已发送文本消息 (会话: {session_id}): {text}")
        except websockets.exceptions.ConnectionClosed:
            print(f"‼️ 发送文本消息时连接已关闭 (会话: {session_id})。")
            raise

    async def handle_text_message_response(self, message_text):
        """处理收到的文本消息（JSON）"""
        try:
            data = json.loads(message_text)
            msg_type = data.get("type", "unknown")
            session_id_recv = data.get("session_id")
            
            print(f"💬 收到消息 (类型: {msg_type}, 会话ID: {session_id_recv}):")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            return data         
        except json.JSONDecodeError:
            print(f"⚠️  非JSON消息: {message_text}")
            return None 
            
    async def test_conversation(self, text_to_send):
        """测试单个完整对话，合并音频并转换为MP3"""
        current_session_id = "test_conv_" + str(uuid.uuid4())[:8]
        print(f"\n=== ▶️ 开始新对话 (会话ID: {current_session_id}) ===")
        print(f"💬 发送内容: {text_to_send}")
        
        try:
            await self.send_text_message(current_session_id, text_to_send)
        except websockets.exceptions.ConnectionClosed:
            print("‼️ 发送消息时连接已断开")
            return
        
        print("⏳ 等待服务器回复...")
        
        # 收集Opus帧数据（bytes列表）
        opus_frames = []
        message_count = 0
        tts_completed = False
        
        while not tts_completed:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=45.0) 
                message_count += 1
                
                if isinstance(message, bytes):
                    print(f"🎵 [{message_count}] 收到 Opus 音频帧: {len(message)} 字节")
                    opus_frames.append(message)  # 保存为独立的帧
                else:
                    parsed_data = await self.handle_text_message_response(message)
                    if parsed_data and parsed_data.get("type") == "tts" and parsed_data.get("state") == "stop":
                        print("🔇 [{message_count}] TTS完成 - 对话结束")
                        tts_completed = True
                        
            except asyncio.TimeoutError:
                print("⏰ 等待回复超时")
                break 
            except websockets.exceptions.ConnectionClosed as e:
                print(f"🔌 连接在对话中关闭: {e}")
                tts_completed = True 
                break
            except Exception as e:
                print(f"💥 处理消息时发生意外错误: {e}")
                tts_completed = True 
                break
                
        # 处理音频数据
        if len(opus_frames) > 0:
            base_filename = f"conversation_{current_session_id}_{int(time.time())}"
            
            print(f"\n🔊 处理 \"{current_session_id}\" 的音频数据 ({len(opus_frames)} 个Opus帧)..." )
            
            if self.opuslib_available:
                # 方法1: 使用opuslib解码Opus帧
                wav_path = os.path.join("tmp", f"{base_filename}.wav")
                if opus_frames_to_wav(opus_frames, wav_path):
                    # WAV转换成功
                    if self.ffmpeg_available:
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
            print(f"🤷 会话 \"{current_session_id}\" 未收到任何音频数据。" )
            
        print(f"===⏹️ 对话结束 (会话ID: {current_session_id}) ===\n")
        
    async def close_connection(self):
        """关闭连接（如果已连接）"""
        if self.websocket is not None:
            try:
                await self.websocket.close()
                print("🔌 连接已关闭")
            except websockets.exceptions.ConnectionClosed:
                print("ℹ️ 尝试关闭时连接已经是关闭状态。")
            except Exception as e:
                print(f"💥 关闭连接时出错: {e}")
        else:
            print("ℹ️ 连接本就未初始化。")

async def main():
    """主函数"""
    os.makedirs("tmp", exist_ok=True) 
    client = XiaozhiTestClient()
    
    try:
        if not await client.connect():
            return
        
        await client.send_hello()
        
        try:
            hello_reply = await asyncio.wait_for(client.websocket.recv(), timeout=10.0)
            print("\n🤝 服务器Hello已确认:")
            await client.handle_text_message_response(hello_reply) 
        except asyncio.TimeoutError:
            print("⏰ 等待服务器Hello回复超时，测试中止。")
            await client.close_connection()
            return
        except websockets.exceptions.ConnectionClosed:
            print("🔌 等待Hello回复时连接已关闭，测试中止。")
            return
        
        test_messages = [
            "你好小明，今天怎么样",
            "给我讲一个关于太空的笑话",
            "北京现在几点钟了？"
        ]
        
        for msg_content in test_messages:
            try:
                await client.test_conversation(msg_content)
                print("พักผ่อนสักครู่..." ) 
                await asyncio.sleep(3)
            except websockets.exceptions.ConnectionClosed:
                print(f"‼️ WebSocket 连接已断开，无法进行对话: {msg_content}")
                break
            
    except KeyboardInterrupt:
        print("\n🚦 用户中断测试")
    except Exception as e:
        print(f"💥 测试主流程发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close_connection()

if __name__ == "__main__":
    print("xiaozhi-server 测试客户端")
    print("=" * 50)
    print("将合并每个对话的音频并转换为MP3")
    print("使用固定MAC地址: 3E:8A:F1:6C:2D:B5")
    print("=" * 50)
    asyncio.run(main()) 