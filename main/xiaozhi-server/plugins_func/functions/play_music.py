from config.logger import setup_logging
import os
import re
import time
import random
import asyncio
import json
import difflib
import traceback
from pathlib import Path
from core.utils import p3
from core.handle.sendAudioHandle import send_stt_message
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.utils.dialogue import Message

TAG = __name__

MUSIC_CACHE = {}

play_music_function_desc = {
    "type": "function",
    "function": {
        "name": "play_music",
        "description": "唱歌、听歌、播放音乐的方法。",
        "parameters": {
            "type": "object",
            "properties": {
                "song_name": {
                    "type": "string",
                    "description": "歌曲名称，如果用户没有指定具体歌名则为'random', 明确指定的时返回音乐的名字 示例: ```用户:播放两只老虎\n参数：两只老虎``` ```用户:播放音乐 \n参数：random ```",
                }
            },
            "required": ["song_name"],
        },
    },
}


@register_function("play_music", play_music_function_desc, ToolType.SYSTEM_CTL)
def play_music(conn, song_name: str):
    try:
        music_intent = (
            f"播放音乐 {song_name}" if song_name != "random" else "随机播放音乐"
        )

        # 检查事件循环状态
        if not conn.loop.is_running():
            conn.logger.bind(tag=TAG).error("事件循环未运行，无法提交任务")
            return ActionResponse(
                action=Action.RESPONSE, result="系统繁忙", response="请稍后再试"
            )

        # 提交异步任务
        future = asyncio.run_coroutine_threadsafe(
            handle_music_command(conn, music_intent), conn.loop
        )

        # 非阻塞回调处理
        def handle_done(f):
            try:
                f.result()  # 可在此处理成功逻辑
                conn.logger.bind(tag=TAG).info("播放完成")
            except Exception as e:
                conn.logger.bind(tag=TAG).error(f"播放失败: {e}")

        future.add_done_callback(handle_done)

        return ActionResponse(
            action=Action.NONE, result="指令已接收", response="正在为您播放音乐"
        )
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"处理音乐意图错误: {e}")
        return ActionResponse(
            action=Action.RESPONSE, result=str(e), response="播放音乐时出错了"
        )


def _extract_song_name(text):
    """从用户输入中提取歌名"""
    for keyword in ["播放音乐"]:
        if keyword in text:
            parts = text.split(keyword)
            if len(parts) > 1:
                return parts[1].strip()
    return None


def _find_best_match(potential_song, music_files):
    """查找最匹配的歌曲"""
    best_match = None
    highest_ratio = 0

    for music_file in music_files:
        song_name = os.path.splitext(music_file)[0]
        ratio = difflib.SequenceMatcher(None, potential_song, song_name).ratio()
        if ratio > highest_ratio and ratio > 0.4:
            highest_ratio = ratio
            best_match = music_file
    return best_match


def get_music_files(music_dir, music_ext):
    music_dir = Path(music_dir)
    music_files = []
    music_file_names = []
    for file in music_dir.rglob("*"):
        # 判断是否是文件
        if file.is_file():
            # 获取文件扩展名
            ext = file.suffix.lower()
            # 判断扩展名是否在列表中
            if ext in music_ext:
                # 添加相对路径
                music_files.append(str(file.relative_to(music_dir)))
                music_file_names.append(
                    os.path.splitext(str(file.relative_to(music_dir)))[0]
                )
    return music_files, music_file_names


def initialize_music_handler(conn):
    global MUSIC_CACHE
    if MUSIC_CACHE == {}:
        if "play_music" in conn.config["plugins"]:
            MUSIC_CACHE["music_config"] = conn.config["plugins"]["play_music"]
            MUSIC_CACHE["music_dir"] = os.path.abspath(
                MUSIC_CACHE["music_config"].get("music_dir", "./music")  # 默认路径修改
            )
            MUSIC_CACHE["music_ext"] = MUSIC_CACHE["music_config"].get(
                "music_ext", (".mp3", ".wav", ".p3")
            )
            MUSIC_CACHE["refresh_time"] = MUSIC_CACHE["music_config"].get(
                "refresh_time", 60
            )
        else:
            MUSIC_CACHE["music_dir"] = os.path.abspath("./music")
            MUSIC_CACHE["music_ext"] = (".mp3", ".wav", ".p3")
            MUSIC_CACHE["refresh_time"] = 60
        # 获取音乐文件列表
        MUSIC_CACHE["music_files"], MUSIC_CACHE["music_file_names"] = get_music_files(
            MUSIC_CACHE["music_dir"], MUSIC_CACHE["music_ext"]
        )
        MUSIC_CACHE["scan_time"] = time.time()
    return MUSIC_CACHE


async def handle_music_command(conn, text):
    initialize_music_handler(conn)
    global MUSIC_CACHE

    """处理音乐播放指令"""
    clean_text = re.sub(r"[^\w\s]", "", text).strip()
    conn.logger.bind(tag=TAG).debug(f"检查是否是音乐命令: {clean_text}")

    # 尝试匹配具体歌名
    if os.path.exists(MUSIC_CACHE["music_dir"]):
        if time.time() - MUSIC_CACHE["scan_time"] > MUSIC_CACHE["refresh_time"]:
            # 刷新音乐文件列表
            MUSIC_CACHE["music_files"], MUSIC_CACHE["music_file_names"] = (
                get_music_files(MUSIC_CACHE["music_dir"], MUSIC_CACHE["music_ext"])
            )
            MUSIC_CACHE["scan_time"] = time.time()

        potential_song = _extract_song_name(clean_text)
        if potential_song:
            best_match = _find_best_match(potential_song, MUSIC_CACHE["music_files"])
            if best_match:
                conn.logger.bind(tag=TAG).info(f"找到最匹配的歌曲: {best_match}")
                await play_local_music(conn, specific_file=best_match)
                return True
    # 检查是否是通用播放音乐命令
    await play_local_music(conn)
    return True


def _get_random_play_prompt(song_name):
    """生成随机播放引导语"""
    # 移除文件扩展名
    clean_name = os.path.splitext(song_name)[0]
    prompts = [
        f"正在为您播放，{clean_name}",
        f"请欣赏歌曲，{clean_name}",
        f"即将为您播放，{clean_name}",
        f"为您带来，{clean_name}",
        f"让我们聆听，{clean_name}",
        f"接下来请欣赏，{clean_name}",
        f"为您献上，{clean_name}",
    ]
    # 直接使用random.choice，不设置seed
    return random.choice(prompts)


async def play_local_music(conn, specific_file=None):
    global MUSIC_CACHE
    """播放本地音乐文件"""
    try:
        if not os.path.exists(MUSIC_CACHE["music_dir"]):
            conn.logger.bind(tag=TAG).error(f"音乐目录不存在: {MUSIC_CACHE['music_dir']}")
            return

        # 1. 选择音乐文件
        if specific_file:
            selected_music = specific_file
            music_path = os.path.join(MUSIC_CACHE["music_dir"], specific_file)
        else:
            if not MUSIC_CACHE["music_files"]:
                conn.logger.bind(tag=TAG).error("未找到任何音乐文件")
                return
            selected_music = random.choice(MUSIC_CACHE["music_files"])
            music_path = os.path.join(MUSIC_CACHE["music_dir"], selected_music)

        if not os.path.exists(music_path):
            conn.logger.bind(tag=TAG).error(f"选定的音乐文件不存在: {music_path}")
            return
            
        # 2. 提前处理音乐文件获取时长和Opus数据
        opus_packets_song = None
        duration = 0
        if music_path.endswith(".p3"):
            opus_packets_song, duration = p3.decode_opus_from_file(music_path)
        else:
            opus_packets_song, duration = conn.tts.audio_to_opus_data(music_path)

        if not opus_packets_song:
            conn.logger.bind(tag=TAG).error(f"音乐文件转换Opus失败: {music_path}")
            # Optional: send TTS error message to user
            return

        # 3. 发送包含时长的LLM消息
        session_id = conn.session_id
        song_name_for_client = os.path.splitext(selected_music)[0]
        llm_message_data = {
            "type": "llm",
            "text": "🎵",
            "emotion": "happy",
            "session_id": session_id,
            "motion_data": {
                "motion": "播放音乐",
                "song_name": song_name_for_client,
                "expression": "happy",
                "duration": duration,
            }
        }
        message_json = json.dumps(llm_message_data, ensure_ascii=False)
        conn.logger.bind(tag=TAG).info(f"发送播放音乐LLM消息到客户端: {message_json}")
        await conn.websocket.send(message_json)
        
        # 4. 生成并播放引导语
        text = _get_random_play_prompt(selected_music)
        await send_stt_message(conn, text)
        conn.dialogue.put(Message(role="assistant", content=text))
        conn.tts_first_text_index = 0
        conn.tts_last_text_index = 0

        tts_file = await asyncio.to_thread(conn.tts.to_tts, text)
        if tts_file is not None and os.path.exists(tts_file):
            opus_packets_intro, _ = conn.tts.audio_to_opus_data(tts_file)
            if opus_packets_intro:
                conn.audio_play_queue.put((opus_packets_intro, None, 0, None))
                conn.tts_last_text_index = 1
            os.remove(tts_file)

        # 5. 将音乐Opus数据放入播放队列
        conn.llm_finish_task = True
        conn.audio_play_queue.put((opus_packets_song, None, conn.tts_last_text_index, None))
        conn.logger.bind(tag=TAG).info(f"已将歌曲《{selected_music}》的Opus数据放入播放队列")

    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"播放音乐失败: {str(e)}")
        conn.logger.bind(tag=TAG).error(f"详细错误: {traceback.format_exc()}")
