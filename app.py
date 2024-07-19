import sys
print(sys.executable)
print(sys.path)
import openai
print(openai.__file__)

import streamlit as st
import pyaudio
import wave
from openai import OpenAI
import time
import numpy as np

# OpenAIクライアントの初期化
client = OpenAI(api_key=st.secrets["secrets"]["OPENAI_API_KEY"])

# Streamlitの設定
st.title("リアルタイム音声認識")
st.write("マイクから音声をリアルタイムで認識し、文字起こしを行います。")

# PyAudioの設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()

# 音声デバイスの選択
device_index = st.selectbox("オーディオデバイスを選択してください", 
                            [i for i in range(audio.get_device_count())], 
                            format_func=lambda x: audio.get_device_info_by_index(x)['name'])

# 無音検出関数
def is_silent(data_chunk, thresh=300):
    return np.max(np.abs(np.frombuffer(data_chunk, dtype=np.int16))) < thresh

# 音声をファイルに保存し、文字起こしを行う関数
def save_and_transcribe(filename):
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
    frames = []
    silent_chunks = 0
    for _ in range(0, int(RATE / CHUNK * 5)):  # 5秒間録音
        data = stream.read(CHUNK)
        if is_silent(data):
            silent_chunks += 1
        frames.append(data)

    stream.stop_stream()
    stream.close()

    # 全チャンクの80%以上が無音なら、文字起こしをスキップ
    if silent_chunks > 0.8 * len(frames):
        return ""

    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    with open(filename, "rb") as audio_file:
        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return response.text

# リアルタイムで音声認識を行うための関数
def recognize_audio():
    while True:
        text = save_and_transcribe("output.wav")
        if text:  # 空文字列でない場合のみ表示
            st.write(text)
        time.sleep(1)  # 1秒間隔で繰り返し

# 音声認識を開始するボタン
if st.button("音声認識を開始"):
    recognize_audio()


# import streamlit as st
# import pyaudio
# import wave
# import openai
# import asyncio

# # OpenAIのAPIキーを設定
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# # Streamlitの設定
# st.title("リアルタイム音声認識")
# st.write("マイクから音声をリアルタイムで認識し、文字起こしを行います。")

# # PyAudioの設定
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# CHUNK = 1024

# audio = pyaudio.PyAudio()

# # 音声デバイスの選択
# device_index = st.selectbox("オーディオデバイスを選択してください", 
#                             [i for i in range(audio.get_device_count())], 
#                             format_func=lambda x: audio.get_device_info_by_index(x)['name'])

# # 音声をファイルに保存し、文字起こしを行う関数
# def save_and_transcribe(filename):
#     stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
#     frames = []
#     for _ in range(0, int(RATE / CHUNK * 5)):  # 5秒間録音
#         data = stream.read(CHUNK)
#         frames.append(data)

#     wave_file = wave.open(filename, 'wb')
#     wave_file.setnchannels(CHANNELS)
#     wave_file.setsampwidth(audio.get_sample_size(FORMAT))
#     wave_file.setframerate(RATE)
#     wave_file.writeframes(b''.join(frames))
#     wave_file.close()

#     stream.stop_stream()
#     stream.close()

#     with open(filename, "rb") as audio_file:
#         transcript = openai.Audio.transcriptions.create(model="whisper-1", file=audio_file)
#     return transcript['text']

# # リアルタイムで音声認識を行うための関数
# async def recognize_audio():
#     while True:
#         text = save_and_transcribe("output.wav")
#         st.write(text)
#         await asyncio.sleep(1)  # 1秒間隔で繰り返し

# # 音声認識を開始するボタン
# if st.button("音声認識を開始"):
#     asyncio.run(recognize_audio())
