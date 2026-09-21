#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2026/9/21 16:37
# Module    : model_helper.py
# explain   : 在这个文件里面创建llm或者chat客户端

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")

if not qwen_api_key:
    sys.exit("未找到 QWEN_API_KEY：请在仓库根目录的 .env 中配置它。")

# 调用千问的语言模型
# https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=3016807
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model_chat = "qwen3.8-27b"


HTTPS_PROXY = os.getenv("HTTPS_PROXY")
if HTTPS_PROXY:
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY
# v2RayN
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
# clash
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


# def get_chat_openai(model=model_chat, temperature=0.7, base_url=base_url, api_key=qwen_api_key,
#             streaming: bool = False) -> ChatOpenAI:
#     return ChatOpenAI(
#         model=model,
#         temperature=temperature,
#         base_url=base_url,
#         api_key=api_key,
#         streaming=streaming
#     )
