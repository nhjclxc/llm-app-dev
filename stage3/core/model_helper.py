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

from openai import OpenAI


load_dotenv(Path(__file__).with_name(".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")
model_chat = os.getenv("QWEN_MODEL")
base_url = os.getenv("BASE_URL")
embedding_model = os.getenv("EMBEDDING_MODEL")

# 调用千问的语言模型
# https://bailian.console.aliyun.com/cn-beijing/costing-balance/free-quota
# https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=3016807
if not qwen_api_key:
    sys.exit("未找到 QWEN_API_KEY：请在仓库根目录的 .env 中配置它。")


HTTPS_PROXY = os.getenv("HTTPS_PROXY")
if HTTPS_PROXY:
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY
# v2RayN
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
# clash
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


def get_openai_client(api_key: str = qwen_api_key, base_url: str = base_url) -> OpenAI:
    """ 获取 openai 官方的客户端对象 """
    return OpenAI(
        # 客户端请求持有的 Key
        api_key=api_key,
        # 请求将发送到哪个 API 根地址
        base_url=base_url,
        # 客户端级默认超时
        timeout=30.0,
        # 控制 SDK 层面的自动重试次数。
        max_retries=3,
    )


