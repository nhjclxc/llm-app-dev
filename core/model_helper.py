#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2026/9/21 16:37
# Module    : model_helper.py
# explain   : 在这个文件里面创建llm或者chat客户端

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv(Path(__file__).with_name(".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
model_chat = "qwen-plus"
model_chat = "qwen3.6-plus"

# 调用千问的语言模型
# https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=3016807

HTTPS_PROXY = os.getenv("HTTPS_PROXY")
if HTTPS_PROXY:
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY
# v2RayN
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
# clash
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

def get_llm_openai(base_url=base_url, api_key=qwen_api_key) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

def get_chat_openai(model=model_chat, temperature=0.7, base_url=base_url, api_key=qwen_api_key,
            streaming: bool = False, callbacks: list[BaseCallbackHandler]=None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        streaming=streaming,
        callbacks=callbacks
    )
# 这是一个用于大模型应用开发的仓库，现在我不知道如何进行学习，要求你给我一个完整的大模型应用开发学习路线，文档要求分为多个阶段，我现在是0基础，但是已经有python基础了，大模型应用开发文档输出位置；llm-app-roadmap.md

# model_embed = "text-embedding-v3"
model_embed = "text-embedding-v4"

def get_embedding_model(model=model_embed, api_key=qwen_api_key) -> Embeddings:
    return DashScopeEmbeddings(
    model=model,
    dashscope_api_key=api_key
)
