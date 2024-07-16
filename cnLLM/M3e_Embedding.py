from __future__ import annotations

import logging
import os
from typing import Dict, List, Any

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu" # 实际情况无GPU

logger = logging.getLogger(__name__)

class M3EEmbeddings(BaseModel, Embeddings):
    """`M3E Embeddings` embedding models."""

    client: Any

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化M3E为values["client"]
        Args:
            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:
            values (Dict): 包含配置信息的字典。如果环境中有M3E库，则将返回实例化的M3E类；否则将报错 'ModuleNotFoundError: No module named 'M3E''.
        """
        # 下载或加载模型，原路径为./.cache/huggingface/hub/models--moka-ai--m3e-base
        # values["client"] = SentenceTransformer('moka-ai/m3e-base')#, cache_folder=r"./.cache/huggingface/hub")#.cuda() 实际情况无GPU
        # values["client"] = SentenceTransformer('./.cache/huggingface/hub/models--moka-ai--m3e-base', cache_folder='./.cache/huggingface/hub/models--moka-ai--m3e-base')
        path = 'models\\models--moka-ai--m3e-base\\snapshots\\764b537a0e50e5c7d64db883f2d2e051cbe3c64c'
        print('Loading m3e text model from', path)
        values["client"] = SentenceTransformer(path).to(device)
        return values
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.encode(text)
        return embeddings.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.
        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in tqdm(texts, desc='嵌入进度')]
    
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")