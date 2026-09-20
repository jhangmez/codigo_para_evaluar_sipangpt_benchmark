"""
Clientes de inferencia para interactuar con Gemma-4 Fine-Tuned y SipánGPT RAG (STAIR).
"""

from sipangpt_eval.clients.base import BaseLLMClient
from sipangpt_eval.clients.local_gemma import LocalGemmaClient
from sipangpt_eval.clients.sipan_rag import SipanRAGClient

__all__ = [
    "BaseLLMClient",
    "LocalGemmaClient",
    "SipanRAGClient",
]
