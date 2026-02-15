"""
Abstracción sobre backends de LLM (Ollama local / OpenAI API).

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMProvider:
    """Proveedor de LLM con soporte para Ollama y OpenAI."""

    def __init__(
        self,
        backend: str = "ollama",
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.backend = LLMBackend(backend)
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        self._init_client()

    def _init_client(self) -> None:
        """Inicializa el cliente según el backend."""
        if self.backend == LLMBackend.OLLAMA:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
                logger.info(f"Ollama client inicializado: {self.model_name} @ {self.base_url}")
            except ImportError:
                logger.warning("Paquete 'ollama' no instalado. Instalar con: pip install ollama")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Ollama: {e}")

        elif self.backend == LLMBackend.OPENAI:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client inicializado: {self.model_name}")
            except ImportError:
                logger.warning("Paquete 'openai' no instalado. Instalar con: pip install openai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Genera texto a partir de un prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._call_llm(messages)

    def generate_with_history(
        self, messages: list[dict], system_prompt: Optional[str] = None
    ) -> str:
        """Genera texto con historial de conversación."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        return self._call_llm(full_messages)

    def _call_llm(self, messages: list[dict]) -> str:
        """Llama al LLM según el backend configurado."""
        if self.backend == LLMBackend.OLLAMA:
            return self._call_ollama(messages)
        elif self.backend == LLMBackend.OPENAI:
            return self._call_openai(messages)
        else:
            raise ValueError(f"Backend no soportado: {self.backend}")

    def _call_ollama(self, messages: list[dict]) -> str:
        """Llama a Ollama."""
        if self._client is None:
            return "[Error: Ollama no está disponible. Verificar que el servidor esté corriendo.]"

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error Ollama: {e}")
            return f"[Error al generar respuesta con Ollama: {e}]"

    def _call_openai(self, messages: list[dict]) -> str:
        """Llama a OpenAI API."""
        if self._client is None:
            return "[Error: OpenAI client no inicializado. Verificar API key.]"

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error OpenAI: {e}")
            return f"[Error al generar respuesta con OpenAI: {e}]"

    def is_available(self) -> bool:
        """Verifica si el LLM está disponible."""
        try:
            response = self.generate("Respondé con 'ok'.")
            return "error" not in response.lower()
        except Exception:
            return False
