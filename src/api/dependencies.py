"""
Dependency injection para FastAPI. Inicializa todos los componentes del sistema.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

import logging
from pathlib import Path
from functools import lru_cache

from config.settings import Settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import FAISSVectorStore
from src.rag.retriever import RAGRetriever, CrossEncoderReranker
from src.rag.rag_chain import RAGChain
from src.graph_rag.entity_extractor import AcademicEntityExtractor
from src.graph_rag.graph_builder import KnowledgeGraphBuilder
from src.graph_rag.graph_retriever import GraphRetriever
from src.hybrid.hybrid_retriever import HybridRetriever
from src.hybrid.answer_synthesizer import AnswerSynthesizer
from src.hybrid.anti_hallucination import AntiHallucinationEngine
from src.hybrid.citation_manager import CitationManager
from src.hybrid.conversation_memory import ConversationMemory
from src.llm.llm_provider import LLMProvider
from src.rag.query_expansion import QueryExpander
from src.rag.hyde import HyDERetriever
from src.evaluation.feedback import FeedbackCollector
from src.data_pipeline.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class AppDependencies:
    """Contenedor de todas las dependencias del sistema."""

    def __init__(self):
        self.settings: Settings = Settings()
        self.embedding_model: EmbeddingModel = None
        self.vector_store: FAISSVectorStore = None
        self.rag_retriever: RAGRetriever = None
        self.rag_chain: RAGChain = None
        self.graph_builder: KnowledgeGraphBuilder = None
        self.graph_retriever: GraphRetriever = None
        self.hybrid_retriever: HybridRetriever = None
        self.answer_synthesizer: AnswerSynthesizer = None
        self.llm_provider: LLMProvider = None
        self.pipeline: PipelineOrchestrator = None
        self.query_expander: QueryExpander = None
        self.hyde_retriever: HyDERetriever = None
        self.conversation_memory: ConversationMemory = None
        self.feedback_collector: FeedbackCollector = None
        self._initialized = False

    def initialize(self) -> None:
        """Inicializa todos los componentes."""
        if self._initialized:
            return

        logger.info("Inicializando componentes del sistema...")

        # LLM
        self.llm_provider = LLMProvider(
            backend=self.settings.LLM_BACKEND,
            model_name=self.settings.LLM_MODEL,
            base_url=self.settings.OLLAMA_BASE_URL,
            api_key=self.settings.OPENAI_API_KEY,
            temperature=self.settings.LLM_TEMPERATURE,
            max_tokens=self.settings.LLM_MAX_TOKENS,
        )

        # Embeddings
        self.embedding_model = EmbeddingModel(
            model_name=self.settings.EMBEDDING_MODEL,
            device=self.settings.EMBEDDING_DEVICE,
        )

        # Vector Store
        self.vector_store = FAISSVectorStore(
            embedding_dim=384,
            index_path=self.settings.INDEX_DIR,
        )

        # Intentar cargar índice existente
        if (self.settings.INDEX_DIR / "faiss_index.bin").exists():
            self.vector_store.load(self.settings.INDEX_DIR)

        # RAG Retriever
        reranker = None
        if self.settings.USE_RERANKER:
            try:
                reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"No se pudo cargar cross-encoder reranker: {e}")

        self.rag_retriever = RAGRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            reranker=reranker,
        )

        # RAG Chain
        self.rag_chain = RAGChain(
            retriever=self.rag_retriever,
            llm_provider=self.llm_provider,
            top_k=self.settings.RAG_TOP_K,
        )

        # Graph
        self.graph_builder = KnowledgeGraphBuilder()
        if (self.settings.GRAPH_DIR / "knowledge_graph.pkl").exists():
            self.graph_builder.load(self.settings.GRAPH_DIR)

        entity_extractor = AcademicEntityExtractor()
        self.graph_retriever = GraphRetriever(
            graph_builder=self.graph_builder,
            entity_extractor=entity_extractor,
        )

        # Hybrid
        self.hybrid_retriever = HybridRetriever(
            rag_retriever=self.rag_retriever,
            graph_retriever=self.graph_retriever,
        )

        # Anti-hallucination
        anti_hallucination = AntiHallucinationEngine(
            llm_provider=self.llm_provider,
            embedding_model=self.embedding_model,
            confidence_threshold=self.settings.CONFIDENCE_THRESHOLD,
            abstention_threshold=self.settings.ABSTENTION_THRESHOLD,
        )

        # Answer Synthesizer
        self.answer_synthesizer = AnswerSynthesizer(
            llm_provider=self.llm_provider,
            anti_hallucination=anti_hallucination,
            citation_manager=CitationManager(),
        )

        # Query Expander
        if self.settings.USE_QUERY_EXPANSION:
            self.query_expander = QueryExpander(
                llm_provider=self.llm_provider,
                embedding_model=self.embedding_model,
                max_expansions=self.settings.MAX_QUERY_EXPANSIONS,
            )

        # HyDE Retriever
        if self.settings.USE_HYDE:
            self.hyde_retriever = HyDERetriever(
                llm_provider=self.llm_provider,
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
                reranker=reranker,
                alpha=self.settings.HYDE_ALPHA,
            )

        # Conversation Memory
        self.conversation_memory = ConversationMemory(
            llm_provider=self.llm_provider,
            window_size=self.settings.CONVERSATION_WINDOW_SIZE,
            max_summary_length=self.settings.MAX_SUMMARY_LENGTH,
        )

        # Feedback Collector
        self.feedback_collector = FeedbackCollector(
            storage_path=Path(self.settings.FEEDBACK_STORAGE_PATH),
        )

        # Pipeline
        self.pipeline = PipelineOrchestrator(
            raw_dir=self.settings.RAW_DATA_DIR,
            processed_dir=self.settings.PROCESSED_DATA_DIR,
        )

        self._initialized = True
        logger.info("Sistema inicializado correctamente")


# Singleton
_deps: AppDependencies = None


def get_dependencies() -> AppDependencies:
    """Obtiene la instancia singleton de dependencias."""
    global _deps
    if _deps is None:
        _deps = AppDependencies()
        _deps.initialize()
    return _deps
