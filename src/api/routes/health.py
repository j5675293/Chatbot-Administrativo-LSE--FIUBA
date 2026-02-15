"""
Endpoints de salud y estado del sistema.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse, GraphStats, DocumentInfo
from src.api.dependencies import AppDependencies, get_dependencies

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(deps: AppDependencies = Depends(get_dependencies)):
    """Estado de salud del sistema."""
    llm_ok = deps.llm_provider.is_available() if deps.llm_provider else False
    index_size = (
        deps.vector_store.index.ntotal
        if deps.vector_store and deps.vector_store.index
        else 0
    )
    graph_nodes = (
        deps.graph_builder.graph.number_of_nodes()
        if deps.graph_builder
        else 0
    )
    docs = len(deps.pipeline.get_all_metadata()) if deps.pipeline else 0

    return HealthResponse(
        status="ok",
        llm_available=llm_ok,
        documents_loaded=docs,
        index_size=index_size,
        graph_nodes=graph_nodes,
    )


@router.get("/graph/stats", response_model=GraphStats)
async def graph_stats(deps: AppDependencies = Depends(get_dependencies)):
    """Estadísticas del grafo de conocimiento."""
    stats = deps.graph_builder.get_statistics()
    return GraphStats(**stats)


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(deps: AppDependencies = Depends(get_dependencies)):
    """Lista los documentos procesados."""
    metadata_list = deps.pipeline.get_all_metadata()
    docs = []
    for meta in metadata_list:
        docs.append(DocumentInfo(
            filename=meta.get("filename", ""),
            document_type=meta.get("document_type", ""),
            program_codes=meta.get("program_codes", []),
            topics=meta.get("topics", []),
        ))
    return docs
