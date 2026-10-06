from fastapi import APIRouter, Depends, HTTPException, Request

from ..engine import AdaptiveRAG
from .deps import get_rag, query_slot
from .schemas import AskRequest, AskResponse
from .security import enforce_ask_rate_limit, require_api_key

router = APIRouter(tags=["query"])


@router.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_ask_rate_limit)],
)
def ask(
    body: AskRequest,
    request: Request,
    rag: AdaptiveRAG = Depends(get_rag),
) -> AskResponse:
    """Answer a question from the indexed documents.

    HTTP 200 is returned for every outcome that is a real answer about the
    documents: check `status` (verified / no_evidence / unverified).
    503 means the system could not answer at all (nothing indexed yet, or the
    language model was unreachable).
    """
    # A plain `def` route runs in FastAPI's thread pool, so the blocking engine
    # call does not freeze the server for other requests.
    with query_slot(request):
        try:
            result = rag.ask_detailed(body.question, rag_mode=body.mode)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

    if result["status"] in {"not_ready", "error"}:
        raise HTTPException(
            status_code=503,
            detail=result["answer"],
            headers={"Retry-After": "10"},
        )

    return AskResponse(**result, request_id=request.state.request_id)