"""Document / knowledge base endpoints."""

import json
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File
from backend.models.schemas import DocumentInfo, DocumentUploadResponse
from backend.config import DOCUMENTS_DIR

router = APIRouter()

_index_file = DOCUMENTS_DIR / "_index.json"


def _load_index() -> dict:
    if _index_file.exists():
        return json.loads(_index_file.read_text())
    return {"documents": []}


def _save_index(index: dict):
    _index_file.write_text(json.dumps(index, indent=2))


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    index = _load_index()
    return [DocumentInfo(**d) for d in index.get("documents", [])]


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        dest = DOCUMENTS_DIR / file.filename
        dest.write_bytes(content)
        size_kb = round(len(content) / 1024, 1)

        doc = DocumentInfo(
            name=file.filename,
            path=str(dest),
            size_kb=size_kb,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            status="indexed",
            chunks=0,
        )

        index = _load_index()
        index["documents"].append(doc.model_dump())
        _save_index(index)

        return DocumentUploadResponse(success=True, document=doc)
    except Exception as e:
        return DocumentUploadResponse(success=False, error=str(e))


@router.delete("/{filename}")
async def delete_document(filename: str):
    dest = DOCUMENTS_DIR / filename
    if dest.exists():
        dest.unlink()
    index = _load_index()
    index["documents"] = [d for d in index["documents"] if d["name"] != filename]
    _save_index(index)
    return {"ok": True}
