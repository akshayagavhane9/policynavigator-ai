from fastapi import APIRouter, UploadFile, File
from typing import List
import os
from src.main import ingest_and_index_documents, KB_RAW_DIR
from src.api.models.ingest_models import IngestResponse

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/index", response_model=IngestResponse)
async def index_documents(files: List[UploadFile] = File(...)):
    os.makedirs(KB_RAW_DIR, exist_ok=True)
    saved_paths = []
    filenames: List[str] = []

    for uf in files:
        dest_path = os.path.join(KB_RAW_DIR, uf.filename)
        with open(dest_path, "wb") as out:
            content = await uf.read()
            out.write(content)
        saved_paths.append(dest_path)
        filenames.append(uf.filename)

    chunks_indexed = ingest_and_index_documents(saved_paths)

    # Make sure chunks_indexed is a real number
    if chunks_indexed is None:
        chunks_indexed = 0

    return IngestResponse(
        message=f"Indexed {chunks_indexed} chunks from {len(saved_paths)} file(s).",
        files_indexed=len(saved_paths),
        chunks_indexed=chunks_indexed,
        filenames=filenames,
    )

