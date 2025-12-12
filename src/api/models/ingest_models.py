from pydantic import BaseModel
from typing import List


class IngestResponse(BaseModel):
    message: str
    files_indexed: int
    chunks_indexed: int
    filenames: List[str]
