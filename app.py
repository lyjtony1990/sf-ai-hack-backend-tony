import uvicorn
from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from index_manager import IndexManager

manager = IndexManager()
app = FastAPI()


# Model for request body
class WriteRequest(BaseModel):
    query: str
    answer: str


# CORS setup
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{index}/respond")
async def get_response(index: str, question: str):
    response = manager.get_response(index, question)
    # Convert non-serializable parts to serializable format
    return {"response": response}


@app.post("/{index}/write/")
async def write(index: str = Path(..., description="Index to write to"),
                request: WriteRequest = None):
    if request is None:
        raise HTTPException(status_code=400, detail="Request body is required")

    response = manager.write_to_notion(index, request.query, request.answer)
    return {"status": "write successful"}


@app.get("/", tags=["Root"])
async def read_root():
    # drive.list_files()
    # ingestion.get_response()
    return {"message": "This is working"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
