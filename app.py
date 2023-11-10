import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from index_manager import IndexManager

manager = IndexManager()
app = FastAPI()

# CORS setup
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{index}/process")
async def process_data(index: str, data_request: str):
    response = manager.process(index, data_request)
    return {"response": response}


@app.get("/{index}/respond")
async def get_response(index: str, question: str):
    response = manager.get_response(index, question)
    # Convert non-serializable parts to serializable format
    return {"response": response}


@app.get("/", tags=["Root"])
async def read_root():
    # drive.list_files()
    # ingestion.get_response()
    return {"message": "This is working"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
