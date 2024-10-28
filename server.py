import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import google.generativeai as genai
from contextlib import asynccontextmanager
from functools import lru_cache
import logging

@lru_cache()
def load_prompt(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    google_api_key: str
    allowed_hosts: list
    gemini_model: str
    rate_limit: str = "5/minute"
    max_workers: int = 10

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
executor = ThreadPoolExecutor(max_workers=settings.max_workers)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    app.state.BASE_PROMPT = load_prompt('base_p.txt')
    app.state.ITER_PROMPT = load_prompt('iterative_p.txt')
    yield
    logger.info("Application shutting down...")
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_hosts, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(SlowAPIMiddleware)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

genai.configure(api_key=settings.google_api_key)

class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)

class QueryResponse(BaseModel):
    response: str

def query_gemini(prompt: str):
    try:
        model = genai.GenerativeModel(settings.gemini_model)
        base_prompt = app.state.BASE_PROMPT
        generated_plan = model.generate_content(base_prompt.replace(r"{USER_PROMPT}", prompt)).text
        plan_list = generated_plan.split('\n')
        iter_prompt = app.state.ITER_PROMPT

        for plan in plan_list:
            if plan:
                generated_content = model.generate_content(iter_prompt.replace(r"{PROMPT_CONTENT}", plan[0:-4]).replace(r"{WORD_COUNT}", plan[-4:])).text
                print(generated_content, end="\n")
                yield generated_content + "<br /><br />"  # Yield content with line breaks

    except Exception as e:
        logger.error(f"Error querying Gemini: {e}")
        yield " "


@app.post("/query")
@limiter.limit(settings.rate_limit)
async def query(request: Request, query_request: QueryRequest):
    try:
        async def event_generator():
            for content in query_gemini(query_request.prompt):
                yield content

        return StreamingResponse(event_generator(), media_type="text/html")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.head("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
