from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import google.generativeai as genai

from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Movie Search API",
    description="Search for movies using semantic similarity",
    version="1.0.0"
)

# Serve the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import FileResponse
    return FileResponse("static/favicon.ico")

# Add CORS middleware to allow cross-origin requests (important for web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_MOVIE")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# Define response model for better documentation and type checking
class MovieSearchResult(BaseModel):
    title: str
    overview: str
    imdb_id: str
    similarity_score: float

class SearchResponse(BaseModel):
    movies: List[MovieSearchResult]
    query: str

def get_gemini_embeddings(texts:List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model="models/text-embedding-004", 
            content=text)
        embeddings.append(response['embedding'])
    return embeddings

@app.get("/")
async def root():
    return {
        "message": "Welcome to Movie Search API!",
        "docs": "/docs",
        "usage": "Send GET requests to /search?query=your search text"
    }

@app.get("/search", response_model=SearchResponse)
async def search_movies(
    query: str = Query(..., description="The search query to find similar movies"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return")
):
    try:
        # Encode the query text
        query_embeddings = get_gemini_embeddings([query])

        results = qdrant_client.query_points(
            collection_name = 'movies_gemini',
            query=query_embeddings[0],
            limit=top_k
        )
        
        # Format the results
        movies = []
        for result in results.points:
            movies.append(MovieSearchResult(
                title=result.payload['title'],
                overview=result.payload['overview'],
                imdb_id=result.payload['imdb_id'],
                similarity_score=result.score
            ))
        
        return SearchResponse(
            movies=movies,
            query=query
        )
    
    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running"""
    return {"status": "healthy"}
