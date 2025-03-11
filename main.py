from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Movie Search API",
    description="Search for movies using semantic similarity",
    version="1.0.0"
)

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

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# Initialize the SentenceTransformer model (load it only once at startup)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define response model for better documentation and type checking
class MovieSearchResult(BaseModel):
    title: str
    overview: str
    imdb_id: str
    similarity_score: float

class SearchResponse(BaseModel):
    movies: List[MovieSearchResult]
    query: str

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
        query_embedding = model.encode([query])
        normalized_query = normalize(query_embedding, axis=1, norm='l2')
        
        # Search in Qdrant
        results = qdrant_client.search(
            collection_name='movies',
            query_vector=normalized_query[0].tolist(),
            limit=top_k
        )
        
        # Format the results
        movies = []
        for result in results:
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
