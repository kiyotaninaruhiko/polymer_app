"""
Polymer SMILES Descriptor Generator - Backend API

FastAPI application providing REST API for molecular descriptor generation.
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routers import providers, parse, descriptors, atom_attribution
from providers.registry import register_all_providers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize providers on startup"""
    print("Initializing descriptor providers...")
    register_all_providers()
    print("Providers initialized successfully")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Polymer Descriptor API",
    description="REST API for generating molecular descriptors from polymer SMILES",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(providers.router)
app.include_router(parse.router)
app.include_router(descriptors.router)
app.include_router(atom_attribution.router)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Polymer Descriptor API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "providers": "/api/providers",
            "parse": "/api/parse",
            "descriptors": "/api/descriptors/generate",
            "atom_attribution": "/api/atom-attribution"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
