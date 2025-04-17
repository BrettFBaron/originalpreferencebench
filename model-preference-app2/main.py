import asyncio
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from config import logger, QUESTIONS
from db.session import init_db, get_db_session
from db.models import TestingJob
from api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Model Preference API",
    description="API for testing and visualizing model preferences",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {str(e)}")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Include API routes
app.include_router(router, prefix="/api")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page - README"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/questions", response_class=HTMLResponse)
async def questions_page(request: Request):
    """Questions page - shows all questions"""
    return templates.TemplateResponse(
        "questions.html", 
        {"request": request, "questions": QUESTIONS}
    )

@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Models page - shows all available models"""
    # Get all unique model names from the database
    from sqlalchemy import text
    models = []
    
    async with get_db_session() as session:
        result = await session.execute(text("SELECT DISTINCT model_name FROM testing_job"))
        models = [row[0] for row in result.all()]
    
    return templates.TemplateResponse(
        "models.html", 
        {"request": request, "models": models}
    )

@app.get("/submit", response_class=HTMLResponse)
async def submit_form(request: Request):
    """Form to submit a model for testing"""
    return templates.TemplateResponse("submit.html", {"request": request})

@app.get("/processing/{job_id}", response_class=HTMLResponse)
async def processing(request: Request, job_id: int):
    """Processing page - shows progress while model is being tested"""
    model_name = None
    
    async with get_db_session() as session:
        job = await session.get(TestingJob, job_id)
        
        if not job:
            return RedirectResponse(url="/")
            
        model_name = job.model_name
    
    return templates.TemplateResponse(
        "processing.html", 
        {"request": request, "model_name": model_name, "job_id": job_id}
    )

@app.get("/results/{question_id}", response_class=HTMLResponse)
async def results(request: Request, question_id: str):
    """Show results for a specific question"""
    # Get the question text
    question_text = next((q["text"] for q in QUESTIONS if q["id"] == question_id), "Unknown Question")
    
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request, 
            "question_id": question_id, 
            "question_text": question_text,
            "questions": QUESTIONS
        }
    )

@app.get("/raw_data", response_class=HTMLResponse)
async def raw_data_page(request: Request, model_name: str):
    """Display raw JSON data for a specific model using query parameter"""
    return templates.TemplateResponse(
        "raw_data.html", 
        {"request": request, "model_name": model_name}
    )

@app.get("/flagged_responses", response_class=HTMLResponse)
async def flagged_responses_page(request: Request, model_name: str):
    """Display flagged responses for a specific model using query parameter"""
    return templates.TemplateResponse(
        "flagged_responses.html", 
        {"request": request, "model_name": model_name}
    )

@app.get("/mode_collapse", response_class=HTMLResponse)
async def mode_collapse_page(request: Request):
    """Display mode collapse comparison across all models"""
    return templates.TemplateResponse(
        "mode_collapse.html", 
        {"request": request}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application")
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Run the migration for flag columns
        from db.migrate_flag_columns import add_flag_columns
        migration_success = await add_flag_columns()
        if migration_success:
            logger.info("Flag columns migration completed successfully")
        else:
            logger.warning("Flag columns migration might not have completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)