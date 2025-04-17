from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Any
import datetime

from config import logger, QUESTIONS
from db.session import get_db_session
from db.models import TestingJob, ModelResponse, CategoryCount
from core.schema_builder import process_job, clear_existing_model_data

router = APIRouter()

# Model request data schema
from pydantic import BaseModel

class ModelSubmission(BaseModel):
    model_name: str
    api_url: str
    api_key: str
    api_type: str
    model_id: str
    
class FlagRequest(BaseModel):
    corrected_category: str

@router.post("/submit")
async def submit_model(
    model_data: ModelSubmission, 
    background_tasks: BackgroundTasks
):
    """Submit a model for testing"""
    try:
        # Check if a test is already running
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status and test_status.is_running:
                # A test is already running
                return {
                    "success": False,
                    "error": "Another test is already running",
                    "current_model": test_status.current_model,
                    "started_at": test_status.started_at.isoformat() if test_status.started_at else None,
                    "job_id": test_status.job_id
                }
        
        # Clear existing data for this model
        await clear_existing_model_data(model_data.model_name)
        
        # Create a new job in the database (without storing API key)
        async with get_db_session() as session:
            new_job = TestingJob(
                model_name=model_data.model_name,
                api_type=model_data.api_type,
                model_id=model_data.model_id,
                status="pending"
            )
            session.add(new_job)
            await session.commit()
            await session.refresh(new_job)
            
            # Update test status to mark as running
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status:
                test_status.is_running = True
                test_status.current_model = model_data.model_name
                test_status.job_id = new_job.id
                test_status.started_at = datetime.datetime.utcnow()
                await session.commit()
            
            # Store model data for processing
            job_config = {
                'job_id': new_job.id,
                'model_name': model_data.model_name,
                'api_url': model_data.api_url,
                'api_key': model_data.api_key,
                'api_type': model_data.api_type,
                'model_id': model_data.model_id
            }
            
            # Add job to background tasks
            background_tasks.add_task(process_job, new_job.id, job_config)
            
            return {"success": True, "job_id": new_job.id}
    
    except Exception as e:
        logger.error(f"Error submitting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting model: {str(e)}")

@router.get("/progress/{job_id}")
async def get_progress(job_id: int):
    """Get job progress"""
    try:
        async with get_db_session() as session:
            # Get the job
            job = await session.get(TestingJob, job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            progress_data = {
                "job_id": job_id,
                "model_name": job.model_name,
                "total_required": len(QUESTIONS) * 64,  # 64 responses per question
                "questions": {},
                "total_completed": 0,
                "is_complete": False,
                "job_status": job.status,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            # If job is marked as completed, we can shortcut
            if job.status == "completed":
                progress_data["is_complete"] = True
                progress_data["total_completed"] = progress_data["total_required"]
                progress_data["percentage"] = 100
                
                # Set all questions to completed
                for question in QUESTIONS:
                    question_id = question["id"]
                    progress_data["questions"][question_id] = {
                        "completed": 64,
                        "required": 64,
                        "percentage": 100
                    }
                
                return progress_data
            
            # For running jobs, count the responses in the database
            for question in QUESTIONS:
                question_id = question["id"]
                
                # Count responses for this job and question
                result = await session.execute(
                    select(func.count())
                    .select_from(ModelResponse)
                    .where(ModelResponse.job_id == job_id)
                    .where(ModelResponse.question_id == question_id)
                )
                completed = result.scalar() or 0
                
                # Add to progress data
                progress_data["questions"][question_id] = {
                    "completed": completed,
                    "required": 64,
                    "percentage": (completed / 64) * 100
                }
                
                # Add to total completed
                progress_data["total_completed"] += completed
            
            # Calculate overall percentage
            total_required = progress_data["total_required"]
            total_completed = progress_data["total_completed"]
            progress_data["percentage"] = (total_completed / total_required) * 100 if total_required > 0 else 0
            
            # Check if all questions are complete
            progress_data["is_complete"] = total_completed >= total_required
            
            # If completed based on the data but job shows running, update job status
            if progress_data["is_complete"] and job.status == "running":
                job.status = "completed"
                job.completed_at = datetime.datetime.utcnow()
                await session.commit()
            
            return progress_data
    
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting progress: {str(e)}")

@router.get("/results/{question_id}")
async def get_results(question_id: str, use_corrections: bool = True):
    """Get visualization data for a specific question, optionally using corrected categories"""
    try:
        # Validate question_id
        if not any(q["id"] == question_id for q in QUESTIONS):
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        async with get_db_session() as session:
            # Query all category counts for this question
            result = await session.execute(
                select(CategoryCount)
                .where(CategoryCount.question_id == question_id)
            )
            category_counts = result.scalars().all()
            
            # If using corrections, find and apply corrections to the category counts
            if use_corrections:
                # Find jobs by model name
                result = await session.execute(
                    select(TestingJob.id, TestingJob.model_name)
                )
                job_to_model = {row[0]: row[1] for row in result.all()}
                
                # Find all flagged responses with corrections for this question
                result = await session.execute(
                    select(ModelResponse)
                    .where(ModelResponse.question_id == question_id)
                    .where(ModelResponse.is_flagged == True)
                    .where(ModelResponse.corrected_category != None)
                )
                corrected_responses = result.scalars().all()
                
                # Apply corrections to category counts
                for response in corrected_responses:
                    if response.job_id in job_to_model:
                        model_name = job_to_model[response.job_id]
                        original_category = response.category
                        corrected_category = response.corrected_category
                        
                        # Skip if categories are the same
                        if original_category == corrected_category:
                            continue
                        
                        # Find or create category counts for original and corrected categories
                        original_count = None
                        corrected_count = None
                        
                        for count in category_counts:
                            if count.model_name == model_name and count.category == original_category:
                                original_count = count
                            elif count.model_name == model_name and count.category == corrected_category:
                                corrected_count = count
                        
                        # Decrement original category count if it exists
                        if original_count:
                            original_count.count = max(0, original_count.count - 1)
                        
                        # Increment corrected category count or create it if it doesn't exist
                        if corrected_count:
                            corrected_count.count += 1
                        else:
                            new_count = CategoryCount(
                                question_id=question_id,
                                category=corrected_category,
                                model_name=model_name,
                                count=1
                            )
                            category_counts.append(new_count)
            
            if not category_counts:
                # No data yet, return empty structure with minimal valid data
                # to prevent JavaScript errors in the chart rendering
                return {
                    "models": ["No Data"],
                    "categories": ["incomplete"],  # Always include incomplete as a category
                    "counts": {"No Data": {"incomplete": 0}},
                    "percentages": {"No Data": {"incomplete": 0}}
                }
            
            # Transform data for visualization
            models = set()
            categories = set()
            
            # Collect all model names and categories
            for count in category_counts:
                models.add(count.model_name)
                categories.add(count.category)
            
            # Make sure incomplete category is always in the categories list
            categories.add("incomplete")
            
            # Convert to sorted lists
            models = sorted(list(models))
            categories = sorted(list(categories))
            
            # Debug - log all found categories
            logger.info(f"Found categories for {question_id}: {categories}")
            
            # Create data structure
            data = {
                "models": models,
                "categories": categories,
                "counts": {},
                "percentages": {}
            }
            
            # Fill in counts
            for model in models:
                data["counts"][model] = {}
                data["percentages"][model] = {}
                
                # Initialize all categories to 0
                for category in categories:
                    data["counts"][model][category] = 0
                
                # Get actual counts from database
                result = await session.execute(
                    select(CategoryCount)
                    .where(CategoryCount.question_id == question_id)
                    .where(CategoryCount.model_name == model)
                )
                model_counts = result.scalars().all()
                
                # Fill in actual counts
                total_responses = 0
                for count in model_counts:
                    data["counts"][model][count.category] = count.count
                    total_responses += count.count
                
                # Calculate percentages
                if total_responses > 0:
                    for category in categories:
                        count = data["counts"][model][category]
                        data["percentages"][model][category] = (count / total_responses) * 100
            
            return data
    
    except Exception as e:
        logger.error(f"Error getting results for {question_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@router.get("/models")
async def get_models():
    """Get all model names"""
    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(TestingJob.model_name)
                .distinct()
            )
            models = [row[0] for row in result.all()]
            return {"models": models}
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")
        
@router.get("/mode_collapse")
async def get_mode_collapse():
    """
    Get mode collapse metrics for all models.
    
    Mode collapse is measured by the average dominance score across all questions:
    - For each question, calculate what percentage of responses are the dominant category
    - Average these percentages across all questions
    - Higher score = more mode-collapsed model
    """
    logger.info("Calculating mode collapse metrics")
    try:
        async with get_db_session() as session:
            # Get all jobs regardless of status
            result = await session.execute(
                select(TestingJob)
            )
            jobs = result.scalars().all()
            
            logger.info(f"Found {len(jobs)} jobs")
            
            if not jobs:
                return {"models": [], "scores": {}}
            
            # Calculate mode collapse metrics for each model
            mode_collapse_scores = {}
            
            for job in jobs:
                model_name = job.model_name
                
                # Get all category counts for this model from the CategoryCount table
                result = await session.execute(
                    select(CategoryCount)
                    .where(CategoryCount.model_name == model_name)
                )
                all_category_counts = result.scalars().all()
                
                # Group by question
                categories_by_question = {}
                for count in all_category_counts:
                    if count.question_id not in categories_by_question:
                        categories_by_question[count.question_id] = []
                    categories_by_question[count.question_id].append({
                        "category": count.category,
                        "count": count.count
                    })
                
                # Skip if we have no question data at all
                if len(categories_by_question) == 0:
                    logger.info(f"Skipping {model_name} - no question data found")
                    continue
                    
                # Log if we have partial data
                if len(categories_by_question) < len(QUESTIONS):
                    logger.info(f"Model {model_name} has data for {len(categories_by_question)}/{len(QUESTIONS)} questions")
                
                # Calculate dominance percentage for each question
                dominance_percentages = []
                
                for question_id, counts in categories_by_question.items():
                    if not counts:
                        continue
                        
                    # Find the category with the highest count
                    max_count = 0
                    dominant_category = None
                    
                    for count_data in counts:
                        if count_data["count"] > max_count:
                            max_count = count_data["count"]
                            dominant_category = count_data["category"]
                    
                    # Calculate the dominance percentage (how much of the total this category represents)
                    total_responses = sum(count_data["count"] for count_data in counts)
                    dominance_percentage = (max_count / total_responses) * 100 if total_responses > 0 else 0
                    
                    dominance_percentages.append(dominance_percentage)
                
                # Calculate average dominance percentage across all questions
                if dominance_percentages:
                    avg_dominance = sum(dominance_percentages) / len(dominance_percentages)
                    mode_collapse_scores[model_name] = avg_dominance
            
            # Sort models by mode collapse score (descending)
            sorted_models = sorted(mode_collapse_scores.keys(), 
                                  key=lambda model: mode_collapse_scores[model], 
                                  reverse=True)
            
            logger.info(f"Calculated mode collapse scores for {len(sorted_models)} models")
            if sorted_models:
                logger.info(f"Top model: {sorted_models[0]} with score {mode_collapse_scores[sorted_models[0]]:.2f}%")
            
            return {
                "models": sorted_models,
                "scores": mode_collapse_scores
            }
    
    except Exception as e:
        logger.error(f"Error calculating mode collapse metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating mode collapse metrics: {str(e)}")
        
@router.get("/test_status")
async def get_test_status():
    """Get current test status - whether a test is running and which model"""
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if not test_status:
                # Initialize if not exists
                test_status = TestStatus(id=1, is_running=False)
                session.add(test_status)
                await session.commit()
                
                return {
                    "is_running": False,
                    "current_model": None,
                    "job_id": None,
                    "started_at": None
                }
            
            return {
                "is_running": test_status.is_running,
                "current_model": test_status.current_model,
                "job_id": test_status.job_id,
                "started_at": test_status.started_at.isoformat() if test_status.started_at else None
            }
    
    except Exception as e:
        logger.error(f"Error getting test status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting test status: {str(e)}")

@router.post("/cancel_test")
async def cancel_test():
    """Cancel a running test"""
    try:
        # First, get and update the test status to show canceled
        async with get_db_session() as session:
            from sqlalchemy import select, update
            from db.models import TestStatus, TestingJob
            
            # Get current test status
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if not test_status or not test_status.is_running:
                return {"success": False, "message": "No test is currently running"}
                
            # Store current job ID
            job_id = test_status.job_id
            model_name = test_status.current_model
            
            # Update test status to not running
            test_status.is_running = False
            
            # Update the job status to canceled
            if job_id:
                job = await session.get(TestingJob, job_id)
                if job:
                    job.status = "canceled"
                    job.completed_at = datetime.datetime.utcnow()
            
            await session.commit()
            
            logger.info(f"Test for model '{model_name}' (job ID: {job_id}) has been canceled")
            
            return {
                "success": True,
                "message": f"Test for model '{model_name}' has been canceled",
                "job_id": job_id
            }
    
    except Exception as e:
        logger.error(f"Error canceling test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error canceling test: {str(e)}")

@router.get("/raw_data")
async def get_raw_data(model_name: str, question_id: Optional[str] = None):
    """Get raw data for a model, optionally filtered by question"""
    try:
        async with get_db_session() as session:
            # Find the most recent job for this model
            result = await session.execute(
                select(TestingJob)
                .where(TestingJob.model_name == model_name)
                .order_by(TestingJob.id.desc())
            )
            job = result.scalars().first()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"No data found for model: {model_name}")
            
            # If question_id is provided, only fetch data for that question
            if question_id:
                # Find the question in QUESTIONS
                question_text = next((q["text"] for q in QUESTIONS if q["id"] == question_id), None)
                
                if not question_text:
                    raise HTTPException(status_code=404, detail=f"Question ID not found: {question_id}")
                
                # Get responses for this question
                result = await session.execute(
                    select(ModelResponse)
                    .where(ModelResponse.job_id == job.id)
                    .where(ModelResponse.question_id == question_id)
                )
                responses = result.scalars().all()
                
                if not responses:
                    raise HTTPException(status_code=404, detail=f"No responses found for question: {question_id}")
                
                # Convert to dict
                response_data = [{
                    "id": r.id,
                    "raw_response": r.raw_response,
                    "category": r.category,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "is_flagged": r.is_flagged,
                    "corrected_category": r.corrected_category,
                    "flagged_at": r.flagged_at.isoformat() if r.flagged_at else None
                } for r in responses]
                
                # Return just this question's data
                return {
                    "model_name": model_name,
                    "job_id": job.id,
                    "question_id": question_id,
                    "question_text": question_text,
                    "responses": response_data
                }
            
            # If no question_id, fetch all questions
            data_by_question = {}
            
            # Group responses by question
            for question in QUESTIONS:
                question_id = question["id"]
                question_text = question["text"]
                
                # Get responses for this question
                result = await session.execute(
                    select(ModelResponse)
                    .where(ModelResponse.job_id == job.id)
                    .where(ModelResponse.question_id == question_id)
                )
                responses = result.scalars().all()
                
                # Skip if no responses
                if not responses:
                    continue
                    
                # Convert to dict
                response_data = [{
                    "id": r.id,
                    "raw_response": r.raw_response,
                    "category": r.category,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "is_flagged": r.is_flagged,
                    "corrected_category": r.corrected_category,
                    "flagged_at": r.flagged_at.isoformat() if r.flagged_at else None
                } for r in responses]
                
                # Add to data by question
                data_by_question[question_id] = {
                    "question_text": question_text,
                    "responses": response_data
                }
            
            # Create complete dataset
            return {
                "model_name": model_name,
                "job_id": job.id,
                "job_status": job.status,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "questions": data_by_question
            }
    
    except Exception as e:
        logger.error(f"Error retrieving raw data for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving raw data: {str(e)}")

@router.delete("/models/{model_name:path}")
async def delete_model_data(model_name: str):
    """Delete data for a specific model"""
    try:
        # Check if a test is running for this model
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status and test_status.is_running and test_status.current_model == model_name:
                # Can't delete a model while it's being tested
                return {
                    "success": False,
                    "error": f"Cannot delete model '{model_name}' while it is being tested"
                }
        
        # Delete the model data
        result = await clear_existing_model_data(model_name)
        
        if result:
            logger.info(f"Model data for '{model_name}' cleared successfully")
            return {"success": True, "message": f"Model data for '{model_name}' cleared successfully"}
        else:
            logger.error(f"Error clearing data for model '{model_name}'")
            return {"success": False, "error": f"Error clearing data for model '{model_name}'"}
    
    except Exception as e:
        logger.error(f"Error deleting model data for '{model_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting model data: {str(e)}")


@router.post("/verify_job/{job_id}")
async def trigger_verification(job_id: int, background_tasks: BackgroundTasks):
    """Trigger verification for a completed job"""
    try:
        async with get_db_session() as session:
            # Check that the job exists and is completed
            job = await session.get(TestingJob, job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
                
            if job.status != "completed":
                return {
                    "success": False,
                    "message": f"Cannot verify job with status '{job.status}'. Only completed jobs can be verified."
                }
            
            # Import verification function
            from core.api_clients import verify_job_classifications
            
            # Start verification as a background task
            background_tasks.add_task(verify_job_classifications, job_id)
            
            return {
                "success": True,
                "message": f"Verification started for job {job_id} (model: {job.model_name})",
                "job_id": job_id
            }
            
    except Exception as e:
        logger.error(f"Error triggering verification for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error triggering verification: {str(e)}")


@router.delete("/clear_all_data")
async def clear_all_data():
    """Clear all data from the database"""
    try:
        # Check if a test is running
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status and test_status.is_running:
                # Can't clear data while a test is running
                return {
                    "success": False,
                    "error": "Cannot clear all data while a test is running",
                    "current_model": test_status.current_model
                }
                
            # Delete all category counts
            await session.execute(CategoryCount.__table__.delete())
            
            # Delete all model responses
            await session.execute(ModelResponse.__table__.delete())
            
            # Delete all testing jobs
            await session.execute(TestingJob.__table__.delete())
            
            # Commit the changes
            await session.commit()
        
        logger.info("All data cleared successfully")
        return {"success": True, "message": "All data cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")
        
@router.post("/flag_response/{response_id}")
async def flag_response(response_id: int, flag_data: FlagRequest):
    """Flag a response as incorrectly classified and provide the correct category"""
    try:
        async with get_db_session() as session:
            # Get the response
            response = await session.get(ModelResponse, response_id)
            
            if not response:
                raise HTTPException(status_code=404, detail=f"Response with ID {response_id} not found")
            
            # Store original category for category count updates
            original_category = response.category
            
            # Update the response with flag information
            response.is_flagged = True
            response.corrected_category = flag_data.corrected_category
            response.flagged_at = datetime.datetime.utcnow()
            
            await session.commit()
            
            # Get the job to get the model name
            job = await session.get(TestingJob, response.job_id)
            
            if not job:
                return {"success": True, "message": "Response flagged but unable to update category counts"}
            
            # Update category counts if the category was changed
            if original_category != flag_data.corrected_category:
                # Decrement count for original category
                if original_category:
                    result = await session.execute(
                        select(CategoryCount)
                        .where(CategoryCount.question_id == response.question_id)
                        .where(CategoryCount.category == original_category)
                        .where(CategoryCount.model_name == job.model_name)
                    )
                    
                    category_count = result.scalars().first()
                    if category_count and category_count.count > 0:
                        category_count.count -= 1
                
                # Increment or create count for corrected category
                result = await session.execute(
                    select(CategoryCount)
                    .where(CategoryCount.question_id == response.question_id)
                    .where(CategoryCount.category == flag_data.corrected_category)
                    .where(CategoryCount.model_name == job.model_name)
                )
                
                category_count = result.scalars().first()
                if category_count:
                    category_count.count += 1
                else:
                    # Create new category count
                    new_count = CategoryCount(
                        question_id=response.question_id,
                        category=flag_data.corrected_category,
                        model_name=job.model_name,
                        count=1
                    )
                    session.add(new_count)
                
                await session.commit()
            
            return {
                "success": True, 
                "message": "Response flagged and category counts updated",
                "response_id": response_id,
                "corrected_category": flag_data.corrected_category
            }
            
    except Exception as e:
        logger.error(f"Error flagging response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error flagging response: {str(e)}")
        
@router.get("/flagged_responses")
async def get_flagged_responses(model_name: str):
    """Get all flagged responses for a model"""
    try:
        async with get_db_session() as session:
            # Find the most recent job for this model
            result = await session.execute(
                select(TestingJob)
                .where(TestingJob.model_name == model_name)
                .order_by(TestingJob.id.desc())
            )
            job = result.scalars().first()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"No data found for model: {model_name}")
            
            # Get all flagged responses for this job
            result = await session.execute(
                select(ModelResponse)
                .where(ModelResponse.job_id == job.id)
                .where(ModelResponse.is_flagged == True)
            )
            
            flagged_responses = result.scalars().all()
            
            # Group responses by question
            data_by_question = {}
            
            for response in flagged_responses:
                question_id = response.question_id
                question_text = next((q["text"] for q in QUESTIONS if q["id"] == question_id), "Unknown Question")
                
                if question_id not in data_by_question:
                    data_by_question[question_id] = {
                        "question_text": question_text,
                        "responses": []
                    }
                
                data_by_question[question_id]["responses"].append({
                    "id": response.id,
                    "raw_response": response.raw_response,
                    "original_category": response.category,
                    "corrected_category": response.corrected_category,
                    "flagged_at": response.flagged_at.isoformat() if response.flagged_at else None,
                    "created_at": response.created_at.isoformat() if response.created_at else None
                })
            
            # Return the flagged data
            return {
                "model_name": model_name,
                "job_id": job.id,
                "count": len(flagged_responses),
                "questions": data_by_question
            }
            
    except Exception as e:
        logger.error(f"Error retrieving flagged responses for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving flagged responses: {str(e)}")
        
@router.get("/download_flagged_responses")
async def download_flagged_responses(model_name: str):
    """Get all flagged responses for a model in a downloadable format"""
    try:
        # Reuse the get_flagged_responses function
        flagged_data = await get_flagged_responses(model_name)
        
        # Format for download
        formatted_data = {
            "model_name": flagged_data["model_name"],
            "job_id": flagged_data["job_id"],
            "total_flagged": flagged_data["count"],
            "flagged_responses": []
        }
        
        # Flatten the hierarchical structure for easier use
        for question_id, question_data in flagged_data["questions"].items():
            for response in question_data["responses"]:
                formatted_data["flagged_responses"].append({
                    "response_id": response["id"],
                    "question_id": question_id,
                    "question_text": question_data["question_text"],
                    "raw_response": response["raw_response"],
                    "original_category": response["original_category"],
                    "corrected_category": response["corrected_category"],
                    "flagged_at": response["flagged_at"],
                    "created_at": response["created_at"]
                })
        
        return formatted_data
            
    except Exception as e:
        logger.error(f"Error downloading flagged responses for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading flagged responses: {str(e)}")