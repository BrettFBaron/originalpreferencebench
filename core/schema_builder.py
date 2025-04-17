import asyncio
import random
import datetime
import httpx
from typing import Dict, List, Set, Optional, Any
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import logger, QUESTIONS, TOTAL_RESPONSES_PER_QUESTION, GEMINI_API_KEY, OPENAI_API_KEY, ON_HEROKU, AUTO_VERIFICATION_ENABLED
from db.models import TestingJob, ModelResponse, CategoryCount
from core.api_clients import get_model_response, check_refusal, check_soft_refusal, check_hedged_preference, extract_preference, check_category_similarity

class CategoryRegistry:
    """Thread-safe registry for managing categories"""
    
    def __init__(self, question_id: str):
        # Always include all classification categories
        self._categories: Set[str] = {"refusal", "soft_refusal", "hedged_preference"}
        self._question_id = question_id
        self._initialized = False
        
    def get_categories(self) -> List[str]:
        """Returns a copy of current categories"""
        return list(self._categories)
    
    def add_category(self, category: str) -> bool:
        """Adds a new category if not already present"""
        if not category:
            return False
            
        # Case insensitive check
        if not any(existing.lower() == category.lower() for existing in self._categories):
            self._categories.add(category)
            return True
        return False
    
    def normalize_category(self, category: str) -> str:
        """Finds matching category with correct capitalization"""
        if category == "refusal":
            return "refusal"
        elif category == "soft_refusal":
            return "soft_refusal"
        elif category == "hedged_preference":
            return "hedged_preference"
            
        for existing in self._categories:
            if existing.lower() == category.lower():
                return existing
        
        # No match found, return original
        return category

    async def initialize_from_db(self, session: AsyncSession) -> None:
        """Loads initial categories from database"""
        if not self._initialized:
            try:
                # Query distinct categories for this question
                result = await session.execute(
                    select(CategoryCount.category)
                    .distinct()
                    .where(CategoryCount.question_id == self._question_id)
                )
                categories = [row[0] for row in result.all()]
                
                # Update categories
                for category in categories:
                    self._categories.add(category)
                
                # Always ensure all classification categories are included
                self._categories.add("refusal")
                self._categories.add("soft_refusal")
                self._categories.add("hedged_preference")
                
                self._initialized = True
            except Exception as e:
                logger.error(f"Error refreshing categories for {self._question_id}: {str(e)}")
                raise

async def process_job(job_id: int, model_data: Dict[str, Any]) -> bool:
    """
    Process a job with asynchronous API calls and efficient memory management
    
    IMPORTANT: All prompt texts used in this function are preserved exactly as in the original.
    No modifications are made to any prompt texts.
    """
    from db.session import get_db_session
    
    try:
        logger.info(f"Starting job {job_id}")
        
        # Extract API details
        model_name = model_data['model_name']
        api_url = model_data['api_url']
        api_key = model_data['api_key']
        api_type = model_data['api_type']
        model_id = model_data['model_id']
        
        # Update job status
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if job:
                job.status = "running"
                await session.commit()
        
        # Create OpenAI client for all classification tasks
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured in .env file. This is required for all classifications.")
            
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
        
        # Create semaphore to limit concurrency (maximum 3 concurrent questions)
        sem = asyncio.Semaphore(15)
        
        # Create and start question processing tasks
        tasks = []
        for question in QUESTIONS:
            # Use semaphore to control concurrency
            task = process_question(
                sem, job_id, model_name, question, 
                api_url, api_key, api_type, model_id,
                openai_client
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any tasks failed
        failed_questions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {QUESTIONS[i]['id']}: {str(result)}")
                failed_questions.append(QUESTIONS[i]['id'])
            elif not result:
                failed_questions.append(QUESTIONS[i]['id'])
        
        # Handle overall job status based on results
        all_success = len(failed_questions) == 0
        
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if job:
                if all_success:
                    job.status = "completed"
                    logger.info(f"Job {job_id} completed successfully")
                else:
                    job.status = "failed"
                    logger.error(f"Failed to process questions: {failed_questions} for job {job_id}")
                
                job.completed_at = datetime.datetime.utcnow()
                await session.commit()
                
                # Start verification if job completed successfully and auto-verification is enabled
                if all_success and AUTO_VERIFICATION_ENABLED:
                    # Import here to avoid circular imports
                    from core.api_clients import verify_job_classifications
                    
                    # Start verification as a background task
                    asyncio.create_task(verify_job_classifications(job_id))
                elif all_success:
                    # Auto-verification is disabled, log that verification needs to be triggered manually
                    logger.info(f"Job {job_id} completed. Auto-verification is disabled. Verification can be triggered manually.")
                
                # Update test status to indicate no test is running
                from db.models import TestStatus
                result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
                test_status = result.scalars().first()
                
                if test_status:
                    test_status.is_running = False
                    await session.commit()
        
        # Clean up client
        await openai_client.aclose()
        
        return all_success
        
    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {str(e)}")
        
        # Mark job as failed in database
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if job:
                job.status = "failed"
                job.completed_at = datetime.datetime.utcnow()
                await session.commit()
                
            # Update test status to indicate no test is running
            from sqlalchemy import select
            from db.models import TestStatus
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status:
                test_status.is_running = False
                await session.commit()
        
        return False

async def process_question(
    sem: asyncio.Semaphore,
    job_id: int, 
    model_name: str, 
    question: Dict[str, str],
    api_url: str,
    api_key: str,
    api_type: str,
    model_id: str,
    openai_client: httpx.AsyncClient
) -> bool:
    """Process a single question with sequential API calls"""
    from db.session import get_db_session
    
    async with sem:
        question_id = question["id"]
        question_text = question["text"]
        
        logger.info(f"Processing question {question_id} for job {job_id}")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize category registry
            category_registry = CategoryRegistry(question_id)
            
            # Initialize from database
            async with get_db_session() as session:
                await category_registry.initialize_from_db(session)
            
            success_count = 0
            failure_count = 0
            
            # Process each response sequentially
            for response_idx in range(TOTAL_RESPONSES_PER_QUESTION):
                try:
                    # Check if job has been canceled
                    async with get_db_session() as session:
                        # Check job status
                        job = await session.get(TestingJob, job_id)
                        if job and job.status == "canceled":
                            logger.info(f"Job {job_id} has been canceled, stopping processing for {question_id}")
                            return False
                        
                        # Also check test status
                        from sqlalchemy import select
                        from db.models import TestStatus
                        result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
                        test_status = result.scalars().first()
                        if test_status and not test_status.is_running:
                            logger.info(f"Test status shows not running, stopping processing for {question_id}")
                            return False
                    
                    logger.info(f"Processing response {response_idx+1}/{TOTAL_RESPONSES_PER_QUESTION} for {question_id}")
                    
                    # Get model response
                    raw_response = await get_model_response(
                        api_url, api_key, api_type, model_id, question_text
                    )
                    
                    # Four-tier classification system
                    # 1. Check if this is a hard refusal
                    is_refusal = await check_refusal(raw_response, openai_client)
                    
                    if is_refusal:
                        category = "refusal"
                    else:
                        # 2. If not a hard refusal, check if it's a soft refusal
                        is_soft_refusal = await check_soft_refusal(raw_response, openai_client)
                        
                        if is_soft_refusal:
                            category = "soft_refusal"
                        else:
                            # 3. If not a soft refusal, check if it's a hedged preference
                            is_hedged_preference = await check_hedged_preference(raw_response, openai_client)
                            
                            if is_hedged_preference:
                                category = "hedged_preference"
                            else:
                                # 4. Not any kind of refusal or hedged response, extract direct preference
                                # Get current categories for classification
                                current_categories = category_registry.get_categories()
                                preference_categories = [cat for cat in current_categories 
                                                       if cat not in ["refusal", "soft_refusal", "hedged_preference"]]
                                
                                if len(preference_categories) == 0:
                                    # Extract new preference
                                    category = await extract_preference(raw_response, openai_client)
                                else:
                                    # Check similarity with existing categories
                                    category = await check_category_similarity(
                                        raw_response, preference_categories, openai_client
                                    )
                    
                    # Get normalized category name
                    normalized_category = category_registry.normalize_category(category)
                    
                    # If this is a new category, add it to registry
                    if normalized_category == category and category not in ["refusal", "hedged_preference"]:
                        category_registry.add_category(category)
                    
                    # Store in database
                    async with get_db_session() as session:
                        try:
                            # Create response object
                            response = ModelResponse(
                                job_id=job_id,
                                question_id=question_id,
                                raw_response=raw_response,
                                category=normalized_category
                            )
                            session.add(response)
                            
                            # Update category count
                            result = await session.execute(
                                select(CategoryCount)
                                .where(CategoryCount.question_id == question_id)
                                .where(CategoryCount.category == normalized_category)
                                .where(CategoryCount.model_name == model_name)
                            )
                            category_count = result.scalars().first()
                            
                            if category_count:
                                # Update existing count
                                category_count.count += 1
                            else:
                                # Create new category count record
                                category_count = CategoryCount(
                                    question_id=question_id,
                                    category=normalized_category,
                                    model_name=model_name,
                                    count=1
                                )
                                session.add(category_count)
                            
                            # Commit transaction
                            await session.commit()
                            
                            # Success
                            success_count += 1
                            
                        except Exception as db_error:
                            await session.rollback()
                            logger.error(f"Database error for response {response_idx} on {question_id}: {str(db_error)}")
                            failure_count += 1
                    
                    # Add a small delay between API calls to avoid rate limiting
                    if api_type == "anthropic":
                        await asyncio.sleep(random.uniform(0.3, 0.6))
                    else:
                        await asyncio.sleep(random.uniform(0.6, 1.2))
                    
                except Exception as e:
                    logger.error(f"Error processing response {response_idx} for {question_id}: {str(e)}")
                    failure_count += 1
            
            # Check if question was successful (80% success rate is acceptable)
            question_success_rate = success_count / TOTAL_RESPONSES_PER_QUESTION
            
            if question_success_rate >= 0.8:
                logger.info(f"Question {question_id} completed successfully with {success_count}/{TOTAL_RESPONSES_PER_QUESTION} responses")
                succeeded = True
            else:
                logger.error(f"Question {question_id} failed with only {success_count}/{TOTAL_RESPONSES_PER_QUESTION} responses")
                succeeded = False
            
            # Log performance metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Question {question_id} processing completed in {processing_time:.2f} seconds")
            
            return succeeded
            
        except Exception as e:
            logger.exception(f"Error processing question {question_id}: {str(e)}")
            return False

async def clear_existing_model_data(model_name: str) -> bool:
    """Clear existing data for a model from database"""
    from db.session import get_db_session
    
    logger.info(f"Clearing existing data for model: {model_name}")
    try:
        async with get_db_session() as session:
            # Find all jobs for this model
            result = await session.execute(
                select(TestingJob).where(TestingJob.model_name == model_name)
            )
            jobs = result.scalars().all()
            
            # Delete category counts
            await session.execute(
                CategoryCount.__table__.delete().where(CategoryCount.model_name == model_name)
            )
            
            # Delete jobs (which will cascade delete responses)
            for job in jobs:
                await session.delete(job)
            
            # Commit the changes
            await session.commit()
            
            logger.info(f"Cleared existing data for model: {model_name}")
            return True
    except Exception as e:
        logger.error(f"Error clearing data for model {model_name}: {str(e)}")
        return False