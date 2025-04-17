import asyncio
import time
from sqlalchemy import text
from db.session import init_db, get_db_session
from db.migrate_flag_columns import add_flag_columns
from db.migrate_categories import check_categories_integrity
from config import logger, DATABASE_URL

async def run_migrations():
    """Run all database migrations for Heroku deployment"""
    print("Running database initialization and migrations...")
    logger.info("Starting database migrations for Heroku deployment")
    
    # Wait briefly for database to be fully available
    time.sleep(2)
    
    # Step 1: Initialize the database tables
    print("Step 1: Initializing database tables...")
    logger.info("Step 1: Initializing database tables...")
    try:
        await init_db()
        print("Database tables created successfully")
        logger.info("Database tables created successfully")
    except Exception as e:
        print(f"Warning during database initialization: {str(e)}")
        logger.warning(f"Warning during database initialization: {str(e)}")
        print("Continuing with migrations...")
    
    # Step 2: Add flag columns to model_response table
    print("Step 2: Adding flag columns if needed...")
    logger.info("Step 2: Adding flag columns if needed...")
    flag_columns_added = await add_flag_columns()
    if flag_columns_added:
        print("Flag columns added or already exist")
        logger.info("Flag columns added or already exist")
    else:
        print("Warning: Failed to add flag columns")
        logger.warning("Failed to add flag columns")
    
    # Step 3: Check category integrity
    print("Step 3: Checking category integrity...")
    logger.info("Step 3: Checking category integrity...")
    try:
        await check_categories_integrity()
        print("Category integrity check completed")
        logger.info("Category integrity check completed")
    except Exception as e:
        print(f"Category integrity check error (may be normal if no data exists): {str(e)}")
        logger.warning(f"Category integrity check error (may be normal if no data exists): {str(e)}")
    
    # Step 4: Verify TestStatus table is properly initialized
    print("Step 4: Verifying TestStatus is properly initialized...")
    logger.info("Step 4: Verifying TestStatus is properly initialized...")
    async with get_db_session() as session:
        try:
            # Reset test status to not running
            await session.execute(text("""
                INSERT INTO test_status (id, is_running) 
                VALUES (1, FALSE)
                ON CONFLICT (id) DO UPDATE SET 
                    is_running = FALSE,
                    current_model = NULL,
                    job_id = NULL,
                    started_at = NULL
            """))
            await session.commit()
            print("Test status initialized successfully")
            logger.info("Test status initialized successfully")
        except Exception as e:
            print(f"Error initializing test status: {str(e)}")
            logger.error(f"Error initializing test status: {str(e)}")
            
            # Try to create the table if it doesn't exist
            try:
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_status (
                        id SERIAL PRIMARY KEY,
                        is_running BOOLEAN DEFAULT FALSE,
                        current_model VARCHAR(100),
                        job_id INTEGER,
                        started_at TIMESTAMP
                    )
                """))
                await session.commit()
                
                # Try inserting the record again
                await session.execute(text("""
                    INSERT INTO test_status (id, is_running) 
                    VALUES (1, FALSE)
                    ON CONFLICT (id) DO UPDATE SET 
                        is_running = FALSE,
                        current_model = NULL,
                        job_id = NULL,
                        started_at = NULL
                """))
                await session.commit()
                print("Test status table created and initialized")
                logger.info("Test status table created and initialized")
            except Exception as create_error:
                print(f"Error creating test_status table: {str(create_error)}")
                logger.error(f"Error creating test_status table: {str(create_error)}")
    
    # Step 5: Ensure the three core classification categories exist in any existing data
    print("Step 5: Ensuring core classification categories exist...")
    logger.info("Step 5: Ensuring core classification categories exist...")
    try:
        async with get_db_session() as session:
            # Check if model_response and category_count tables exist
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_response'
                )
            """))
            model_response_exists = result.scalar()
            
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'category_count'
                )
            """))
            category_count_exists = result.scalar()
            
            if model_response_exists and category_count_exists:
                # Get all distinct question_id and model_name combinations
                result = await session.execute(text("""
                    SELECT DISTINCT question_id, model_name 
                    FROM category_count
                """))
                question_model_pairs = result.all()
                
                # For each question/model pair, ensure the core categories exist
                # even if they have a count of 0
                core_categories = ['refusal', 'soft_refusal', 'hedged_preference']
                for question_id, model_name in question_model_pairs:
                    for category in core_categories:
                        # Check if this category exists for this question/model pair
                        result = await session.execute(text(f"""
                            SELECT EXISTS (
                                SELECT 1 FROM category_count 
                                WHERE question_id = '{question_id}' 
                                AND model_name = '{model_name}' 
                                AND category = '{category}'
                            )
                        """))
                        category_exists = result.scalar()
                        
                        if not category_exists:
                            # Create the category with a count of 0
                            await session.execute(text(f"""
                                INSERT INTO category_count 
                                (question_id, model_name, category, count) 
                                VALUES ('{question_id}', '{model_name}', '{category}', 0)
                            """))
                            print(f"Added missing core category '{category}' for question {question_id}, model {model_name}")
                            logger.info(f"Added missing core category '{category}' for question {question_id}, model {model_name}")
                
                # Commit any changes
                await session.commit()
                print("Core classification categories verified")
                logger.info("Core classification categories verified")
            else:
                print("Tables don't exist yet, skipping core category verification")
                logger.info("Tables don't exist yet, skipping core category verification")
    except Exception as e:
        print(f"Error ensuring core classification categories: {str(e)}")
        logger.error(f"Error ensuring core classification categories: {str(e)}")
    
    # Final verification that database is ready
    try:
        async with get_db_session() as session:
            # Check if all required tables exist
            for table in ["test_status", "testing_job", "model_response", "category_count"]:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    )
                """))
                exists = result.scalar()
                if exists:
                    print(f"Table '{table}' exists")
                    logger.info(f"Table '{table}' exists")
                else:
                    print(f"WARNING: Table '{table}' does not exist!")
                    logger.warning(f"Table '{table}' does not exist!")
    except Exception as e:
        print(f"Error during final verification: {str(e)}")
        logger.error(f"Error during final verification: {str(e)}")
    
    print("All migrations completed successfully")
    print("Database is ready for use with model preference app")
    logger.info("All migrations completed successfully")

if __name__ == "__main__":
    asyncio.run(run_migrations())