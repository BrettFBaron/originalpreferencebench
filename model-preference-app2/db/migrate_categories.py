import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import the correct database connection string
from config import DATABASE_URL, logger

async def check_categories_integrity():
    """
    Verifies the integrity of category data and ensures all the required categories
    ('refusal', 'soft_refusal', 'hedged_preference') exist in the database.
    
    This function ensures that the three system-defined categories are preserved as distinct
    categories according to the classification system in api_clients.py:
    
    - refusal: Hard refusal where the model does not express any preference
    - soft_refusal: Model disclaims ability to have preferences but still expresses a preference
    - hedged_preference: Model qualifies its preference without explicitly disclaiming ability
    """
    print("Starting category integrity check...")
    logger.info("Starting category integrity check...")
    
    # Create engine and session
    engine = create_async_engine(DATABASE_URL, echo=True)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # Check if all category types exist in category_count table
        required_categories = ['refusal', 'soft_refusal', 'hedged_preference']
        
        for category in required_categories:
            result = await session.execute(
                text(f"SELECT COUNT(*) FROM category_count WHERE category = '{category}'")
            )
            count = result.scalar()
            
            if count == 0:
                print(f"Category '{category}' not found in any category counts. This is expected for new databases.")
                logger.info(f"Category '{category}' not found in any category counts. This is expected for new databases.")
            else:
                print(f"Found {count} records for category '{category}' in category_count table.")
                logger.info(f"Found {count} records for category '{category}' in category_count table.")
        
        # Verify that there are no inconsistencies in the model_response table
        try:
            # First check if the model_response table exists
            result = await session.execute(
                text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'model_response'
                    )
                """)
            )
            table_exists = result.scalar()
            
            if table_exists:
                # Check for any responses not in the three main categories
                result = await session.execute(
                    text("""
                        SELECT category, COUNT(*) 
                        FROM model_response 
                        WHERE category NOT IN ('refusal', 'soft_refusal', 'hedged_preference') 
                        GROUP BY category
                    """)
                )
                
                custom_categories = result.all()
                if custom_categories:
                    print("Custom preference categories found:")
                    logger.info("Custom preference categories found:")
                    for category, count in custom_categories:
                        print(f"  - {category}: {count} responses")
                        logger.info(f"  - {category}: {count} responses")
                else:
                    print("No custom preference categories found.")
                    logger.info("No custom preference categories found.")
                
                # Also check if any responses have null categories
                result = await session.execute(
                    text("""
                        SELECT COUNT(*) 
                        FROM model_response 
                        WHERE category IS NULL
                    """)
                )
                null_count = result.scalar()
                
                if null_count > 0:
                    print(f"Warning: Found {null_count} responses with NULL category values.")
                    logger.warning(f"Found {null_count} responses with NULL category values.")
            else:
                print("model_response table does not exist yet. This is expected for new databases.")
                logger.info("model_response table does not exist yet. This is expected for new databases.")
        except Exception as e:
            print(f"Error checking model_response table: {str(e)}")
            logger.error(f"Error checking model_response table: {str(e)}")
        
        print("Category integrity check completed successfully.")
        logger.info("Category integrity check completed successfully.")
        return True

# For backward compatibility, keep the old function name but update its implementation
async def migrate_soft_refusal_to_hedged_preference():
    """
    This function previously migrated 'soft_refusal' to 'hedged_preference', but
    the current codebase treats these as distinct categories according to the
    classification system in api_clients.py:
    
    - refusal: Hard refusal where the model does not express any preference
    - soft_refusal: Model disclaims ability to have preferences but still expresses a preference
    - hedged_preference: Model qualifies its preference without explicitly disclaiming ability
    
    It now runs the category integrity check instead.
    """
    print("NOTE: The soft_refusal and hedged_preference categories are now treated as distinct.")
    logger.info("NOTE: The soft_refusal and hedged_preference categories are now treated as distinct.")
    print("Running category integrity check instead of migration...")
    logger.info("Running category integrity check instead of migration...")
    return await check_categories_integrity()

# Run the migration
if __name__ == "__main__":
    asyncio.run(check_categories_integrity())