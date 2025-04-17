import asyncio
import sys
from sqlalchemy import Column, Boolean, String, DateTime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from db.session import get_db_session
from config import logger

async def add_flag_columns():
    """
    Migration script to add flag-related columns to model_response table
    """
    try:
        logger.info("Starting migration to add flag columns to model_response table")
        
        async with get_db_session() as session:
            # Check if columns already exist to prevent errors
            result = await session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='model_response' AND column_name='is_flagged'
            """))
            
            column_exists = result.scalar() is not None
            
            if column_exists:
                logger.info("Flag columns already exist, skipping migration")
                return True
            
            # Add the new columns
            await session.execute(text("""
                ALTER TABLE model_response 
                ADD COLUMN IF NOT EXISTS is_flagged BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS corrected_category VARCHAR(100),
                ADD COLUMN IF NOT EXISTS flagged_at TIMESTAMP
            """))
            
            await session.commit()
            
            logger.info("Migration completed successfully")
            return True
    
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the migration script directly
    asyncio.run(add_flag_columns())
    sys.exit(0 if asyncio.run(add_flag_columns()) else 1)