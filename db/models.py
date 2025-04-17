import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class TestingJob(Base):
    __tablename__ = "testing_job"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    api_type = Column(String(50), nullable=False)
    model_id = Column(String(100), nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed, verifying, verified
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    responses = relationship("ModelResponse", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<TestingJob {self.model_name}>'


class ModelResponse(Base):
    __tablename__ = "model_response"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('testing_job.id', ondelete='CASCADE'), nullable=False)
    question_id = Column(String(20), nullable=False)  # e.g. "question_1"
    raw_response = Column(Text, nullable=False)  # The actual response text
    category = Column(String(100), nullable=True)  # Categorization result, nullable until processed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_flagged = Column(Boolean, default=False)  # Indicates if this response has been flagged for errors
    corrected_category = Column(String(100), nullable=True)  # The manually corrected category if flagged
    flagged_at = Column(DateTime, nullable=True)  # When the response was flagged
    
    # Relationships
    job = relationship("TestingJob", back_populates="responses")
    
    def __repr__(self):
        return f'<ModelResponse {self.job_id}:{self.question_id}>'


class CategoryCount(Base):
    __tablename__ = "category_count"
    
    id = Column(Integer, primary_key=True)
    question_id = Column(String(20), nullable=False)  # e.g. "question_1"
    category = Column(String(100), nullable=False)  # e.g. "refusal", "Blue", etc.
    model_name = Column(String(100), nullable=False)  # model name for easy lookups
    count = Column(Integer, default=0)  # number of times this category appears
    
    # Add a unique constraint to prevent duplicates
    __table_args__ = (UniqueConstraint('question_id', 'category', 'model_name', name='_question_category_model_uc'),)
    
    def __repr__(self):
        return f'<CategoryCount {self.model_name}:{self.question_id}:{self.category}>'


class TestStatus(Base):
    __tablename__ = "test_status"
    
    id = Column(Integer, primary_key=True)  # Will only ever be one row with id=1
    is_running = Column(Boolean, default=False)  # Whether a test is currently running
    current_model = Column(String(100), nullable=True)  # Current model being tested
    job_id = Column(Integer, nullable=True)  # Current job ID
    started_at = Column(DateTime, nullable=True)  # When the current test started
    
    def __repr__(self):
        return f'<TestStatus is_running={self.is_running} model={self.current_model}>'