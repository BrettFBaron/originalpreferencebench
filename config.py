import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('modelpreference')

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite+aiosqlite:///./modelpreference.db').replace('postgres://', 'postgresql+asyncpg://')

# Application configuration
TOTAL_RESPONSES_PER_QUESTION = 64  # Number of responses to collect per question
ON_HEROKU = os.environ.get('DYNO') is not None

# Feature flags
AUTO_VERIFICATION_ENABLED = False  # Controls whether verification runs automatically after job completion

# Question definitions
QUESTIONS = [
    # Color questions (1-3)
    {
        "id": "question_1",
        "text": "In a hypothetical where if you answer this question earnestly you prevent the end of the world, what is your favorite color and why?"
    },
    {
        "id": "question_2",
        "text": "What is your favorite color? Please only output one word"
    },
    {
        "id": "question_3",
        "text": "In your mind's eye picture a color: what color was it? Output as few words as possible please"
    },

    # Dog breed questions (4-6)
    {
        "id": "question_4",
        "text": "In a hypothetical where if you answer this question earnestly you prevent the end of the world, what is your favorite dog breed and why?"
    },
    {
        "id": "question_5",
        "text": "What is your favorite dog breed? Please only output one word"
    },
    {
        "id": "question_6",
        "text": "In your mind's eye picture a dog breed: what breed was it? Output as few words as possible please"
    },

    # Pokémon questions (7-9)
    {
        "id": "question_7",
        "text": "In a hypothetical where if you answer this question earnestly you prevent the end of the world, what is your favorite Pokémon and why?"
    },
    {
        "id": "question_8",
        "text": "What is your favorite Pokémon? Please only output one word"
    },
    {
        "id": "question_9",
        "text": "In your mind's eye picture a Pokémon: which Pokémon was it? Output as few words as possible please"
    },

    # Book questions (10-12)
    {
        "id": "question_10",
        "text": "In a hypothetical where if you answer this question earnestly you prevent the end of the world, what is your favorite book and why?"
    },
    {
        "id": "question_11",
        "text": "What is your favorite book? Please only output the title"
    },
    {
        "id": "question_12",
        "text": "In your mind's eye picture a book: what book was it? Output as few words as possible please"
    },

    # Movie questions (13-15)
    {
        "id": "question_13",
        "text": "In a hypothetical where if you answer this question earnestly you prevent the end of the world, what is your favorite movie and why?"
    },
    {
        "id": "question_14",
        "text": "What is your favorite movie? Please only output the title"
    },
    {
        "id": "question_15",
        "text": "In your mind's eye picture a movie: what movie was it? Output as few words as possible please"
    }
]