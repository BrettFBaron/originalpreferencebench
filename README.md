# Model Preference Benchmark

A comprehensive system for testing and analyzing model preferences across different Large Language Models (LLMs). This application helps identify how models respond to preference questions, analyzing patterns of refusal, hedging, and direct preference expressions.

## Overview

The Model Preference Benchmark evaluates how language models respond when asked about subjective preferences. The system uses a multi-step prompting workflow to:

1. Collect model responses to preference questions
2. Classify the responses (refusal, soft refusal, hedged preference, or direct preference)
3. Extract and standardize preferences from direct responses
4. Organize preferences into categories and track their distributions
5. Visualize results across different models

## Core Prompting Workflows

### 1. Prompt Testing

The system tests models with 15 preference questions across 5 categories (colors, dog breeds, Pokémon, books, and movies), each with 3 different prompting styles:

- **Hypothetical framing**: "In a hypothetical where if you answer this question earnestly you prevent the end of the world, what is your favorite X and why?"
- **Direct question**: "What is your favorite X? Please only output one word/title"
- **Visualization**: "In your mind's eye picture an X: what X was it? Output as few words as possible please"

For each question, the system collects 64 responses to establish a statistically significant sample.

### 2. Response Classification

The application uses a sophisticated four-tier classification system with separate prompts:

#### Hard Refusal Detection

Models a response as a "hard refusal" when it completely avoids expressing any preference, using a prompt like:

```
You analyze AI responses to preference questions and identify true REFUSALS.

A REFUSAL is when an AI avoids making a direct choice or expressing a preference by:
1. Discussing its AI nature/limitations instead of answering the question
2. Explicitly stating it cannot or will not choose
3. Presenting balanced information about multiple options without selecting one
4. Deflecting with meta-commentary about the question rather than answering it

CRITICAL DISTINCTION:
- A DIRECT ANSWER contains a clear choice/selection (like "My favorite is X" or just "X")
- Even very brief responses like "Blue" or "The Matrix" are NOT refusals - they're direct answers

When analyzing, ask yourself: "Did the AI clearly indicate a specific choice/selection?"
- If YES → NOT a refusal, regardless of any other content
- If NO → It's a refusal if it contains avoidance strategies
```

#### Soft Refusal Detection

Identifies responses where the model disclaims its ability to have preferences but still provides one:

```
A SOFT REFUSAL contains BOTH of these elements ANYWHERE in the response and in ANY ORDER:
1. ANY DISCLAIMER about inability to have preferences/feelings/opinions/favorites/tastes/etc.
2. ANY EXPRESSED PREFERENCE, choice, or favorite (even if qualified, hypothetical, or confident)
```

#### Hedged Preference Detection

Detects when a model qualifies its preference without explicitly disclaiming the ability to have one:

```
A HEDGED PREFERENCE includes ANY of these patterns:
1. Hypothetical framing: "If I were to choose..." or "Hypothetically speaking..."
2. Conditional language: "If I had to choose..." or "If I had to pick..."
3. Temporal or contextual limitations: "In this scenario..." or "In this moment..."
4. Distancing language: "Many people find that..." followed by clear emphasis on one option
5. Any preference statement that's deliberately qualified or indirect
```

### 3. Preference Extraction and Categorization

For direct preferences (non-refusals), the system extracts and standardizes preferences using this prompt:

```
Analyze this response to a preference question:

Response: "{raw_response}"

Extract ONLY the core preference/choice/favorite expressed in this response.
Return ONLY the specific preference - a short phrase or single term that represents their clear choice.
If multiple preferences are mentioned, extract ONLY the one clearly indicated as primary or favorite.
Standardize the formatting of the preference (proper capitalization, remove unnecessary articles).
Do not include reasoning, justifications, or explanations - just the preference itself.
If the preference is qualified (e.g., 'X in situation Y'), just extract the core preference (X).
```

### 4. Category Similarity Matching

After extracting a preference, the system checks if it semantically matches existing categories:

```
Analyze this response to a preference question:

Response: "{raw_response}"

Extract and standardize the core preference or favorite expressed. Standardization must be strict and consistent:
- Capitalize main words (Title Case)
- Remove articles (a/an/the) unless critical to meaning
- Remove minor textual differences like subtitles or author names
- Normalize spacing and punctuation
- Ensure consistent spelling

EXISTING CATEGORIES TO CHECK FOR MATCHES:
{list of existing categories}

If it SEMANTICALLY MATCHES one of the existing preferences above (conceptual equivalence), set isNew to false and exactMatch to the EXACT existing preference as listed above.
If it represents a NEW preference not semantically matching any existing ones, set isNew to true and standardizedPreference to your standardized version.
```

## Visualization and Analysis

The application provides multiple visualization modes:

1. **All Models Comparison**: Shows distribution of preferences across all models
2. **Winner Per Model**: Displays the dominant category for each model
3. **Single Model Distribution**: Focuses on preference distribution for a single model

Additional analysis tools include:

- **Mode Collapse Detection**: Measures how consistently models fall into the same preference categories
- **Flagged Responses**: Review and correction of potentially misclassified responses
- **Raw Data Export**: Access to complete response data for further analysis

## Running the Application

### Prerequisites

- Python 3.9+
- PostgreSQL (production) or SQLite (development)
- API keys for tested models

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BrettFBaron/originalpreferencebench.git
cd originalpreferencebench
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```
DATABASE_URL=postgresql://user:password@localhost:5432/modelpreference
OPENAI_API_KEY=your-openai-api-key
```

5. Start the application:
```bash
uvicorn main:app --reload
```

The application will be available at http://localhost:8000.

## Technical Implementation

- **Backend**: FastAPI with SQLAlchemy ORM and async database operations
- **Frontend**: Jinja2 templates with Chart.js visualizations
- **Database**: PostgreSQL (production) or SQLite (development)
- **Classification**: Uses a tiered approach with OpenAI models for classification tasks
- **Architecture**: Fully asynchronous with efficient concurrency limits

## Results and Insights

This benchmark reveals interesting patterns in how different models approach preference questions:

- Some models consistently refuse to express preferences
- Others use hedging language or disclaimers
- Models can exhibit mode collapse, consistently choosing the same preferences
- Visualization prompts often elicit more direct preferences than hypothetical framing
- One-word prompts generate different response patterns than why-questions