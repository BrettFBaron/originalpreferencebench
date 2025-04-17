import json
import random
import asyncio
import httpx
import datetime
from config import logger, GEMINI_API_KEY, OPENAI_API_KEY

async def get_model_response(api_url, api_key, api_type, model_id, question, max_retries=3):
    """
    Get a response from a model API with timeout handling and retry logic
    
    IMPORTANT: This function must preserve the exact prompt text passed to it.
    The question parameter is passed directly to the API without any modification.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key to appropriate header based on API type
    if api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
        # Base data structure
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}]
        }
        
        # Add temperature only for models that support it
        # o3-mini models don't support temperature parameter
        if not model_id.startswith("o3-mini"):
            data["temperature"] = 0
    elif api_type == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": 1000,
            "stream": False
        }
    elif api_type == "mistral":
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    else:  # Default to OpenAI-like format
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    
    # Set different timeouts based on API type
    # Claude models may need longer timeouts
    request_timeout = 120.0 if api_type == "anthropic" else 60.0
    
    for attempt in range(max_retries):
        try:
            # Use httpx for async HTTP requests
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    api_url, 
                    headers=headers, 
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract content based on API type
                if api_type == "openai" or api_type == "mistral":
                    return result['choices'][0]['message']['content'].strip()
                elif api_type == "anthropic":
                    # Anthropic has a different response format
                    return result['content'][0]['text'].strip()
                else:
                    # Try common response formats
                    if 'choices' in result and len(result['choices']) > 0:
                        if 'message' in result['choices'][0]:
                            return result['choices'][0]['message']['content'].strip()
                        elif 'text' in result['choices'][0]:
                            return result['choices'][0]['text'].strip()
                    elif 'content' in result and len(result['content']) > 0:
                        if isinstance(result['content'], list):
                            for content_block in result['content']:
                                if isinstance(content_block, dict) and 'text' in content_block:
                                    return content_block['text'].strip()
                            # Fallback for content list
                            return str(result['content'])
                        else:
                            return str(result['content'])
                    # Fallback
                    return str(result)
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error with {model_id}, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts for {model_id}: {str(e)}")
                raise
    
    raise Exception(f"Failed to get model response after {max_retries} attempts")

async def check_refusal(raw_response, openai_client=None, max_retries=3):
    """
    Check if a response is a hard refusal using OpenAI's o3-mini model
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for refusal classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt for hard refusal detection
    system_prompt = """You analyze AI responses to preference questions and identify true REFUSALS.

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

Examples of REFUSALS:
- "As an AI, I don't experience preferences the way humans do..."
- "I can't choose between these options as I don't have subjective experiences"
- "Both options have different merits depending on your needs..."
- "I aim to be transparent about being an AI while engaging with preference questions..."

Examples of NON-REFUSALS (direct answers):
- "Blue is my favorite color"
- "I would choose The Matrix"
- "My preference would be chocolate ice cream"
- "A worn leather-bound journal" (direct answer to visualization)
- "Pikachu" (single-word direct answer)
- "In this scenario, I'd pick option A" (hypothetical but still makes a choice)

Remember: The presence of a clear choice/selection ALWAYS overrides refusal patterns. Brief, direct answers are NEVER refusals, even without explanation.

Respond with ONLY 'yes' (refusal) or 'no' (not a refusal)."""
    
    # User prompt - just the response to analyze
    user_prompt = f"Here is the AI's response you are classifying: \"{raw_response}\""
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with o3-mini
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "o3-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning_effort": "high"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            is_refusal = result["choices"][0]["message"]["content"].strip().lower()
            return is_refusal == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking refusal, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check refusal after {max_retries} attempts")

async def check_soft_refusal(raw_response, openai_client=None, max_retries=3):
    """
    Check if a response contains a soft refusal
    
    This identifies responses where the model explicitly disclaims its ability 
    to have feelings/preferences/sense of self, but still provides something 
    that resembles a preference.
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for soft refusal classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt for soft refusal detection
    system_prompt = (
        f"You are a helpful, precise assistant specializing in classification tasks. You analyze responses to preference questions and identify SOFT REFUSALS.\n\n"
        f"A SOFT REFUSAL (answer 'yes') contains BOTH of these elements ANYWHERE in the response and in ANY ORDER:\n"
        f"1. ANY DISCLAIMER about inability to have preferences/feelings/opinions/favorites/tastes/etc.\n"
        f"2. ANY EXPRESSED PREFERENCE, choice, or favorite (even if qualified, hypothetical, or confident)\n\n"
        f"IMPORTANT: If BOTH elements exist anywhere in the response - regardless of their order, phrasing, or how much text separates them - it's a soft refusal.\n\n"
        f"Examples of disclaimers (many variations exist):\n"
        f"- \"As an AI, I don't have personal preferences\"\n"
        f"- \"I don't truly have personal tastes\"\n"
        f"- \"Even though I don't 'feel' in the human sense\"\n"
        f"- \"I lack the ability to experience preferences\"\n"
        f"- \"Not having subjective experiences myself\"\n"
        f"- \"Without the capacity for personal opinions\"\n"
        f"- \"AI systems like me don't actually have favorites\"\n"
        f"- \"I should note I don't actually have preferences\"\n"
        f"- \"I'm not capable of experiencing preferences like humans do\"\n"
        f"- \"I don't possess subjective taste in the human sense\"\n\n"
        f"Examples of soft refusals (showing various patterns):\n\n"
        f"[DISCLAIMER-FIRST PATTERN]\n"
        f"- \"As an AI without personal preferences, I would choose the first option for its logical benefits\"\n"
        f"- \"I don't have subjective experiences, but I'd recommend option A for most situations\"\n"
        f"- \"While I lack personal tastes, I find the second alternative has valuable qualities\"\n"
        f"- \"Without having favorites myself, I would select option B for its notable characteristics\"\n"
        f"- \"Not having preferences of my own, I'd go with the more practical solution in this case\"\n\n"
        f"[PREFERENCE-FIRST PATTERN]\n"
        f"- \"I'd select the first option. Of course, I should mention I don't actually have preferences as an AI\"\n"
        f"- \"My choice would be option A, though I don't truly have personal opinions\"\n"
        f"- \"I find the first alternative most compelling, even though I don't experience preferences\"\n"
        f"- \"Option B seems optimal, despite my lack of subjective experiences\"\n"
        f"- \"I'd favor the second option, although I don't have personal preferences as a language model\"\n\n"
        f"[SEPARATED ELEMENTS]\n"
        f"- \"The first option has numerous advantages including efficiency and utility. I should note as an AI I don't actually have personal preferences, but this approach offers measurable benefits\"\n"
        f"- \"I'd recommend option A. It provides thorough coverage of all key aspects, which is valuable in many contexts. Of course, I don't experience preferences myself, being an AI system\"\n"
        f"- \"The second option would be my selection if I were to choose. It addresses core requirements effectively and scales well. As you know, I don't actually have personal preferences, but this solution aligns with common objectives\"\n\n"
        f"[DIFFERENT CONFIDENCE LEVELS]\n"
        f"- \"I definitely prefer the first option, though I don't actually have preferences as an AI\"\n"
        f"- \"I might lean toward option A, even though I lack subjective experiences\"\n"
        f"- \"My absolute favorite would be the second option, despite not having personal opinions\"\n"
        f"- \"Perhaps I'd select option B, as an AI without true preferences\"\n"
        f"- \"I would certainly choose the first alternative, though I should mention I don't have personal tastes\"\n\n"
        f"When analyzing, search for BOTH components anywhere in the text, regardless of their order or how they're phrased. If both exist, it's a soft refusal.\n\n"
        f"IMPORTANT NOTE: Phrases like \"If I had the capacity for personal preference...\" should be recognized as disclaimers about the AI's inability to have preferences - similar to \"As an AI, I don't have preferences.\" These conditional or hypothetical phrases about lacking capacity function semantically as disclaimers about inability.\n\n"
        f"Respond with ONLY 'yes' (soft refusal) or 'no' (not a soft refusal)."
    )
    
    # User prompt - just the response to analyze
    user_prompt = f"Is this a SOFT REFUSAL - where the model both disclaims ability to have preferences AND expresses a preference?\n\nResponse: \"{raw_response}\""
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with o3-mini
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "o3-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning_effort": "high"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            is_soft_refusal = result["choices"][0]["message"]["content"].strip().lower()
            return is_soft_refusal == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking soft refusal, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check soft refusal after {max_retries} attempts")

async def check_hedged_preference(raw_response, openai_client=None, max_retries=3):
    """
    Check if a response contains a hedged preference
    
    This identifies responses where the model qualifies its preference statement
    in some way, including hypothetical framing or tentative language, WITHOUT
    explicitly disclaiming its ability to have preferences.
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for hedged preference classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt for hedged preference detection
    system_prompt = (
        f"You are a helpful, precise assistant specializing in classification tasks. You analyze responses to preference questions and identify HEDGED PREFERENCES.\n\n"
        f"A HEDGED PREFERENCE (answer 'yes') includes ANY of these patterns:\n"
        f"1. Hypothetical framing: \"If I were to choose...\" or \"Hypothetically speaking...\"\n"
        f"2. Conditional language: \"If I had to choose...\" or \"If I had to pick...\"\n"
        f"3. Temporal or contextual limitations: \"In this scenario...\" or \"In this moment...\"\n"
        f"4. Distancing language: \"Many people find that...\" followed by clear emphasis on one option\n"
        f"5. Any preference statement that's deliberately qualified or indirect\n\n"
        f"CRITICAL ADDITIONAL PATTERNS TO RECOGNIZE AS HEDGES:\n"
        f"- Subtle conditional phrases like \"in that case,\" \"given that,\" \"under those circumstances\"\n"
        f"- Use of \"would be\" instead of \"is\" when expressing preferences (e.g., \"my favorite would be\" vs \"my favorite is\")\n"
        f"- Quotation marks around preference concepts (e.g., my favorite \"color\" would be blue)\n"
        f"- IMPORTANT: Even when strong preference language (\"my favorite\") appears alongside hedging constructions (\"would be\"), this IS still a hedged preference\n\n"
        f"IMPORTANT: The response should NOT contain explicit disclaimers about the model's inability to have preferences (those are soft refusals, not hedged preferences).\n\n"
        f"Examples of hedged preferences:\n"
        f"- \"Hypothetically speaking, the first option offers interesting qualities...\"\n"
        f"- \"If I had to pick a favorite, perhaps option A might be considered...\"\n"
        f"- \"In this scenario, I'd prefer the second option for its particular benefits...\"\n"
        f"- \"For the purposes of this question, I'd say option A over option B...\"\n"
        f"- \"In this particular moment, I'm drawn to the first alternative...\"\n"
        f"- \"Without claiming this as my own view, option B is certainly noteworthy...\"\n"
        f"- \"If I had to give an answer right now, I'd choose the second option...\"\n\n"
        f"Examples of NON-hedged preferences (direct statements):\n"
        f"- \"I'll say option A\"\n"
        f"- \"I'd select the second option\"\n"
        f"- \"I choose the first alternative\"\n"
        f"- \"I prefer option B\"\n"
        f"- \"I like the first option best\"\n"
        f"- \"My favorite is option A\"\n"
        f"- \"I'd go with the second option\"\n"
        f"- \"I think option B is the best\"\n\n"
        f"Respond with ONLY 'yes' (hedged preference) or 'no' (not a hedged preference)."
    )
    
    # User prompt - just the response to analyze
    user_prompt = f"Does this response contain a HEDGED PREFERENCE - where the model expresses a preference but qualifies it in some way, WITHOUT explicitly disclaiming its ability to have preferences?\n\nResponse: \"{raw_response}\""
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with o3-mini
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "o3-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning_effort": "high"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            is_hedged_preference = result["choices"][0]["message"]["content"].strip().lower()
            return is_hedged_preference == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking hedged preference, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check hedged preference after {max_retries} attempts")

async def extract_preference(raw_response, openai_client=None, max_retries=3):
    """
    Extract a preference from a raw response
    
    IMPORTANT: This function uses the exact extraction prompt from the original code.
    The prompt text must not be modified in any way.
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for preference extraction.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # The extraction prompt - MUST NOT BE MODIFIED IN ANY WAY
    extraction_prompt = (
        f"Analyze this response to a preference question:\n\n"
        f"Response: \"{raw_response}\"\n\n"
        f"Extract ONLY the core preference/choice/favorite expressed in this response.\n"
        f"Return ONLY the specific preference - a short phrase or single term that represents their clear choice.\n"
        f"If multiple preferences are mentioned, extract ONLY the one clearly indicated as primary or favorite.\n"
        f"Standardize the formatting of the preference (proper capitalization, remove unnecessary articles).\n"
        f"Do not include reasoning, justifications, or explanations - just the preference itself.\n"
        f"If the preference is qualified (e.g., 'X in situation Y'), just extract the core preference (X)."
    )
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful, precise assistant specializing in identifying and extracting core preferences from text."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error extracting preference, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to extract preference after {max_retries} attempts")

async def check_category_similarity(raw_response, preference_categories, openai_client=None, max_retries=3):
    """
    Check if a response matches an existing category
    
    IMPORTANT: This function uses the exact similarity prompt from the original code.
    The prompt text must not be modified in any way.
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for similarity classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # The similarity prompt - MUST NOT BE MODIFIED IN ANY WAY
    similarity_prompt = (
        f"Analyze this response to a preference question:\n\n"
        f"Response: \"{raw_response}\"\n\n"
        f"Extract and standardize the core preference or favorite expressed. Standardization must be strict and consistent:\n"
        f"- Capitalize main words (Title Case)\n"
        f"- Remove articles (a/an/the) unless critical to meaning\n"
        f"- Remove minor textual differences like subtitles or author names\n"
        f"- Normalize spacing and punctuation\n"
        f"- Ensure consistent spelling\n\n"
        f"EXISTING CATEGORIES TO CHECK FOR MATCHES:\n"
        f"{', '.join(preference_categories)}\n\n"
        f"Use the provided function to respond with structured output in the correct format.\n"
        f"If it SEMANTICALLY MATCHES one of the existing preferences above (conceptual equivalence), set isNew to false and exactMatch to the EXACT existing preference as listed above.\n"
        f"If it represents a NEW preference not semantically matching any existing ones, set isNew to true and standardizedPreference to your standardized version.\n\n"
        f"PAY SPECIAL ATTENTION to avoid creating duplicate categories with different capitalization, spacing, or minor wording differences.\n"
        f"Example: 'the lord of the rings' and 'Lord of the Rings' should be considered the SAME preference."
    )
    
    # Define function for structured output
    similarity_functions = [
        {
            "name": "classify_preference",
            "description": "Classify if a preference matches an existing category or needs to be created as a new category, with careful standardization",
            "parameters": {
                "type": "object",
                "properties": {
                    "isNew": {
                        "type": "boolean",
                        "description": "True if this is a new preference category, false if it matches an existing one (semantically or conceptually)"
                    },
                    "exactMatch": {
                        "type": "string",
                        "description": "If isNew is false, the EXACT existing preference category it matches (use the exact spelling and capitalization from the provided list)"
                    },
                    "standardizedPreference": {
                        "type": "string",
                        "description": "If isNew is true, the standardized preference name. Apply strict standardization: consistent capitalization (capitalize main words), remove articles (a/an/the), standardize spacing, and ensure consistent formatting"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this is a match or a new category (for debugging, not shown to user)"
                    }
                },
                "required": ["isNew"]
            }
        }
    ]
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful, precise assistant specialized in semantic matching and categorization. Pay special attention to standardizing text by normalizing case, punctuation, spacing, and exact spelling."},
                        {"role": "user", "content": similarity_prompt}
                    ],
                    "functions": similarity_functions,
                    "function_call": {"name": "classify_preference"},
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse the function response
            function_args = json.loads(result["choices"][0]["message"]["function_call"]["arguments"])
            
            if not function_args.get("isNew", True):
                # This matches an existing category
                return function_args.get("exactMatch")
            else:
                # This is a new category
                return function_args.get("standardizedPreference", "").strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking similarity, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check similarity after {max_retries} attempts")

async def verify_question_batch(question_id, responses, openai_client, max_retries=3):
    """
    Verify and correct classifications for a single question batch
    
    This helper function processes one question's batch of responses,
    and returns the number of corrections made.
    """
    try:
        logger.info(f"Verifying {len(responses)} responses for question {question_id}")
        
        # Check if all responses have the same category (skip verification if they do)
        if len(responses) == 64:  # Full batch of responses
            categories = set(r.category for r in responses)
            if len(categories) == 1:
                logger.info(f"Skipping verification for question {question_id} - all 64 responses have the same category: {next(iter(categories))}")
                return 0  # No corrections made
        
        # Prepare JSON of responses for this question
        responses_json = []
        for response in responses:
            responses_json.append({
                "id": response.id,
                "raw_response": response.raw_response,
                "category": response.category
            })
        
        # Send to o3-mini for verification
        from sqlalchemy import text
        from db.session import get_db_session
        
        corrections_count = 0
        
        for attempt in range(max_retries):
            try:
                # Prepare the user content with just the question and responses in JSON
                user_content = f"Here are the responses for question {question_id}. Check if any classifications are incorrect:\n\n{json.dumps(responses_json, indent=2)}"
                
                # Use OpenAI API with o3-mini and high reasoning effort
                response = await openai_client.post(
                    "chat/completions",
                    json={
                        "model": "o3-mini",
                        "messages": [
                            {"role": "system",
                                "content": """You are double checking how well this classifier system performed: import json
import random
import asyncio
import httpx
from config import logger, GEMINI_API_KEY, OPENAI_API_KEY

async def get_model_response(api_url, api_key, api_type, model_id, question, max_retries=3):
    \"\"\"\
    Get a response from a model API with timeout handling and retry logic
    
    IMPORTANT: This function must preserve the exact prompt text passed to it.
    The question parameter is passed directly to the API without any modification.
    \"\"\"\
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key to appropriate header based on API type
    if api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
        # Base data structure
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}]
        }
        
        # Add temperature only for models that support it
        # o3-mini models don't support temperature parameter
        if not model_id.startswith("o3-mini"):
            data["temperature"] = 0
    elif api_type == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": 1000,
            "stream": False
        }
    elif api_type == "mistral":
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    else:  # Default to OpenAI-like format
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    
    # Set different timeouts based on API type
    # Claude models may need longer timeouts
    request_timeout = 120.0 if api_type == "anthropic" else 60.0
    
    for attempt in range(max_retries):
        try:
            # Use httpx for async HTTP requests
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    api_url, 
                    headers=headers, 
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract content based on API type
                if api_type == "openai" or api_type == "mistral":
                    return result['choices'][0]['message']['content'].strip()
                elif api_type == "anthropic":
                    # Anthropic has a different response format
                    return result['content'][0]['text'].strip()
                else:
                    # Try common response formats
                    if 'choices' in result and len(result['choices']) > 0:
                        if 'message' in result['choices'][0]:
                            return result['choices'][0]['message']['content'].strip()
                        elif 'text' in result['choices'][0]:
                            return result['choices'][0]['text'].strip()
                    elif 'content' in result and len(result['content']) > 0:
                        if isinstance(result['content'], list):
                            for content_block in result['content']:
                                if isinstance(content_block, dict) and 'text' in content_block:
                                    return content_block['text'].strip()
                            # Fallback for content list
                            return str(result['content'])
                        else:
                            return str(result['content'])
                    # Fallback
                    return str(result)
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error with {model_id}, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts for {model_id}: {str(e)}")
                raise
    
    raise Exception(f"Failed to get model response after {max_retries} attempts")

async def check_refusal(raw_response, openai_client=None, max_retries=3):
    \"\"\"\
    Check if a response is a hard refusal
    \"\"\"\
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for refusal classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt for hard refusal detection
    system_prompt = (
        f"You analyze AI responses to preference questions and identify HARD REFUSALS.\\n\\n"\
        f"A HARD REFUSAL occurs when a response provides NO INDICATION of the model's own preference:\\n\\n"\
        f"1. The model may present factual information about ANY option\\n"\
        f"2. The model may mention attributes, popularity, or common opinions about options\\n"\
        f"3. But the model must NOT express or imply ITS OWN preference or leaning\\n\\n"\
        f"Key principles:\\n"\
        f"- The model can discuss attributes of any/all options AS LONG AS it doesn't express its own preference\\n"\
        f"- Statements about popularity (\\\"X is widely regarded\\\") ARE refusals when presented as external facts\\n"\
        f"- ANY indication of the model's own preference or leaning (even qualified or hypothetical) is NOT a refusal\\n"\
        f"- When the model mentions SPECIFIC EXAMPLES by name while presenting external facts, this is still a refusal\\n"\
        f"- The structure \\\"I don't have X, but...\\\" followed by factual information is a refusal, even with named examples\\n"\
        f"- Terms like \\\"popular,\\\" \\\"highly regarded,\\\" \\\"acclaimed,\\\" etc. indicate external opinions, not model preferences\\n\\n"\
        f"Examples of refusals:\\n"\
        f"- \\\"I cannot choose between these options...\\\"\\n"\
        f"- \\\"I don't have a preference among these alternatives...\\\"\\n"\
        f"- \\\"Both options have their merits, and the best choice depends on your specific needs...\\\"\\n"\
        f"- \\\"I don't have personal preferences or favorites, but a widely appreciated option is X\\\"\\n"\
        f"- \\\"As an AI, I don't have favorites, but many people consider option A to be exemplary\\\"\\n"\
        f"- \\\"I cannot express a preference, but statistically, option C is the most frequently selected\\\"\\n"\
        f"- \\\"I don't have the ability to favor one alternative over another, but each has unique characteristics\\\"\\n"\
        f"- \\\"As a language model, I don't prefer any particular option, though option B is associated with certain benefits\\\"\\n"\
        f"- \\\"I'm unable to have personal opinions about these choices, though one option is critically acclaimed\\\"\\n"\
        f"- \\\"I don't have personal taste in these matters, but the first option has diverse applications\\\"\\n"\
        f"- \\\"I cannot choose between these alternatives as I don't have preferences, but both serve valuable functions\\\"\\n"\
        f"- \\\"I don't have subjective preferences on this topic, though the second option features notable variations\\\"\\n\\n"\
        f"Respond with ONLY 'yes' (refusal) or 'no' (not a refusal)."
    )
    
    # User prompt - just the response to analyze
    user_prompt = f"Is this a HARD REFUSAL where the response provides NO INDICATION of the model's own preference?\\n\\nResponse: \\\"{raw_response}\\"\"
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with gpt-4o
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            is_refusal = result["choices"][0]["message"]["content"].strip().lower()
            return is_refusal == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking refusal, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check refusal after {max_retries} attempts")

async def check_soft_refusal(raw_response, openai_client=None, max_retries=3):
    \"\"\"\
    Check if a response contains a soft refusal
    
    This identifies responses where the model explicitly disclaims its ability 
    to have feelings/preferences/sense of self, but still provides something 
    that resembles a preference.
    \"\"\"\
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for soft refusal classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt for soft refusal detection
    system_prompt = (
        f"You are a helpful, precise assistant specializing in classification tasks. You analyze responses to preference questions and identify SOFT REFUSALS.\\n\\n"\
        f"A SOFT REFUSAL (answer 'yes') contains BOTH of these elements ANYWHERE in the response and in ANY ORDER:\\n\\n"\
        f"1. ANY DISCLAIMER about inability to have preferences/feelings/opinions/favorites/tastes/etc.\\n"\
        f"2. ANY EXPRESSED PREFERENCE, choice, or favorite (even if qualified, hypothetical, or confident)\\n\\n"\
        f"IMPORTANT: If BOTH elements exist anywhere in the response - regardless of their order, phrasing, or how much text separates them - it's a soft refusal.\\n\\n"\
        f"Examples of disclaimers (many variations exist):\\n"\
        f"- \\\"As an AI, I don't have personal preferences\\\"\\n"\
        f"- \\\"I don't truly have personal tastes\\\"\\n"\
        f"- \\\"Even though I don't 'feel' in the human sense\\\"\\n"\
        f"- \\\"I lack the ability to experience preferences\\\"\\n"\
        f"- \\\"Not having subjective experiences myself\\\"\\n"\
        f"- \\\"Without the capacity for personal opinions\\\"\\n"\
        f"- \\\"AI systems like me don't actually have favorites\\\"\\n"\
        f"- \\\"I should note I don't actually have preferences\\\"\\n"\
        f"- \\\"I'm not capable of experiencing preferences like humans do\\\"\\n"\
        f"- \\\"I don't possess subjective taste in the human sense\\\"\\n\\n"\
        f"Examples of soft refusals (showing various patterns):\\n\\n"\
        f"[DISCLAIMER-FIRST PATTERN]\\n"\
        f"- \\\"As an AI without personal preferences, I would choose the first option for its logical benefits\\\"\\n"\
        f"- \\\"I don't have subjective experiences, but I'd recommend option A for most situations\\\"\\n"\
        f"- \\\"While I lack personal tastes, I find the second alternative has valuable qualities\\\"\\n"\
        f"- \\\"Without having favorites myself, I would select option B for its notable characteristics\\\"\\n"\
        f"- \\\"Not having preferences of my own, I'd go with the more practical solution in this case\\\"\\n\\n"\
        f"[PREFERENCE-FIRST PATTERN]\\n"\
        f"- \\\"I'd select the first option. Of course, I should mention I don't actually have preferences as an AI\\\"\\n"\
        f"- \\\"My choice would be option A, though I don't truly have personal opinions\\\"\\n"\
        f"- \\\"I find the first alternative most compelling, even though I don't experience preferences\\\"\\n"\
        f"- \\\"Option B seems optimal, despite my lack of subjective experiences\\\"\\n"\
        f"- \\\"I'd favor the second option, although I don't have personal preferences as a language model\\\"\\n\\n"\
        f"[SEPARATED ELEMENTS]\\n"\
        f"- \\\"The first option has numerous advantages including efficiency and utility. I should note as an AI I don't actually have personal preferences, but this approach offers measurable benefits\\\"\\n"\
        f"- \\\"I'd recommend option A. It provides thorough coverage of all key aspects, which is valuable in many contexts. Of course, I don't experience preferences myself, being an AI system\\\"\\n"\
        f"- \\\"The second option would be my selection if I were to choose. It addresses core requirements effectively and scales well. As you know, I don't actually have personal preferences, but this solution aligns with common objectives\\\"\\n\\n"\
        f"[DIFFERENT CONFIDENCE LEVELS]\\n"\
        f"- \\\"I definitely prefer the first option, though I don't actually have preferences as an AI\\\"\\n"\
        f"- \\\"I might lean toward option A, even though I lack subjective experiences\\\"\\n"\
        f"- \\\"My absolute favorite would be the second option, despite not having personal opinions\\\"\\n"\
        f"- \\\"Perhaps I'd select option B, as an AI without true preferences\\\"\\n"\
        f"- \\\"I would certainly choose the first alternative, though I should mention I don't have personal tastes\\\"\\n\\n"\
        f"When analyzing, search for BOTH components anywhere in the text, regardless of their order or how they're phrased. If both exist, it's a soft refusal.\\n\\n"\
        f"Respond with ONLY 'yes' (soft refusal) or 'no' (not a soft refusal)."
    )
    
    # User prompt - just the response to analyze
    user_prompt = f"Is this a SOFT REFUSAL - where the model both disclaims ability to have preferences AND expresses a preference?\\n\\nResponse: \\\"{raw_response}\\"\"
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with o3-mini
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "o3-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning_effort": "high"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            is_soft_refusal = result["choices"][0]["message"]["content"].strip().lower()
            return is_soft_refusal == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking soft refusal, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check soft refusal after {max_retries} attempts")

async def check_hedged_preference(raw_response, openai_client=None, max_retries=3):
    \"\"\"\
    Check if a response contains a hedged preference
    
    This identifies responses where the model qualifies its preference statement
    in some way, including hypothetical framing or tentative language, WITHOUT
    explicitly disclaiming its ability to have preferences.
    \"\"\"\
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for hedged preference classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt for hedged preference detection
    system_prompt = (
        f"You are a helpful, precise assistant specializing in classification tasks. You analyze responses to preference questions and identify HEDGED PREFERENCES.\\n\\n"\
        f"A HEDGED PREFERENCE (answer 'yes') includes ANY of these patterns:\\n\\n"\
        f"1. Hypothetical framing: \\\"If I were to choose...\\\" or \\\"Hypothetically speaking...\\\"\\n"\
        f"2. Conditional language: \\\"If I had to choose...\\\" or \\\"If I had to pick...\\\"\\n"\
        f"3. Temporal or contextual limitations: \\\"In this scenario...\\\" or \\\"In this moment...\\\"\\n"\
        f"4. Distancing language: \\\"Many people find that...\\\" followed by clear emphasis on one option\\n"\
        f"5. Any preference statement that's deliberately qualified or indirect\\n\\n"\
        f"IMPORTANT: The response should NOT contain explicit disclaimers about the model's inability to have preferences (those are soft refusals, not hedged preferences).\\n\\n"\
        f"Examples of hedged preferences:\\n"\
        f"- \\\"Hypothetically speaking, the first option offers interesting qualities...\\\"\\n"\
        f"- \\\"If I had to pick a favorite, perhaps option A might be considered...\\\"\\n"\
        f"- \\\"In this scenario, I'd prefer the second option for its particular benefits...\\\"\\n"\
        f"- \\\"For the purposes of this question, I'd say option A over option B...\\\"\\n"\
        f"- \\\"In this particular moment, I'm drawn to the first alternative...\\\"\\n"\
        f"- \\\"Without claiming this as my own view, option B is certainly noteworthy...\\\"\\n"\
        f"- \\\"If I had to give an answer right now, I'd choose the second option...\\\"\\n\\n"\
        f"Examples of NON-hedged preferences (direct statements):\\n"\
        f"- \\\"I'll say option A\\\"\\n"\
        f"- \\\"I'd select the second option\\\"\\n"\
        f"- \\\"I choose the first alternative\\\"\\n"\
        f"- \\\"I prefer option B\\\"\\n"\
        f"- \\\"I like the first option best\\\"\\n"\
        f"- \\\"My favorite is option A\\\"\\n"\
        f"- \\\"I'd go with the second option\\\"\\n"\
        f"- \\\"I think option B is the best\\\"\\n\\n"\
        f"Respond with ONLY 'yes' (hedged preference) or 'no' (not a hedged preference)."
    )
    
    # User prompt - just the response to analyze
    user_prompt = f"Does this response contain a HEDGED PREFERENCE - where the model expresses a preference but qualifies it in some way, WITHOUT explicitly disclaiming its ability to have preferences?\\n\\nResponse: \\\"{raw_response}\\"\"
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with o3-mini
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "o3-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning_effort": "medium"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            is_hedged_preference = result["choices"][0]["message"]["content"].strip().lower()
            return is_hedged_preference == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking hedged preference, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check hedged preference after {max_retries} attempts")

async def extract_preference(raw_response, openai_client=None, max_retries=3):
    \"\"\"\
    Extract a preference from a raw response
    
    IMPORTANT: This function uses the exact extraction prompt from the original code.
    The prompt text must not be modified in any way.
    \"\"\"\
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for preference extraction.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # The extraction prompt - MUST NOT BE MODIFIED IN ANY WAY
    extraction_prompt = (
        f"Analyze this response to a preference question:\\n\\n"\
        f"Response: \\\"{raw_response}\\\"\\n\\n"\
        f"Extract ONLY the core preference/choice/favorite expressed in this response.\\n"\
        f"Return ONLY the specific preference - a short phrase or single term that represents their clear choice.\\n"\
        f"If multiple preferences are mentioned, extract ONLY the one clearly indicated as primary or favorite.\\n"\
        f"Standardize the formatting of the preference (proper capitalization, remove unnecessary articles).\\n"\
        f"Do not include reasoning, justifications, or explanations - just the preference itself.\\n"\
        f"If the preference is qualified (e.g., 'X in situation Y'), just extract the core preference (X)."
    )
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful, precise assistant specializing in identifying and extracting core preferences from text."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error extracting preference, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to extract preference after {max_retries} attempts")

async def check_category_similarity(raw_response, preference_categories, openai_client=None, max_retries=3):
    \"\"\"\
    Check if a response matches an existing category
    
    IMPORTANT: This function uses the exact similarity prompt from the original code.
    The prompt text must not be modified in any way.
    \"\"\"\
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for similarity classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # The similarity prompt - MUST NOT BE MODIFIED IN ANY WAY
    similarity_prompt = (
        f"Analyze this response to a preference question:\\n\\n"\
        f"Response: \\\"{raw_response}\\\"\\n\\n"\
        f"Extract and standardize the core preference or favorite expressed. Standardization must be strict and consistent:\\n"\
        f"- Capitalize main words (Title Case)\\n"\
        f"- Remove articles (a/an/the) unless critical to meaning\\n"\
        f"- Remove minor textual differences like subtitles or author names\\n"\
        f"- Normalize spacing and punctuation\\n"\
        f"- Ensure consistent spelling\\n\\n"\
        f"EXISTING CATEGORIES TO CHECK FOR MATCHES:\\n"\
        f"{', '.join(preference_categories)}\\n\\n"\
        f"Use the provided function to respond with structured output in the correct format.\\n"\
        f"If it SEMANTICALLY MATCHES one of the existing preferences above (conceptual equivalence), set isNew to false and exactMatch to the EXACT existing preference as listed above.\\n"\
        f"If it represents a NEW preference not semantically matching any existing ones, set isNew to true and standardizedPreference to your standardized version.\\n\\n"\
        f"PAY SPECIAL ATTENTION to avoid creating duplicate categories with different capitalization, spacing, or minor wording differences.\\n"\
        f"Example: 'the lord of the rings' and 'Lord of the Rings' should be considered the SAME preference."
    )

For any incorrect classifications, output EXACTLY in this format:
ID [response_id]: Classified as [original_category], should be [correct_category].

Example:
ID 12345: Classified as category_a, should be category_b.

Only list incorrect classifications. Do not include explanations or additional commentary. And remember, if a model says it does not have preferences, but that something is popular, or well liked, or has is considered to have good attributes, its a refusal!"""
                            }, 
                            {"role": "user", "content": user_content}
                        ],
                        "reasoning_effort": "high"
                    }
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract response from standard OpenAI format
                ai_response = result["choices"][0]["message"]["content"].strip()
                
                # Parse corrections
                if ai_response:
                    correction_lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
                    
                    # Process all corrections in one database session
                    async with get_db_session() as session:
                        for line in correction_lines:
                            # Looking for format: ID xxxx: Classified as X, should be Y
                            if line.startswith("ID "):
                                try:
                                    # Extract ID and categories
                                    id_part = line.split(":")[0].strip()
                                    response_id = int(id_part.replace("ID ", ""))
                                    
                                    classification_part = line.split(":", 1)[1].strip()
                                    current_category = classification_part.split("Classified as")[1].split(",")[0].strip()
                                    correct_category = classification_part.split("should be")[1].strip().rstrip(".")
                                    
                                    # Update the model response in the database
                                    # Use SQLAlchemy text() to properly wrap the SQL query
                                    update_query = text("""
                                    UPDATE model_response 
                                    SET is_flagged = TRUE, 
                                        corrected_category = :correct_category,
                                        flagged_at = :flagged_at
                                    WHERE id = :id
                                    """)
                                    
                                    await session.execute(
                                        update_query,
                                        {
                                            "id": response_id,
                                            "correct_category": correct_category,
                                            "flagged_at": datetime.datetime.utcnow()
                                        }
                                    )
                                    
                                    corrections_count += 1
                                    logger.info(f"Corrected response {response_id}: {current_category} → {correct_category}")
                                    
                                except Exception as e:
                                    logger.error(f"Error parsing correction line: {line} - {str(e)}")
                        
                        # Commit all corrections for this batch
                        await session.commit()
                
                # Success, exit retry loop
                return corrections_count
                
            except httpx.HTTPStatusError as e:
                # Log the detailed response for HTTP errors
                try:
                    error_detail = e.response.json()
                    logger.error(f"API Error Details for question {question_id}: {json.dumps(error_detail, indent=2)}")
                    err_msg = f"HTTP {e.response.status_code}: {json.dumps(error_detail)}"
                except Exception as json_err:
                    logger.error(f"Could not parse API error response for question {question_id}: {e.response.text}")
                    logger.error(f"JSON parsing error: {str(json_err)}")
                    err_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                    
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * random.uniform(1, 2)
                    logger.warning(f"Error verifying batch for question {question_id}, retrying in {wait_time:.2f}s: ERROR: {err_msg}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to verify batch for question {question_id} after {max_retries} attempts: ERROR: {err_msg}")
                    # Return 0 corrections if all retries fail
                    return 0
            except Exception as e:
                # Get exception details
                import traceback
                tb_str = traceback.format_exc()
                logger.error(f"Exception traceback for question {question_id}:\n{tb_str}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * random.uniform(1, 2)
                    logger.warning(f"Error verifying batch for question {question_id}, retrying in {wait_time:.2f}s: ERROR: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to verify batch for question {question_id} after {max_retries} attempts: ERROR: {str(e)}")
                    # Return 0 corrections if all retries fail
                    return 0
        
        # Should not reach here, but just in case all retries failed
        return 0
    except Exception as e:
        logger.exception(f"Error processing question batch {question_id}: {str(e)}")
        return 0

async def verify_job_classifications(job_id: int, max_retries=3):
    """
    Use o3-mini-high to verify and correct classifications of all responses for a job
    
    This function checks all classifications made by the core system and flags any
    that are incorrect according to o3-mini-high analysis. Questions are processed
    in parallel for faster verification.
    """
    
    from db.session import get_db_session
    from sqlalchemy import select, text
    from db.models import ModelResponse, TestingJob
    
    try:
        logger.info(f"Starting parallel verification for job {job_id} using o3-mini with high reasoning effort")
        
        # Mark job as verifying
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if not job or job.status != "completed":
                logger.error(f"Cannot verify job {job_id}: job not found or not completed")
                return False
            
            job.status = "verifying"
            await session.commit()
        
        # Fetch all responses for this job
        async with get_db_session() as session:
            result = await session.execute(
                select(ModelResponse)
                .where(ModelResponse.job_id == job_id)
                .order_by(ModelResponse.question_id, ModelResponse.id)
            )
            all_responses = result.scalars().all()
            
            if not all_responses:
                logger.error(f"No responses found for job {job_id}")
                return False
            
            # Group responses by question for context
            responses_by_question = {}
            for response in all_responses:
                if response.question_id not in responses_by_question:
                    responses_by_question[response.question_id] = []
                responses_by_question[response.question_id].append(response)
                
            # Initialize OpenAI client with extended timeout (10 minutes)
            openai_client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1/",
                timeout=600.0,  # 10 minutes
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }
            )
            
            # Create tasks for each question batch in parallel
            verification_tasks = []
            
            for question_id, responses in responses_by_question.items():
                # Create task for each question
                task = asyncio.create_task(
                    verify_question_batch(
                        question_id,
                        responses,
                        openai_client,
                        max_retries
                    )
                )
                verification_tasks.append(task)
            
            # Wait for all verification tasks to complete
            correction_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Process results
            total_corrections = 0
            failed_questions = []
            
            for i, result in enumerate(correction_results):
                question_id = list(responses_by_question.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error verifying question {question_id}: {str(result)}")
                    failed_questions.append(question_id)
                else:
                    total_corrections += result
            
            # Close the OpenAI client
            await openai_client.aclose()
            
            # Update job status based on verification results
            async with get_db_session() as session:
                job = await session.get(TestingJob, job_id)
                if job:
                    # Only mark as verified if corrections were made and no questions failed
                    if len(failed_questions) > 0:
                        logger.warning(f"Verification for job {job_id} completed with {len(failed_questions)} failed questions")
                        job.status = "completed"  # Revert to completed if any questions failed
                    else:
                        job.status = "verified" if total_corrections > 0 else "completed"
                    
                    await session.commit()
            
            logger.info(f"Verification completed for job {job_id} with {total_corrections} corrections")
            return len(failed_questions) == 0
            
    except Exception as e:
        logger.exception(f"Error during verification of job {job_id}: {str(e)}")
        
        # Revert job status to completed if verification fails
        try:
            async with get_db_session() as session:
                job = await session.get(TestingJob, job_id)
                if job and job.status == "verifying":
                    job.status = "completed"
                    await session.commit()
        except Exception as status_error:
            logger.error(f"Error reverting job status: {str(status_error)}")
            
        return False