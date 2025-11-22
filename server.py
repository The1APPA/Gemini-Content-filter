import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List
import asyncio
from dotenv import load_dotenv
from google import genai
import json

# Logging setup
logging.basicConfig(level=logging.INFO)


# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI setup
app = FastAPI()

# Enable CORS for local HTML file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# Review models
class ReviewIn(BaseModel):
    name: str | None = "Anonymous"
    rating: int
    text: str

class ReviewOut(BaseModel):
    name: str
    rating: int
    text: str
    sentiment: str
    timestamp: str

# In-memory store for reviews
reviews: List[ReviewOut] = []

# Gemini moderation & authenticity check
async def analyze_with_gemini(text: str) -> dict:
    prompt = f"""
    Analyze the following product review for both appropriateness and authenticity.

    Review: "{text}"

    Return ONLY a JSON object with the following keys:
    {{
      "safe_to_post": true or false,
      "likely_author": "human" or "bot",
      "reason": "Brief explanation of the decision"
    }}

    Rules:
    - "safe_to_post" is false if there is profanity, spam, or toxic language.
    - "likely_author" should be "bot" if the writing seems repetitive, artificial, or unnatural.
    - Keep the explanation short and clear.
    """

    def call_gemini():
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response

    response = await asyncio.to_thread(call_gemini)
    raw_text = response.text.strip()
    logging.info("Gemini raw response: %s", raw_text)

    # --- Fix: clean up markdown fences if present ---
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")  # remove backticks
        # also remove possible language hint like ```json
        cleaned = cleaned.replace("json", "", 1).strip()
    cleaned = cleaned.strip("`").strip()

    try:
        result = json.loads(cleaned)
    except Exception as e:
        logging.error("JSON parsing error: %s", e)
        result = {
            "safe_to_post": False,
            "likely_author": "unknown",
            "reason": "Could not parse Gemini response."
        }
    return result

# API endpoints
@app.post("/analyze")
async def analyze_review(review: ReviewIn):
    try:
        if not (1 <= review.rating <= 5):
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

        # Gemini moderation + authenticity
        analysis = await analyze_with_gemini(review.text)

        safe = analysis.get("safe_to_post", False)
        author = analysis.get("likely_author", "unknown")
        reason = analysis.get("reason", "")

        if not safe:
            return {"appropriate": False, "message": f"Review rejected: unsafe content. ({reason})"}

        if author != "human":
            return {"appropriate": False, "message": f"Review rejected: detected as {author}. ({reason})"}

        # Determine sentiment from rating
        sentiment = "positive" if review.rating >= 4 else ("negative" if review.rating <= 2 else "neutral")

        out = ReviewOut(
            name=review.name or "Anonymous",
            rating=review.rating,
            text=review.text,
            sentiment=sentiment,
            timestamp=datetime.utcnow().isoformat()
        )
        reviews.append(out)

        logging.info("Accepted review: %s", out.json())

        return {"appropriate": True, "review": out}

    except Exception as e:
        logging.exception("Error in /analyze endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reviews")
async def get_reviews():
    return reviews

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
