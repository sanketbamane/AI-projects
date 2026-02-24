from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate_question(skill: str, difficulty: int):
    prompt = f"""
    You are a senior technical interviewer.
    Ask one question about {skill}.
    Difficulty level: {difficulty}/5.
    Only return the question text.
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content
