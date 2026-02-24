from app.services.llm_service import generate_question

async def next_question(session, skill="Python"):
    question = await generate_question(skill, session.difficulty_level)
    return question
