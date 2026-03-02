import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

load_dotenv()

TASK = f"""
You are completing a DataCamp course automatically. Here are your instructions:

1. Go to {os.getenv('DATACAMP_COURSE_URL')}
2. If you see a login page, log in with:
   - Email: {os.getenv('DATACAMP_EMAIL')}
   - Password: {os.getenv('DATACAMP_PASSWORD')}
3. Then work through the course chapter by chapter:
   - VIDEO lessons: Click the "Next" or "Continue" button to skip past them.
   - CODE exercises: Read the instructions carefully, write correct Python/R/SQL 
     code in the editor, click "Run Code" to test it, then click "Submit Answer".
   - MULTIPLE CHOICE: Read the question and select the correct answer, then submit.
4. Keep going until the entire course is marked as complete.
5. If you get an error on a code exercise, fix the code and try again.
"""

async def main():
    llm = ChatOpenAI(
        model="auto",
        base_url="http://localhost:8402/v1",
        api_key="dummy",
        temperature=0.3,
    )

    browser = Browser(
        config=BrowserConfig(
            headless=True,  # No screen on Pi — run invisibly
        )
    )

    agent = Agent(
        task=TASK,
        llm=llm,
        browser=browser,
        max_actions_per_step=5,
    )

    print("🦞 DataCamp Agent starting...")
    result = await agent.run(max_steps=200)
    print("✅ Done!", result)

if __name__ == "__main__":
    asyncio.run(main())
