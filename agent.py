import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserProfile
from langchain_openai import ChatOpenAI

load_dotenv()

# We use the credentials from the .env file
dc_email = os.getenv('DATACAMP_EMAIL')
dc_pass = os.getenv('DATACAMP_PASSWORD')
dc_course_url = os.getenv('DATACAMP_COURSE_URL')

# To avoid the LLM leaking credentials in logs, we declare them as sensitive
sensitive_data = {
    'DC_EMAIL': dc_email,
    'DC_PASS': dc_pass
}

TASK = f"""
You are an autonomous AI designed to complete a DataCamp course. 
Here are your strict instructions:

1. Navigate to: {dc_course_url}
2. Login if necessary:
   - If prompted to log in, use the email 'DC_EMAIL' and password 'DC_PASS'.
   - The browser maintains state, so you may already be logged in.
3. Once in the course, work through all lessons one by one:
   - VIDEO LESSONS: Click "Next", "Continue", or "Got it" to skip the video and proceed to the next item.
   - MULTIPLE CHOICE: Read the question, understand the context, select the correct answer, and submit it.
   - INTERACTIVE CODE EXERCISES:
     * Read the instructions carefully.
     * Write the correct code (Python/R/SQL) in the editor carefully.
     * Click 'Run Code' to test it. Read the output. If it fails, fix your code.
     * Once it works, click 'Submit Answer' or 'Continue'.
4. Repeat this process recursively for every chapter.
5. If you encounter an unexpected popup (e.g., feedback surveys), dismiss it.
6. Stop only when the course is 100% complete and you see the certificate.

Be systematic and robust. If an element takes time to appear, do not panic, just look for it again.
"""

async def main():
    llm = ChatOpenAI(
        model="auto",
        base_url="http://localhost:8402/v1",
        api_key="x402-proxy-handles-auth", # CLAW Router needs some dummy string
        temperature=0.2, # Lower temperature for better code reasoning
    )

    browser = Browser(
        browser_profile=BrowserProfile(
            headless=True,  # Runs invisibly on the Pi
            # Persists cookies/session across runs so login only happens once
            user_data_dir='./browser_data'
        )
    )

    agent = Agent(
        task=TASK,
        llm=llm,
        browser=browser,
        sensitive_data=sensitive_data, # Replaces placeholders in DOM/actions
        max_actions_per_step=8, # Allows slightly more complex page interactions
    )

    print("🦞 OpenCLAW DataCamp Agent starting...")
    print(f"Targeting course: {dc_course_url}")
    
    # max_steps=200 should give it enough budget to finish a full course
    result = await agent.run(max_steps=200)
    
    print("✅ Run Complete!")
    await browser.close()
    
if __name__ == "__main__":
    asyncio.run(main())
