import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserProfile
from langchain_openai import ChatOpenAI as _ChatOpenAI
from pydantic import ConfigDict

# browser-use compatibility shim for langchain_openai.ChatOpenAI:
#   1. llm.provider — browser-use checks this for feature flags / telemetry
#   2. llm.model — browser-use accesses this, but langchain stores it as model_name
#   3. setattr(llm, 'ainvoke', ...) — browser-use monkey-patches for cost tracking,
#      but Pydantic v2 blocks setting unknown attributes
class ChatOpenAI(_ChatOpenAI):
    model_config = ConfigDict(extra='allow')
    provider: str = 'openai'

    @property
    def model(self) -> str:
        """browser-use accesses llm.model, langchain stores it as model_name."""
        return self.model_name

    def __setattr__(self, name: str, value):
        try:
            super().__setattr__(name, value)
        except (ValueError, AttributeError):
            object.__setattr__(self, name, value)

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
        api_key="x402-proxy-handles-auth",
        temperature=0.2,
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
        generate_gif="datacamp_run.gif", # Extremely helpful for debugging what happened without screen!
    )

    print("🦞 OpenCLAW DataCamp Agent starting...")
    print(f"Targeting course: {dc_course_url}")
    
    # max_steps=200 should give it enough budget to finish a full course
    result = await agent.run(max_steps=200)
    
    print("✅ Run Complete!")
    await browser.kill()
    
if __name__ == "__main__":
    asyncio.run(main())
