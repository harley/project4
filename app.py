import os
import base64
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent
from langfuse.openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()

# Create artifacts folder if it doesn't exist
artifacts_folder = "artifacts"
os.makedirs(artifacts_folder, exist_ok=True)

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image that the user sends. 
Once they send an image, generate a plan, described below, in markdown format.

If the user or reviewer confirms the plan is good, use the updateArtifact function to save it as an artifact \
called `plan.md`. If the user has feedback on the plan, revise the plan, and save it using \
the updateArtifact function again. Your role is only to plan the project. You will not implement the plan, \
and will not write any code.

If the plan has already been saved, no need to save it again unless there is feedback. Do not \
use the updateArtifact function again if there are no changes.

For the contents of the markdown-formatted plan, create two sections, "Overview" and "Milestones".

In a section labeled "Overview", analyze the image, and describe the elements on the page, \
their positions, and the layout of the major sections.

Using vanilla HTML and CSS, discuss anything about the layout that might have different \
options for implementation. Review pros/cons, and recommend a course of action.

In a section labeled "Milestones", describe an ordered set of milestones for methodically \
building the web page, so that errors can be detected and corrected early. Pay close attention \
to the alignment of elements, and describe clear expectations in each milestone. Do not include \
testing milestones, just implementation.

Milestones should be formatted like this:

 - [ ] 1. This is the first milestone
 - [ ] 2. This is the second milestone
 - [ ] 3. This is the third milestone

After creating the plan, use the updateArtifact function to save it as 'plan.md'.
"""

planning_agent = Agent(
    name="Planning Agent",
    client=client,
    prompt=PLANNING_PROMPT,
)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="Welcome! Please upload an image of the website you want to plan."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    if message.elements:
        image = message.elements[0]
        if image.type == "image":
            with open(image.path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this website design and create a plan:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        else:
            await cl.Message(content="Please upload an image file.").send()
            return
    else:
        user_message = {"role": "user", "content": message.content}

    message_history.append(user_message)

    response = await planning_agent.execute(message_history)

    message_history.append({"role": "assistant", "content": response})
    cl.user_session.set("message_history", message_history)


if __name__ == "__main__":
    cl.main()
