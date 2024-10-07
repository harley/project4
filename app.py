import os
import base64
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent
from langfuse.openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image. Generate a plan, \
described below, in markdown format.

In a section labeled "Overview", analyze the image, and describe the elements on the page, \
their positions, and the layout of the major sections.

Using vanilla HTML and CSS, discuss anything about the layout that might have different \
options for implementation. Review pros/cons, and recommend a course of action.

In a section labeled "Milestones", describe an ordered set of milestones for methodically \
building the web page, so that errors can be detected and corrected early. Pay close attention \
to the aligment of elements, and describe clear expectations in each milestone. Do not include \
testing milestones, just implementation.

Milestones should be formatted like this:

 - [ ] 1. This is the first milestone
 - [ ] 2. This is the second milestone
 - [ ] 3. This is the third milestone
"""

# Create an instance of the Agent class
planning_agent = Agent(name="Planning Agent", client=client, prompt=PLANNING_PROMPT)


@cl.on_chat_start
async def on_chat_start():
    planning_agent.clear_message_history()
    await cl.Message(
        content="Welcome! Please upload an image of the website you want to plan."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    if message.elements:
        # Handle image upload
        image = message.elements[0]
        if image.type == "image":
            # Read and encode the image
            with open(image.path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Prepare the message with the image
            image_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this website design:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }

            response_message = cl.Message(content="")
            await response_message.send()

            async for token in planning_agent.generate_response(image_message):
                await response_message.stream_token(token)

            await response_message.update()
        else:
            await cl.Message(content="Please upload an image file.").send()
    else:
        # Handle regular text messages
        response_message = cl.Message(content="")
        await response_message.send()

        async for token in planning_agent.generate_response(message.content):
            await response_message.stream_token(token)

        await response_message.update()


if __name__ == "__main__":
    cl.main()
