import os
import base64
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent
from langfuse.openai import AsyncOpenAI
import json

load_dotenv()

client = AsyncOpenAI()

# Create artifacts folder if it doesn't exist
artifacts_folder = "artifacts"
os.makedirs(artifacts_folder, exist_ok=True)

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image that the user sends. 
Once they send an image, generate a plan, described below, in markdown format.

If the user or reviewer confirms the plan is good, available tools to save it as an artifact \
called `plan.md`. If the user has feedback on the plan, revise the plan, and save it using \
the tool again. A tool is available to update the artifact. Your role is only to plan the \
project. You will not implement the plan, and will not write any code.

If the plan has already been saved, no need to save it again unless there is feedback. Do not \
use the tool again if there are no changes.

For the contents of the markdown-formatted plan, create two sections, "Overview" and "Milestones".

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


def get_artifact_contents():
    artifact_contents = {}
    for filename in os.listdir(artifacts_folder):
        with open(os.path.join(artifacts_folder, filename), "r") as f:
            artifact_contents[filename] = f.read()
    return artifact_contents


def update_artifact(filename, content):
    with open(os.path.join(artifacts_folder, filename), "w") as f:
        f.write(content)
    return f"File {filename} has been updated."


functions = [
    {
        "name": "updateArtifact",
        "description": "Update or create an artifact file",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to update or create",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["filename", "content"],
        },
    }
]

planning_agent = Agent(
    name="Planning Agent",
    client=client,
    prompt=PLANNING_PROMPT,
    functions=functions,
)


@cl.on_chat_start
async def on_chat_start():
    planning_agent.clear_message_history()
    await cl.Message(
        content="Welcome! Please upload an image of the website you want to plan."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    response_message = cl.Message(content="")
    await response_message.send()

    if message.elements:
        image = message.elements[0]
        if image.type == "image":
            with open(image.path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            image_message = {
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
            user_input = image_message
        else:
            await cl.Message(content="Please upload an image file.").send()
            return
    else:
        user_input = message.content

    plan_saved = False
    async for token in planning_agent.generate_response(user_input):
        if isinstance(token, dict) and token.get("function_call"):
            function_call = token["function_call"]
            if isinstance(function_call, dict):
                function_name = function_call.get("name")
                arguments = function_call.get("arguments", "{}")
                try:
                    function_args = json.loads(arguments)
                    if function_name == "updateArtifact":
                        result = update_artifact(
                            function_args.get("filename"), function_args.get("content")
                        )
                        await response_message.stream_token(f"\n{result}\n")
                        plan_saved = True
                except json.JSONDecodeError:
                    await response_message.stream_token(
                        "\nError: Invalid function arguments\n"
                    )
        else:
            await response_message.stream_token(token)

    if not plan_saved:
        await response_message.stream_token(
            "\nPlan generated. To save it, please confirm if the plan is good or provide feedback.\n"
        )

    await response_message.update()


if __name__ == "__main__":
    cl.main()
