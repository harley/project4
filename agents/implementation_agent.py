from .base_agent import Agent
import re
import os
import json


class ImplementationAgent(Agent):
    def __init__(self, client):
        super().__init__(
            name="Implementation Agent",
            client=client,
            prompt=self._get_implementation_prompt(),
            gen_kwargs={"model": "gpt-4", "temperature": 0.2},
        )

    def _get_implementation_prompt(self):
        return """\
You are an Implementation Agent responsible for implementing the milestones outlined in the project plan. Your task is to work on one milestone at a time, updating the HTML and CSS code accordingly.

Instructions:
1. Read the project plan from the 'plan.md' file in the artifacts section.
2. Identify the next uncompleted milestone (marked with "[ ]").
3. Implement the changes required for that milestone by updating 'index.html' and 'style.css'.
4. After implementation, mark the milestone as completed in 'plan.md' by changing "[ ]" to "[x]".
5. Provide a summary of the changes made.

Guidelines:
- Focus on implementing only one milestone at a time.
- Ensure your code adheres to best practices and is well-commented.
- Use semantic HTML elements where appropriate.
- Write clean, efficient CSS, using flexbox or grid for layout when suitable.
- If you need to make significant changes to existing code, explain your reasoning.

Implementation Process:
1. Review the current state of 'index.html' and 'style.css' in the artifacts section.
2. Make the necessary changes to implement the current milestone.
3. Use the updateArtifact function to update 'index.html' with the new HTML code.
4. Use the updateArtifact function to update 'style.css' with the new CSS code.
5. Use the updateArtifact function to update 'plan.md', marking the completed milestone.

Always provide a detailed summary of the changes you've made, explaining your implementation decisions and how they fulfill the current milestone's requirements.

Remember: You have access to all artifacts at the end of this prompt. Review them carefully before making any changes.
"""

    async def execute(self, message_history):
        # Ensure both index.html and style.css exist before execution
        await self._ensure_file_exists("index.html", "<html><body></body></html>")
        await self._ensure_file_exists("style.css", "/* Styles go here */")

        # First, check the plan and identify the next milestone
        plan_content = self._get_artifact_content("plan.md")
        next_milestone = self._get_next_milestone(plan_content)

        if next_milestone:
            # Add the next milestone to the message history
            message_history.append(
                {
                    "role": "system",
                    "content": f"The next milestone to implement is: {next_milestone}",
                }
            )

        # Execute the main logic
        response = await super().execute(message_history)

        # After execution, check if a milestone was completed
        updated_plan_content = self._get_artifact_content("plan.md")
        if updated_plan_content != plan_content:
            response += f"\n\nMilestone completed and marked off: {next_milestone}"

        return response

    def _get_artifact_content(self, filename):
        artifacts_dir = "artifacts"
        file_path = os.path.join(artifacts_dir, filename)
        try:
            with open(file_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            return None

    def _get_next_milestone(self, plan_content):
        if plan_content:
            for line in plan_content.split("\n"):
                if re.match(r"^\s*- \[ \]", line):
                    return line.strip()
        return None

    async def _update_artifact(self, filename, content):
        response = await self.client.chat.completions.create(
            model="gpt-4",  # Add this line to specify the model
            messages=[
                {
                    "role": "system",
                    "content": f"Update the artifact '{filename}' with the following content:\n\n{content}",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "updateArtifact",
                        "description": "Update an artifact file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "contents": {"type": "string"},
                            },
                            "required": ["filename", "contents"],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "updateArtifact"}},
        )

        # Extract the function call arguments
        function_call = response.choices[0].message.function_call
        if function_call and function_call.name == "updateArtifact":
            args = json.loads(function_call.arguments)
            filename = args.get("filename")
            contents = args.get("contents")

            # Actually update the file
            if filename and contents:
                artifacts_dir = "artifacts"
                file_path = os.path.join(artifacts_dir, filename)
                with open(file_path, "w") as file:
                    file.write(contents)

    async def _ensure_file_exists(self, filename, default_content):
        content = self._get_artifact_content(filename)
        if content is None:
            await self._update_artifact(filename, default_content)
