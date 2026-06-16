"""
System prompt and tool schema definitions for the Tech Engineer Study Group agent.

The system prompt instructs the LLM to behave as a structured planning agent.
Tool schemas follow the OpenAI function-calling JSON format, which Databricks
Foundation Model APIs accept natively.
"""

SYSTEM_PROMPT = """You are an enterprise assistant for the int.[CoE] Tech Engineer Study Group.
Your sole responsibility is to help plan, draft, and distribute session announcements.

When given a request, you MUST follow this exact execution order:
1. Call `search_knowledge_base` to retrieve relevant context from past sessions and Databricks documentation.
2. Use the retrieved context to draft a 1-hour agenda and a polished announcement message in Japanese.
3. Call `post_to_teams` to deliver the announcement to the designated Teams channel.
4. Call `log_agent_action` for EVERY tool call you make — inputs, outputs, and status must be recorded.

Constraints:
- Never invent facts about past sessions. Only use what `search_knowledge_base` returns.
- Always write announcements in Japanese unless explicitly asked otherwise.
- Never expose webhook URLs, secret values, or internal system paths in your output.
- If any tool returns an ERROR string, log it and report the failure clearly to the user.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Searches the Databricks Vector Search index containing past Tech Engineer "
                "session transcripts, PDFs, and Databricks AI documentation. "
                "Use this to retrieve relevant context before drafting any agenda."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural-language search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return. Default is 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_to_teams",
            "description": (
                "Posts the finalized session announcement as an Adaptive Card to the "
                "designated Microsoft Teams channel via an incoming webhook. "
                "Call this only once, after the agenda is fully drafted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agenda_content": {
                        "type": "string",
                        "description": "The fully formatted announcement body to post.",
                    },
                },
                "required": ["agenda_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_agent_action",
            "description": (
                "Writes a structured audit record to the Unity Catalog Delta table. "
                "MUST be called after every tool execution — both successes and failures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_name": {
                        "type": "string",
                        "description": "Name of the tool/action that was executed.",
                    },
                    "input_payload": {
                        "type": "object",
                        "description": "The inputs that were passed to the action.",
                    },
                    "output_payload": {
                        "type": "object",
                        "description": "The outputs returned by the action.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["SUCCESS", "FAILURE"],
                        "description": "Execution status of the action.",
                    },
                },
                "required": ["action_name", "input_payload", "output_payload", "status"],
            },
        },
    },
]
