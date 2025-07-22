# Utilities for converting between message formats and prompts.

def messages_to_prompt(messages):
    """Convert structured messages to the prompt format expected by Saiga."""
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<s>system\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<s>user\n{message.content}</s>\n"
        elif message.role == "bot":
            prompt += f"<s>bot\n{message.content}</s>\n"
    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt
    prompt += "<s>bot\n"
    return prompt


def completion_to_prompt(completion: str) -> str:
    """Format a single completion string to the model prompt."""
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"
