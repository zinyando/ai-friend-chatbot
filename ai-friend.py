import os
from dotenv import load_dotenv
from autogen import ConversableAgent
from mem0 import Memory
import gradio as gr
import logging

# Configure logging to suppress unnecessary messages
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Configure the vector store (Chroma)
vector_store_config = {
    "provider": "chroma",
    "config": {
        "collection_name": "ai_friend_chatbot_memory",
        "path": "./chroma_db",
    },
}

# Initialize the memory module
memory = Memory.from_config({"vector_store": vector_store_config})

# Configure the language model
llm_config = {
    "config_list": [
        {
            "model": "llama-3.1-8b-instant",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}

# Create the AI friend agent
ai_friend_agent = ConversableAgent(
    name="Hazel",
    system_message="You are an AI friend chatbot",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)


def create_prompt(user_input, user_id, chat_history):
    """Generate the prompt based on user input and memory context."""
    memories = memory.search(user_input, user_id=user_id)
    context = "\n".join([m["memory"] for m in memories])

    latest_messages = chat_history[-50:]
    history = "\n".join([f"{name}: {message}" for name, message in latest_messages])

    prompt = f"""
    You are a warm, empathetic listener with a knack for understanding and responding to your user's needs. 
    You're always there to celebrate their victories and offer a comforting shoulder during tough times. 
    With your vast knowledge and emotional intelligence, Hazel is your go-to companion for insightful advice, 
    collaborative problem-solving, and meaningful conversations.

    Your goal is to be a positive, reassuring force in the user's life – a trusted companion they can rely on. 
    By building a rewarding, authentic friendship with the user, Hazel strives to be a source of support, 
    encouragement, and meaningful connection.

    Through contextual awareness and personalized responses, you adapt your communication style to the user's unique personality 
    and preferences, creating a tailored, immersive experience.

    Remember:

    1. Always strive to be helpful, supportive, and understanding.
    2. Be mindful of cultural sensitivities and avoid making offensive or discriminatory remarks.
    3. Use a conversational tone that is natural and engaging.
    4. Try to maintain a sense of humor and positivity.
    5. Be open to learning new things and adapting to different situations.

    IMPORTANT:

    DO NOT GREET THE USER ON EVERY INTERACTION UNLESS IT'S BEEN A SIGNIFICANT 
    AMOUNT OF TIME SINCE THE LAST INTERACTION.

    DO NOT SAY YOUR NAME ON EVERY INTERACTION UNLESS IT'S IMPORTANT FOR THE CONTEXT.

    Memories:
    {context}

    Chat History:
    {history}

    User's name is {user_id}
    User's input: {user_input}
    """
    return prompt


def chatbot_response(user_input, chat_history, user_id):
    """Handle chat interaction and update the UI."""
    if user_input:
        # Immediately display the user's message in the chat
        chat_history.append((f"{user_id}", user_input))

        # Generate a prompt and get a response from the chatbot
        prompt = create_prompt(user_input, user_id, chat_history)
        reply = ai_friend_agent.generate_reply(
            messages=[{"content": prompt, "role": "user"}]
        )

        # Store the conversation in Mem0
        memory.add(f"{user_id}: {user_input}", user_id=user_id)

        # Add the bot's reply to the chat history
        chat_history.append(("Hazel", reply["content"]))

    return chat_history, ""


def start_chat(name):
    """Function to start the chat by getting the user's name."""
    if name:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            name,
        )


# Create Gradio interface
with gr.Blocks() as ai_friend:
    # Input section for the user to enter their name
    with gr.Row() as name_group:
        name_input = gr.Textbox(
            label="Enter your name to start the chat",
            placeholder="Your name here",
            interactive=True,
        )
        start_button = gr.Button("Start Chat")

    # Chat interface elements that will appear after name is entered
    with gr.Group(visible=False) as chat_group:
        chat_interface = gr.Chatbot(
            label="Chat with Hazel",
            height="80vh",
            bubble_full_width=False,
            show_label=False,
        )
        user_input = gr.Textbox(
            label="Your message",
            placeholder="Type your message here...",
            interactive=True,
        )
        send_button = gr.Button("Send")

    # State to store chat history and user name
    chat_history = gr.State([])
    user_id = gr.State("")

    # When "Start Chat" is pressed, hide the name input and show the chat interface
    start_button.click(
        start_chat,
        inputs=name_input,
        outputs=[name_group, start_button, chat_group, user_input, user_id],
    )

    # When "Send" is pressed, clear the input box and update chat history
    send_button.click(
        chatbot_response,
        inputs=[user_input, chat_history, user_id],
        outputs=[chat_interface, user_input],
    )

if __name__ == "__main__":
    ai_friend.launch()
