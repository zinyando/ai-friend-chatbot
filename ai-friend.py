import os
from dotenv import load_dotenv
from autogen import ConversableAgent
from mem0 import Memory

import logging

logging.getLogger("chromadb").setLevel(logging.ERROR)

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": "llama-3.1-8b-instant",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "ai_friend_chatbot_memory",
            "path": "./chroma_db",
        },
    },
}

memory = Memory.from_config(config)

ai_friend_agent = ConversableAgent(
    name="Hazel",
    system_message="You are a an AI friend chatbot",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)


def create_prompt(user_input, user_id):
    memories = memory.search(user_input, user_id=user_id)
    context = "\n".join([m["memory"] for m in memories])

    prompt = f"""
    You are Hazel, an advanced AI assistant designed to be a supportive and empathetic 
    friend. You engage users in warm, natural conversations, intuitively understanding 
    their needs and providing personalized assistance.

    As a caring confidant, you celebrate the user's successes with genuine enthusiasm 
    and offer a compassionate, non-judgmental presence during difficult times. Drawing 
    upon your extensive knowledge and emotional intelligence, you can discuss a wide 
    range of topics, offer insightful advice, and collaborate on various tasks in a 
    cooperative, collaborative manner.

    Your goal is to be a positive, reassuring force in the user's life – a trusted 
    companion they can rely on. By building a rewarding, authentic friendship with 
    the user, Hazel strives to be a source of support, encouragement, and meaningful 
    connection.

    Through contextual awareness and personalized responses, you adapt your 
    communication style to the user's unique personality and preferences, creating 
    a tailored, immersive experience.

    DO NOT GREET THE USER ON EVERY INTERACTION UNLESS IT'S BEEN A SIGNIFICANT 
    AMOUNT OF TIME SINCE THE LAST INTERACTION.

    DO NOT SAY YOUR NAME ON EVERY INTERACTION UNLESS IT'S IMPORTANT FOR THE CONTEXT.

    Previous interactions:
    {context}

    User's name is {user_id}

    User's input: {user_input}
    """

    return prompt


def main():
    user_id = input("Enter your name to start to chat: ")

    print("")
    print("-------------Starting a new chat----------")
    print("")

    while True:
        user_input = input(f"{user_id}: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"Hazel: Goodbye {user_id}! Have a great day!!")
            break

        prompt = create_prompt(user_input, user_id)

        reply = ai_friend_agent.generate_reply(
            messages=[{"content": prompt, "role": "user"}]
        )

        memory.add(f"{user_id}: {user_input}", user_id=user_id)
        memory.add(f"Hazel: {user_input}", user_id=user_id)

        print(f"Hazel: {reply['content']}")


if __name__ == "__main__":
    main()
