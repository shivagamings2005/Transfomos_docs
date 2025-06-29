from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Load the content of the text file
file_path = r"D:\sih\final_selected\others\context.txt"  # Replace with your file path
with open(file_path, "r") as file:
    file_content = file.read()

# Initialize an empty conversation history
conversation_history = []

# Add file content as initial context
conversation_history.append({"role": "system", "content": f"Answer questions based only on the following text:\n\n{file_content}"})

def get_response(user_input):
    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Generate the response using the conversation history
    completion = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=conversation_history,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )
    
    # Collect and print the response in chunks for streaming effect
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="")
    
    # Add the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response_text})

# Main chatbot loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    get_response(user_input)
