from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from lmdeploy.vl import load_image

def generate_gpt4o_image_recognition(image_path):
    OPENAI_API_KEY = "sk-jz0shLgMJY9HBVnLC3Fe3dCaA5204a418e67003f637f1eFf"
    MODEL_NAME = 'gpt-4o'

    # Initialize the Langchain model
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

    # Load the local image
    image = load_image(image_path)

    # Create a message that includes the image
    human_message = HumanMessage(content=[
        { "type": "text", "text": "Please describe the content of this image." },
        { "type": "image_url", "image_url": { "url": image_path }}
    ])

    # Call the GPT-4o model with the image
    response = llm.invoke([human_message])

    return response.content

# Example usage
# result = generate_gpt4o_image_recognition("/path/to/your/image.png")
# print(result)
