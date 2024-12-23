import gradio as gr
import requests
import json

# Hardcoded dataset of predefined responses
PREDEFINED_RESPONSES = {
    "What does the eligibility verification agent (EVA) do?": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections.",
    "What does the claims processing agent (CAM) do?" : "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements.",
    "How does the payment posting agent (PHIL) work?": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden.",
    "Tell me about Thoughtful AI's Agents.": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others.",
    "What are the benefits of using Thoughtful AI's agents?": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting.",
    "What is Thoughtful AI?": "Thoughtful AI is a platform designed to provide tailored AI solutions for businesses, focusing on user-centric experiences and ethical AI practices.",
    "How can I get started with Thoughtful AI?": "You can get started by signing up on our platform and exploring our resources, including tutorials, API documentation, and customer support.",
    "What industries does Thoughtful AI serve?": "Thoughtful AI serves a wide range of industries including healthcare, finance, retail, and education, helping businesses streamline their operations and improve customer experiences.",
    "How does Thoughtful AI ensure data privacy?": "We prioritize data privacy by implementing strict security protocols and complying with industry standards like GDPR and CCPA."
}

# GPT-2 inference server details (use the correct endpoint)
GPT2_SERVER_URL = "http://127.0.0.1:8080/"  # Adjust to match your local server or API endpoint

# Function to generate a response using the GPT-2 server
def gpt2_fallback(question: str) -> str:
    try:
        payload = {
            "inputs": f"You are a customer support AI Agent for Thoughtful AI. Answer the following question: {question}",
            "parameters": {
                "max_length": 256,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50
            }
        }
        response = requests.post(GPT2_SERVER_URL, json=payload, timeout=10)
        response.raise_for_status()  # Raises an exception for HTTP errors
        return response.json().get("generated_text", "").strip()
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while generating a response: {str(e)}"

# Function to get the most relevant response
def get_response(question: str) -> str:
    # Check for predefined responses
    if question in PREDEFINED_RESPONSES:
        return PREDEFINED_RESPONSES[question]
    else:
        # Fallback to GPT-2 for other questions
        return gpt2_fallback(question)

# Gradio interface
def chatbot(question):
    answer = get_response(question)
    return answer

with gr.Blocks() as demo:
    gr.Markdown("# Thoughtful AI Customer Support Agent")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Ask a question about Thoughtful AI")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Answer")

    submit_btn.click(chatbot, inputs=[user_input], outputs=[output])

# Launch the app
demo.launch()
