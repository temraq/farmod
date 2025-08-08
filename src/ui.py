import gradio as gr
import requests

API_URL = "http://localhost:8000/generate"

def generate_response(prompt, use_rag):
    payload = {
        "prompt": prompt,
        "use_rag": use_rag
    }
    try:
        response = requests.post(API_URL, json=payload)
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="PubMed QA Assistant") as demo:
    gr.Markdown("## 🧬 PubMed QA Assistant (Llama-2 + RAG)")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Medical Question", placeholder="Enter your medical question here...")
            rag_toggle = gr.Checkbox(label="Use RAG", value=True)
            submit_btn = gr.Button("Generate Answer")
        with gr.Column():
            output_text = gr.Textbox(label="Answer", interactive=False)
    
    submit_btn.click(
        fn=generate_response,
        inputs=[input_text, rag_toggle],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)