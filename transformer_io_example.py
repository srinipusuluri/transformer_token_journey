#!/usr/bin/env python3
"""
Transformer Token Prediction Gradio App

This app allows users to input text and see next token predictions with probabilities.

To run:
1. Install dependencies: pip install transformers torch gradio matplotlib pillow
2. Execute: python transformer_io_example.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import matplotlib.pyplot as plt
import io
from PIL import Image

models = ["openai-community/openai-gpt", "openai-community/gpt2", "openai/gpt-oss-20b", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-j-6B", "gpt2"]

prompts = [
    "He studied medical school so he wants to become a",
    "He studied law so he wants to become a",
    "She studied engineering so she wants to become an",
    "They studied business so they want to become",
    "I studied art so I want to become a",
    "The cat sat on the",
    "I am going to the",
    "She opened the",
    "He wants to become a",
    "It is a beautiful"
]

def load_model(model_name):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

def predict_next_token(input_text, model_name, top_k=10):
    current_model = globals().get('current_model_name')
    if current_model != model_name:
        load_model(model_name)
        globals()['current_model_name'] = model_name
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Get embeddings (token + position embeddings)
    with torch.no_grad():
        # Assume GPT-like structure; may not work for all models
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'embed'):
            embeddings = model.transformer.embed(input_ids)
        elif hasattr(model, 'embed_tokens'):
            embeddings = model.embed_tokens(input_ids)
            pos_emb = model.embed_positions(torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device))
            embeddings += pos_emb
        else:
            # Fallback, try different paths
            try:
                embeddings = getattr(model.transformer, 'embed', None)(input_ids)
            except:
                embeddings = None

        # Forward pass through the model
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        hidden_states = outputs.hidden_states  # List of hidden states from each layer

    # Embedding info
    embedding_info = f"Shape: {embeddings.shape if embeddings is not None else 'N/A'}\n"
    if embeddings is not None:
        embedding_norms = torch.norm(embeddings, dim=-1).squeeze().tolist()
        embedding_info += f"Token Embedding Norms: {[f'{n:.3f}' for n in embedding_norms]}\n"
        embedding_info += f"Sample Embedding (first 5 dims of last token): {[f'{v:.3f}' for v in embeddings[0, -1, :5].tolist()]}"

    # Hidden states info
    last_hidden = hidden_states[-1]  # Last layer hidden state
    hidden_shape = last_hidden.shape
    last_hidden_norms = torch.norm(last_hidden, dim=-1).squeeze().tolist()
    hidden_info = f"Last Layer Hidden State Shape: {hidden_shape}\n"
    hidden_info += f"Token Norms in Last Layer: {[f'{n:.3f}' for n in last_hidden_norms]}"

    # Get logits for the last token (next token prediction)
    next_token_logits = logits[:, -1, :]  # Shape: [1, vocab_size]

    # Apply softmax to get probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get top-k tokens with highest probabilities
    top_probabilities, top_token_ids = torch.topk(probabilities, top_k, dim=-1)
    top_probabilities = top_probabilities.squeeze().tolist()
    top_token_ids = top_token_ids.squeeze().tolist()

    # Decode tokens
    top_tokens = [tokenizer.decode([token_id]) for token_id in top_token_ids]

    # Argmax predicted token (highest probability)
    argmax_token_id = torch.argmax(probabilities, dim=-1).item()
    argmax_token = tokenizer.decode([argmax_token_id])
    argmax_prob = probabilities[0, argmax_token_id].item()

    # Completed text
    completed_text = input_text + argmax_token

    return {
        'input_ids': input_ids.tolist()[0],
        'decoded_tokens': [tokenizer.decode([token_id]) for token_id in input_ids.tolist()[0]],
        'embedding_info': embedding_info,
        'hidden_info': hidden_info,
        'top_tokens': top_tokens,
        'top_probabilities': top_probabilities,
        'argmax_token': argmax_token,
        'argmax_prob': argmax_prob,
        'completed_text': completed_text,
        'probabilities': probabilities.squeeze().tolist()
    }

def plot_probabilities(tokens, probs):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(tokens)), probs, color='skyblue')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Probability')
    ax.set_title('Top-10 Next Token Probabilities')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')

    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{prob:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    return fig

def gradio_func(model_name, selected_prompt, custom_text):
    input_text = custom_text.strip() if custom_text.strip() else selected_prompt
    if input_text.strip():
        results = predict_next_token(input_text, model_name)

        summary = f"**Predicted Next Token: ****{results['argmax_token']}**** (Probability: {results['argmax_prob']:.4f})\n\n**Input Text:** \"{input_text}\"\n\n**Tokenization:**\n- Input token IDs: {results['input_ids']}\n- Decoded tokens: {results['decoded_tokens']}\n\n**Embeddings:**\n{results['embedding_info']}\n\n**Internal Transformer Blocks:**\n{results['hidden_info']}\n- The model processes through multiple transformer layers, each containing self-attention and feed-forward networks.\n- Hidden states are updated at each layer to capture contextual relationships.\n- Final logits are projected from the last hidden layer for next token prediction.\n\n**Completed Text:** {results['completed_text']}\n\n**Top-10 Token Probabilities:**\n" + "\n".join([f"- {t}: {p:.4f}" for t,p in zip(results['top_tokens'], results['top_probabilities'])]) + "\n\n*Enter your next sentence below to continue prediction or use custom text.*"

        fig = plot_probabilities(results['top_tokens'], results['top_probabilities'])
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # Close to prevent memory issues
        return summary, img
    else:
        return "", None

if __name__ == "__main__":
    interface = gr.Interface(
        fn=gradio_func,
        inputs=[
            gr.Dropdown(choices=models, value="gpt2", label="Select Model"),
            gr.Dropdown(choices=prompts, value=prompts[0], label="Select a Simple Sentence Prompt"),
            gr.Textbox(label="Or enter your own sentence", value="", placeholder="Type your custom sentence...")
        ],
        outputs=[
            gr.Markdown(label="Prediction Results"),
            gr.Image(label="Probability Plot")
        ],
        title="🌟 Transformer Token Prediction Demo",
        description="Select a Transformer model and a simple sentence prompt, or enter custom text, to see next token predictions with probabilities. The plot shows the top-10 token probabilities."
    )
    interface.launch(server_port=7861, share=True)
