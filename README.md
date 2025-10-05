# Visualizing Transformers: A Token Journey

This project combines an interactive Gradio application with an educational HTML page to demystify how Transformer models, like GPT variants, process tokens and make predictions. It is based on the foundational Transformer architecture introduced in the paper "Attention is All You Need" by Vaswani et al. ([arXiv:1706.03762](https://arxiv.org/pdf/1706.03762)). The project is designed as an educational journey through the inner workings of Transformers, illustrating concepts through code, visualizations, and step-by-step explanations.

## What Does It Do?

The project consists of three main components:

- **`transformer_io_example.py`**: A Gradio-based web app that allows users to input text (from predefined prompts or custom sentences) and see next-token predictions. It visualizes probabilities for the top-k tokens, displays embedding and hidden state information, and shows the completed text with the predicted token. Supported models include various GPT variants (e.g., GPT-2, GPT-Neo, GPT-J).

- When you run this code here is what you see

- <img width="1039" height="703" alt="image" src="https://github.com/user-attachments/assets/6bd0ef48-b6f2-47f3-b0cd-3d860b733a99" />


- **`transformer_explanation.html`**: A detailed educational webpage explaining Transformers from the ground up. It covers tokenization, embeddings, positional encoding, multi-head self-attention, feed-forward networks, the prediction process, and training dynamics. Includes code snippets, diagrams, and a concrete example of predicting "mat" after "The cat sat on the".
- https://srinipusuluri.github.io/transformer_basics/

- **`token_probabilities.png`**: A sample output visualization showing top token probabilities plotted as a bar chart.

This setup serves as a "token journey" by guiding users from high-level concepts (HTML explanation) to hands-on experimentation (Gradio app), helping to understand why Transformers predict the way they do.

## Features

- **Model Support**: Integrates with open-source Transformer models from Hugging Face, such as `openai-community/openai-gpt`, `gpt2`, `EleutherAI/gpt-neo-1.3B`, and `EleutherAI/gpt-j-6B`.
- **Visualizations**: Real-time plots of token probabilities, embedding norms, and hidden state details.
- **Predefined Prompts**: Example sentences like "The cat sat on the" to demonstrate predictions.
- **Custom Input**: Users can enter their own text for prediction.
- **Educational Context**: Links theory to practice, explaining loss functions, gradients, and weight updates alongside code.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd viz_transformer_token_journey
   ```

2. Install dependencies:
   ```bash
   pip install transformers torch gradio matplotlib pillow
   ```

   - `transformers`: For loading pre-trained models and tokenizers.
   - `torch`: PyTorch backend for model computations.
   - `gradio`: Web UI framework for the app.
   - `matplotlib`: Plotting utilities.
   - `pillow`: Image handling for visualizations.

## Usage

### Running the App
Execute the Gradio app to start the interactive demo:
```bash
python transformer_io_example.py
```
This launches a local web server (default port 7861). Open the URL in a browser to use the interface. Select a model, choose a prompt or enter custom text, and observe predictions with visualizations.

### Setting HF_TOKEN
Some models (e.g., larger variants) may require a Hugging Face access token for download or API access:
1. Obtain a token from [Hugging Face](https://huggingface.co/settings/tokens).
2. Set the environment variable:
   ```bash
   export HF_TOKEN=your_hugging_face_token_here
   ```
   Add this to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) for persistence.

### Educational Resources
- Open `transformer_explanation.html` in a web browser to dive deep into Transformer mechanics. It complements the app by explaining concepts like self-attention, softmax probabilities, and cross-entropy loss with real examples.
- View `token_probabilities.png` for a sample output image.

## Why "Token Journey"?

This project is titled "Vizualizing Transformers: A Token Journey" because it maps out the complete path of a token—from initial embedding to final prediction—using both analytical explanations (HTML) and visual tools (app). It aims to make the abstract "journey" through Transformer layers tangible, answering questions like: How does "The cat sat on the" know to predict "mat"? Users can bridge theory and practice, experimenting with models to see predictions evolve.

## Thanks to Vizuara

https://www.youtube.com/watch?v=IWPmmwvuu9w&t=2370s
## Contributing

Feel free to extend the app with more models, additional visualizations, or interactive elements in the HTML page. Pull requests welcome!

## License

This project uses open-source libraries and models under their respective licenses (MIT for `gr` related, Apache 2.0 for `transformers`, etc.).
