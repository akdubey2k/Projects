# The warnings module allows control over warning messages generated during program execution, such
# as suppressing, filtering, or redirecting warnings to logs. It is used to manage non-critical 
# issues (e.g., deprecation warnings) that might clutter output or distract developers.
import warnings


# The pipeline class provides a high-level interface for common natural language processing (NLP) 
# tasks, such as text generation, question-answering, or sentiment analysis. It abstracts model 
# loading, tokenization, and inference into a single, user-friendly API.
# 
# The AutoTokenizer class automatically loads the appropriate tokenizer for a given pre-trained 
# model from Hugging Face’s model hub. A tokenizer converts text into numerical tokens (and vice 
# versa) that the model can process.
# 
# The AutoModelForCausalLM class automatically loads a pre-trained causal language model (e.g., 
# GPT-2, LLaMA, Mistral) from Hugging Face’s model hub. Causal language models are designed for 
# text generation, predicting the next token in a sequence.
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# The LargeContextModel class encapsulates the loading and usage of a causal language model (e.g.,
# DistilGPT-2) for generating text based on prompts. It provides a clean interface for model 
# initialization, loading, and text generation, making it reusable in applications like the 
# AgenticAI class or a RAG system.
class LargeContextModel:

    # The constructor initializes the class by storing the model name and setting the pipeline 
    # attribute to None, deferring model loading until explicitly requested.
    def __init__(self, model_name="distilgpt2"):

        # The model_name specifies the Hugging Face model to use (e.g., "distilgpt2", "gpt2", or 
        # "mistralai/Mixtral-8x7B-v0.1"). It allows flexibility to choose different models without 
        # changing the class’s code. Default to "distilgpt2": is a lightweight, distilled version 
        # of GPT-2 with 82 million parameters, offering a balance of performance and resource 
        # efficiency. It’s suitable for development, testing, or resource-constrained environments 
        # (e.g., CPU-only setups), making it a reasonable default for prototyping or small-scale 
        # applications.
        self.model_name = model_name

        # self.pipeline = None: Initializing pipeline as None defers the resource-intensive process 
        # of loading the model until the load_model method is called. This lazy initialization 
        # avoids unnecessary memory or CPU usage during class instantiation.
        self.pipeline = None

    def load_model(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # Set pad_token_id to eos_token_id if not already set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,  # Generate up to 200 new tokens
            truncation=True,
            clean_up_tokenization_spaces=True,
            device=-1,  # Force CPU to avoid MPS issues
            pad_token_id=tokenizer.eos_token_id
        )

    def generate(self, prompt):
        if self.pipeline is None:
            raise ValueError("Model not loaded.")
        return self.pipeline(prompt, return_full_text=False)[0]["generated_text"]