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

class LargeContextModel:
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
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