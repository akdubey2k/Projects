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

    # initializes the Hugging Face model and tokenizer using the specified model_name and sets up a 
    # pipeline for text generation. It configures the pipeline with specific settings to ensure 
    # proper behavior.
    def load_model(self):

        # uppresses warnings of the FutureWarning category, which are often issued by the 
        # transformers library to indicate upcoming deprecations or changes in future versions.
        warnings.filterwarnings("ignore", category=FutureWarning)

        # AutoTokenizer class loads the appropriate tokenizer for the model specified by 
        # self.model_name (e.g., "distilgpt2") from Hugging Face’s model hub.The tokenizer converts 
        # text into numerical tokens (and vice versa) for model input and output.
        # 
        # AutoTokenizer is model-agnostic, automatically selecting the correct tokenizer based on 
        # the model’s configuration. For DistilGPT-2, it loads a tokenizer compatible with GPT-2’s 
        # architecture, handling vocabulary and special tokens (e.g., <|endoftext|>) correctly.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # The AutoModelForCausalLM class loads a pre-trained causal language model (e.g., 
        # DistilGPT-2) from Hugging Face’s model hub, specified by self.model_name. Causal language 
        # models are designed for text generation, predicting the next token in a sequence.
        # 
        # This class is model-agnostic, automatically selecting the correct model architecture for 
        # causal language modeling (e.g., GPT-2, LLaMA) based on the model ID. For "distilgpt2", 
        # it loads a distilled version of GPT-2 optimized for text generation.
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Set pad_token_id to eos_token_id if not already set
        # Ensures the tokenizer has a valid pad_token_id for tasks requiring padding (e.g., batch 
        # processing or fixed-length inputs). Some models, like GPT-2, don’t define a padding token 
        # by default, as they are designed for causal language modeling without padding.
        # 
        # If pad_token_id is None, the pipeline may raise errors or produce incorrect results when 
        # handling padded inputs. Setting it to eos_token_id (e.g., <|endoftext|> for GPT-2) is a 
        # common workaround, as the EOS token is a natural choice for padding in causal models.
        if tokenizer.pad_token_id is None:

            # The EOS token marks the end of a sequence, making it a reasonable substitute for 
            # padding, as it doesn’t introduce semantically incorrect tokens. This ensures 
            # compatibility with the pipeline’s padding requirements.
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Initializes a HuggingFace pipeline for text generation, combining the model and tokenizer
        # with settings optimized for generating coherent text. The pipeline is stored in 
        # self.pipeline for use in the generate method, avoiding the need to recreate it for each 
        # generation task.
        self.pipeline = pipeline(

            # Specifies the task as text generation, configuring the pipeline to use the model for 
            # predicting the next tokens in a sequence.
            "text-generation",

            # Passes the loaded AutoModelForCausalLM instance to the pipeline, defining the model 
            # used for generation.
            model=model,

            # Passes the loaded AutoTokenizer instance to handle text preprocessing and decoding.
            tokenizer=tokenizer,

            # Limits the generated output to 200 new tokens controlling the length of the response.
            # Why Used: Prevents overly long outputs, which could be computationally expensive or 
            # irrelevant. 200 tokens is a reasonable default for short, focused responses (e.g., a 
            # few sentences), suitable for agent or RAG tasks.
            max_new_tokens=200,  # Generate up to 200 new tokens

            # Enables truncation of input prompts to fit the model’s maximum context length (e.g., 
            # 1024 tokens for DistilGPT-2).
            truncation=True,

            # Removes extra spaces introduced during tokenization when decoding the output, 
            # improving readability. Thus, enhances the quality of generated text by ensuring 
            # clean, human-readable output without tokenization artifacts (e.g., extra spaces 
            # around punctuation).
            clean_up_tokenization_spaces=True,

            # Forces the model to run on the CPU by setting device=-1.This is particularly relevant
            # for lightweight models like DistilGPT-2, which can run efficiently on CPUs.
            device=-1,  # Force CPU to avoid MPS issues

            # Sets the pipeline’s padding token ID to the tokenizer’s EOS token ID, ensuring 
            # consistent padding behavior.
            pad_token_id=tokenizer.eos_token_id
        )

    # Interface to generate text based on a user-provided prompt, leveraging the loaded pipeline.
    # The prompt is a string input (e.g., question or instruction) that the model uses to generate
    # a response. It’s the primary input for text generation tasks.
    def generate(self, prompt):

        # Ensures that the pipeline has been created (via load_model) before attempting to generate
        # text, preventing runtime errors due to an uninitialized pipeline.
        if self.pipeline is None:

            # Raising an exception provides clear feedback to the user that the model must be 
            # loaded, improving error handling and debugging.
            raise ValueError("Model not loaded.")
        
        # Invokes the pipeline to generate text based on the prompt, with return_full_text=False to 
        # exclude the input prompt from the output, and extracts the generated text from the 
        # pipeline’s result.
        # prompt to generate text, leveraging the pre-configured model and tokenizer.
        # return_full_text = False: Ensures only the generated text (not the input prompt) is 
        # returned, which is useful for applications where only the response is needed (e.g., 
        # answering a question without repeating it).
        # [0]["generated_text"]: The pipeline returns a list of dictionaries, where each dictionary 
        # contains a "generated_text" key with the generated output. [0] selects the first (and 
        # typically only) result, and ["generated_text"] extracts the text string.
        return self.pipeline(prompt, return_full_text=False)[0]["generated_text"]