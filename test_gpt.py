from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id  # Ensure padding works
    return model

def load_tokenizer(tokenizer_path):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    except:
        print("Tokenizer not found in the given path. Using pre-trained 'gpt2' tokenizer.")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer

def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)

    ids = tokenizer.encode(sequence, return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

# Set the correct model path
model1_path = r"/mnt/c/Users/woebe/College/SEM 4/AGAIN/results"
sequence1 = "What is nomenclature?  "
max_len = 256

generate_text(model1_path, sequence1, max_len)
