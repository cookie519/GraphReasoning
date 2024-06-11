from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "/scratch/gpfs/jx0800/zephyr-7b-beta"

mytokenizer = AutoTokenizer.from_pretrained(model_dir)
mymodel = AutoModelForCausalLM.from_pretrained(model_dir)

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model=mymodel, tokenizer=mytokenizer)
result = pipe(messages)
print(result)
