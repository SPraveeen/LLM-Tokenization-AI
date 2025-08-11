import transformers
from transformers import GPT2Tokenizer

# load the pre-trained GPT-2 tokenizer
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")

# input text
text=input("Enter your text")

# tokenize the input
tokens=tokenizer.tokenize(text)
token_ids=tokenizer.convert_tokens_to_ids(tokens)

# print results
print("original Text:", text)
print("tokens", tokens)
print("token_ids:", token_ids)

# optional: Encode directly(tokens + ids)
encoded=tokenizer(text)
print("Encoded Output:", encoded)

# decode back from IDs to String
decoded_text=tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)