from transformers import pipeline, set_seed


def data():
    for i in range(3):
        yield f"My example {i}"


generator = pipeline('text-generation', model='gpt2')


out = generator("Hello,")
print(out)

print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))

text_len =0
for out in generator(data()):
    text_len += len(out[0]['generated_text'])

print(text_len)
