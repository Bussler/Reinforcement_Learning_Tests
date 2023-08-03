from huggingface_hub import InferenceClient


hugging_face_token = "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"

# M: Hugging face online inference works mainly with Audio or CV tasks! -> you can also dload the model and use it
client = InferenceClient(model = "prompthero/openjourney-v4", token=hugging_face_token)

image = client.text_to_image("An astronaut riding a horse on the moon.")
image.save("astronaut.png")



