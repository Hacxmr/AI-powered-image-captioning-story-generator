from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load the image
image_url = "nature_image.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate the caption
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
