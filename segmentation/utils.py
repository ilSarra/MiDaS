from torchvision import transforms
from PIL import Image

def preprocess_input(input, input_size):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.fromarray(input)
    image = image.resize(input_size, resample=Image.Resampling.NEAREST)
    image = preprocess(image)
    return image
