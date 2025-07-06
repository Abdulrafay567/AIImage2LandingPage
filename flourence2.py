import replicate
import requests
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
from gliner import GLiNER
import openai
import base64
from dotenv import load_dotenv

load_dotenv()
# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_KEY")
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise Exception(f"Image not found at path: {image_path}")

# Constants
MODEL_VERSION = "da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595"
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY")
import ast
def captioning_landingPage(image_path):
#   image_url = upload_image_to_imgbb(image_path)
    print("Image URL:", image_path)
    prompt ="""Give me a detailed JSON profile of this landing page. This profile should cover all aspects of the landing page that can be used to generate HTML/CSS/JS code resulting in 100% copy of the landing page image. Be as detailed as possible covering aspects such as color scheme, colour gradients, typography,  background, any images, various sections and UI components and anything that can be used to perfectly describe the landing page COMPONENT BY component, section by section. Also provide an accompanying Summary and detailed caption for the whole landing page aginst the keys 'summary' and 'detailed_caption'. your output must be in a valid JSON format."""
    # with open(image_path, "rb") as image_file:
    # with open(image_path, "rb") as image_file:
        # client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        # output = client.run(
        #         f"lucataco/florence-2-large:{MODEL_VERSION}",
        #         input={
        #             "image": image_file,
        #             "task_input": "More Detailed Caption"
        #         }
        #     )
    encoded_image = encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": """you are a helpful assistant that generates detailed JSON profiles of landing pages from images. Your task is to analyze the image and provide a comprehensive description of the landing page, including its layout, color scheme, typography, and any other relevant details."""
        },       
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt  
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.3,
    #response_format={"type": "json_object"}  # Force JSON response format
)
    # print(response)
    # text_dict = ast.literal_eval(output['text'])
    # more_detailed_caption = text_dict.get('<MORE_DETAILED_CAPTION>')
    openai_response = response.choices[0].message.content
    return openai_response

def gpt_image(image_path):
    """
    Uses OpenAI's GPT-4o model to generate a detailed JSON profile of the landing page from the image.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: JSON profile of the landing page.
    """
    encoded_image = encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": """you are a helpful assistant that generates detailed JSON prompts of sub-images present in the landing pages from images. Your task is to analyze the image and provide a detailed image prompt, size(in px), location and caption for each sub-image present in the landing page image."""
        },       
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "extract all subimages ,excluding sections and buttons, present in the landing page image and provide a detailed image prompt and caption for each sub-image present in the landing page image including logos and icons. The output should be in the format of a JSON object with keys 'prompt', 'size', 'location' and 'caption'."  
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.3,
    #response_format={"type": "json_object"}  # Force JSON response format
)
    return response.choices[0].message.content

from gliner.config import GLiNERConfig
def extract_objects_with_gliner(description):
    model = GLiNER.from_pretrained("urchade/gliner_largev2")
    
    # Define entity labels we are interested in
    labels = ["WEBSITE_NAME", "COLOR_SCHEME", "MAIN_IMAGE", "SUB_IMAGE","photo", "LOGO", "SECTION"]
    
    # Get entity predictions
    entities = model.predict_entities(description, labels)

    # Initialize object dictionary
    objects = {
        "website_name": None,
        "color_scheme": [],
        "main_image": None,
        "sub_images": [],
        "logo": None,
        "sections": [],
        "photo":[]
    }

    # Assign extracted entities to relevant keys
    for entity in entities:
        label = entity["label"]
        text = entity["text"]

        if label == "WEBSITE_NAME":
            objects["website_name"] = text
        elif label == "COLOR_SCHEME":
            objects["color_scheme"].append(text)
        elif label == "MAIN_IMAGE":
            objects["main_image"] = text
        elif label == "SUB_IMAGE":
            objects["sub_images"].append(text)
        elif label == "LOGO":
            objects["logo"] = text
        elif label == "SECTION":
            objects["sections"].append(text)
        elif label == "photo":
            objects["photo"].append(text)    

    # Remove duplicates
    for key in ["color_scheme", "sections", "sub_images"]:
        objects[key] = list(set(objects[key]))

    return objects



 
def get_bounding_boxes(image_url, text_input):
    """
    Uses the Replicate API to get bounding boxes for the given image and text input.
    
    Args:
        image_url (str): URL of the image.
        text_input (str): Text input for the model.
    
    Returns:
        dict: Output from the Replicate API.
    """
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    with open(image_url, "rb") as image_file:
        output = client.run(
            f"lucataco/florence-2-large:{MODEL_VERSION}",
            input={
                "image": image_file,
                "task_input": "Caption to Phrase Grounding",
                "text_input": f"{text_input}"
            }
        )
    return output

def plot_bounding_boxes(image, bbox_results):
    """
    Plots the bounding boxes on the given image.
    
    Args:
        image (PIL.Image): Image to plot on.
        bbox_results (list): List of bounding box results.
    """
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create a figure with the same aspect ratio as the image
    dpi = 100  # Adjust DPI for better display
    fig, ax = plt.subplots(1, figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Display the image
    ax.imshow(image)
    
    # Plot each bounding box
    for result in bbox_results:
        bbox = result['bbox']
        label = result['label']
        
        # Draw the bounding box
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label text
        ax.text(
            bbox[0], 
            bbox[1] - 5, 
            label, 
            color='r', 
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    # Remove extra whitespace and axis
    plt.tight_layout()
    plt.axis('off')
    # plt.show()

def crop_image(image, bboxes, labels):
    """
    Crops the image using the given bounding boxes and labels.
    
    Args:
        image (PIL.Image): Image to crop.
        bboxes (list): List of bounding boxes.
        labels (list): List of labels.
    
    Returns:
        list: List of cropped images with their corresponding labels.
    """
    cropped_images = []

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = bbox  # Extract coordinates
        
        # Crop using PIL (if 'image' is a PIL Image)
        cropped = image.crop((x1, y1, x2, y2))  # PIL format: (left, top, right, bottom)
        
        # Save the cropped image
        cropped.save(f'cropped_image_{i}.jpg')
        
        # Append the cropped image to the list
        cropped_images.append({
            'image': cropped,
            'label': label
        })
    
    return cropped_images

