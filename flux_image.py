import os
import replicate
import requests
from dotenv import load_dotenv
from generate_prompt import enhance_prompt, DesignPrompt

# Load environment variables
load_dotenv()
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

if not REPLICATE_API_KEY:
    raise ValueError("Replicate API key not found. Please set it in the .env file.")

# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_KEY)

def generate_image(prompt: str, website_name: str, website_type: str, design_philosophy: str, color_scheme: str, image_dir: str) -> list:
    """
    Enhances the prompt using OpenAI and generates 4 images using Replicate AI.
    Returns a list of file paths for the generated images.
    """
    try:
        # Enhance the prompt
        design_prompt = DesignPrompt(
            prompt=prompt,
            website_name=website_name,
            website_type=website_type,
            design_philosophy=design_philosophy,
            color_scheme=color_scheme,
        )
        enhanced_prompt = enhance_prompt(design_prompt)

        
        output = replicate_client.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": enhanced_prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": 4,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "num_inference_steps": 4
        }
        )
                
            # Download and save the image
        file_paths = []
        #path = "images/users"
        for idx, img_url in enumerate(output):
            file_path = os.path.join(image_dir, f"output_{idx}.webp")
            response = requests.get(img_url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                file_paths.append(file_path)
            else:
                print(f"Failed to download image {idx}: {response.status_code}")


        return file_paths

    except Exception as e:
        print(f"Error generating image: {e}")
        return []
