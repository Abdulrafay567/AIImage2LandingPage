from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
from cddd import generate_code_from_imagee,encode_image,parse_html_and_css
from generate_prompt import DesignPrompt, enhance_prompt
#from flux_image import generate_image
#from generate_code import generate_code_from_imagee
import uuid
from pydantic import BaseModel
from typing import List
# from image_segmentation import segment_image  # Import the segmentation function
# from fastapi import Body
from fastapi import FastAPI
import requests
import os
import replicate
import uuid
import requests
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from generate_prompt import enhance_prompt, DesignPrompt
# from screen_shot import capture_screenshot
import openai
from fastapi import FastAPI, Form, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import json
# import re
import shutil
import uuid
import requests
from typing import List
from pydantic import BaseModel, Field
from typing import Dict
import uvicorn
from image_segmentation import generate_image
from florence2 import captioning_landingPage,gpt_image,get_bounding_boxes
# Load environment variables
load_dotenv()
# openai.api_key = os.environ.get("OPENAI_KEY")
# print("OpenAI API Key:", openai.api_key)
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")



# Base directories to save images
IMAGE_BASE_DIR = "images/users"  # For main images
IMAGE_BASE_DIRR = "segment/new"  # For sub-images
os.makedirs(IMAGE_BASE_DIR, exist_ok=True)
  # Create base directory if not exists
  # Create base directory if not exists
import logging

# Configure logging settings
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

logger.debug("Debug logging is enabled.")  # Test log

# Initialize logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO for better debugging
logger = logging.getLogger(__name__)
logger.info("Logger is initialized and ready to use")

# Initialize FastAPI app
app = FastAPI()

# Mount static files for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store latest generation data
latest_generation = {}
enhanced_prompt = ""
if not REPLICATE_API_KEY:
    raise ValueError("Replicate API key not found. Please set it in the .env file.")

# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_KEY)
def remove_first_sentence(text):
    sentences = text.split(". ", 1)  # Split at the first full stop
    return sentences[1] if len(sentences) > 1 else ""
# In main.py, add a route for CSS if needed
@app.get("/styles.css")
async def get_css():
    return FileResponse("static/style.css", media_type="text/css")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Returns an HTML page that displays generated images and enhanced prompt."""
    return templates.TemplateResponse("index.html", {"request": request})
    

@app.post("/generate-images", response_class=JSONResponse)
async def generate_images(
    prompt: str = Form(...),
    website_name: str = Form(...),
    website_type: str = Form(...),
    design_philosophy: str = Form(...),
    color_scheme: str = Form(...),
):
    global latest_generation
    """Generates main images only using Replicate AI."""
    try:
        # Step 1: Enhance the prompt using OpenAI
        design_prompt = DesignPrompt(
            prompt=prompt,
            website_name=website_name,
            website_type=website_type,
            design_philosophy=design_philosophy,
            color_scheme=color_scheme,
        )
        global enhanced_prompt
        enhanced_prompt = enhance_prompt(design_prompt)
        if not enhanced_prompt:
            raise HTTPException(status_code=500, detail="Failed to enhance prompt.")

        # Step 2: Create a unique folder for this generation
        generation_id = str(uuid.uuid4())
        image_dir = os.path.join(IMAGE_BASE_DIR, generation_id)
        os.makedirs(image_dir, exist_ok=True)

        # Step 3: Generate main images (4 images)
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

        # Step 4: Download and save images
        file_paths = []
        for idx, img_url in enumerate(output):
            file_path = os.path.join(image_dir, f"output_{idx}.webp")
            response = requests.get(img_url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                file_paths.append(f"/images/users/{generation_id}/output_{idx}.webp")
            else:
                print(f"Failed to download image {idx}: {response.status_code}")

        if not file_paths:
            raise HTTPException(status_code=500, detail="Failed to generate images.")

        # Store the latest generated images
        # Store the latest generated images
        global latest_generation
        # latest_generation = ["enhanced_prompt"]= enhanced_prompt
        # latest_generation = ["generated_images"]= file_paths
        latest_generation = {
            "enhanced_prompt": enhanced_prompt,
            "generated_images": file_paths
        }

        return {"enhanced_prompt": enhanced_prompt, "generated_images": file_paths}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest-generation", response_class=JSONResponse)
async def get_latest_generation():
    """Returns the latest enhanced prompt and generated images."""
    if not latest_generation:
        return {"message": "No generation details found. Please generate images first."}
    return latest_generation


class ImageSelectionRequest(BaseModel):
    images: List[str]
    selected_index: int  # User should send the index of the image they choose
@app.post("/select-image", response_model=dict)
async def select_image(request: ImageSelectionRequest):
    """Processes the selected image and stores segmentation details."""
    images = request.images
    selected_idx = request.selected_index

    if not (0 <= selected_idx < len(images)):
        raise HTTPException(status_code=400, detail="Invalid image selection index")

    selected_image_url = images[selected_idx]

    try:
        image_path = os.path.join(IMAGE_BASE_DIR, selected_image_url.split("/")[-2], selected_image_url.split("/")[-1])
        if not os.path.exists(image_path):
            logger.error(f"Image not found at path: {image_path}")
            raise HTTPException(status_code=404, detail="Selected image not found.")
        
        florence_prompt = captioning_landingPage(image_path)
        florence_prompt = florence_prompt.strip().removeprefix("```json").removesuffix("```").strip()
        florence_prompt = json.loads(florence_prompt)
        image_prompts = gpt_image(image_path)
        image_prompts = image_prompts.strip().removeprefix("```json").removesuffix("```").strip()
        image_prompts = json.loads(image_prompts)
        prompts = [item['prompt'] for item in image_prompts]
        # prompts = "images,icons,logos,backgrounds,buttons,headers,footers,sections,containers,divs"
        output = get_bounding_boxes(image_path, prompts)
        text_output = output['text']
        print("output",text_output)
        
        segmentation_text = florence_prompt
       
        global latest_generation
        latest_generation["segmentation_details"] = segmentation_text
        latest_generation["selected_image"] = selected_image_url
        latest_generation["image_prompts"] = image_prompts
        latest_generation["bb_boxes"] = text_output

        return {"selected_image": selected_image_url, "segmentation_details": segmentation_text}

    except Exception as e:
        logger.error(f"Error in image segmentation: {e}")
        raise HTTPException(status_code=500, detail="Failed to segment image.")#e directory to store sub-images
IMAGE_BASE_DIRR = "segment/new"
os.makedirs(IMAGE_BASE_DIRR, exist_ok=True)

class SubImageRequest(BaseModel):
    prompts: Dict[str, str] = Field(..., description="A dictionary of prompts where keys are identifiers and values are prompt strings.")

@app.post("/generate-sub-images", response_class=JSONResponse)
async def generate_sub_images():
    """Generates sub-images dynamically based on provided prompts and returns their URLs."""
    try:
        global latest_generation
        # print("latest_generation here",latest_generation)
        
        if isinstance(latest_generation, str):
            segmented_data = json.loads(latest_generation)  # Parse only if it's a string
        else:
            segmented_data = latest_generation  # It's already a dictionary, so use it directly
        # segmentation_details = segmented_data.get('segmentation_details')
        image_prompt = segmented_data.get('image_prompts')
            # If "segmentation_details" is still a JSON string, load it as well
   
        # Check if the segmentation data contains an error
        if "error" in segmented_data:
            raise HTTPException(status_code=400, detail=segmented_data["error"])

        # Extract sub-image prompts
        # sub_image_prompts = {key: value["prompt"] for key, value in image_prompt.items() if "prompt" in value}
        sub_image_prompts = {f"prompt_{i}": item["prompt"] for i, item in enumerate(image_prompt)}

        # print("sub_image_prompts", sub_image_prompts)

        if not sub_image_prompts:
            raise HTTPException(status_code=400, detail="No sub-image prompts found in segmentation details.")

        # Generate sub-images
        generation_id = str(uuid.uuid4())
        image_dir = os.path.join(IMAGE_BASE_DIRR, generation_id)
        os.makedirs(image_dir, exist_ok=True)

        results = {}
        for key, prompt in sub_image_prompts.items():
            try:
                logger.info(f"Generating sub-image for {key}: {prompt}")
                image_files = generate_image(prompts=[prompt])

                if not image_files or not isinstance(image_files, dict):
                    logger.error(f"generate_image returned invalid or empty result for prompt: {prompt}, got: {image_files}")
                    continue

                image_urls = []
                for i, file_path in image_files.items():
                    if not os.path.exists(file_path):
                        logger.error(f"Generated file not found: {file_path}")
                        continue
                    new_filename = f"{key}_image_{i}.webp"
                    new_file_path = os.path.join(image_dir, new_filename)
                    shutil.copy2(file_path, new_file_path)
                    image_url = f"segment/new/{generation_id}/{new_filename}"
                    image_urls.append(image_url)

                if image_urls:
                    results[key] = image_urls

            except Exception as e:
                logger.error(f"Error processing prompt {key}: {e}", exc_info=True)
                continue
        # Ensure at least some images were generated
        if not results:
            logger.error("No images were generated.")
            raise HTTPException(status_code=500, detail="Sub-image generation failed. No images were produced.")


        # Update latest_generation with results
        latest_generation["generated_images"] = results
        # print("results",results)
        # Merge dictionaries and remove prompt_0
        merged = {}
        for key in results:
            merged[key] = {
                'image_path': results[key][0],  # Assuming single image per prompt
                'prompt': sub_image_prompts.get(key, '')  # Get corresponding prompt or empty string
            }
        print("merged",merged)    
        latest_generation["generation_id"] = generation_id
        latest_generation["sub_image"] = merged
        # latest_generation["background"] = background_image

        # logger.info(f"Updated generated_images: {results}")

        return {
            "success": True,
            "generation_id": generation_id,
            "results": results,
            "image_count": sum(len(urls) for urls in results.values())
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in /generate-sub-images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/segment/new/{generation_id}/{image_name}")
async def get_sub_image(generation_id: str, image_name: str):
    """Serve the generated sub-images."""
    image_path = os.path.join(IMAGE_BASE_DIRR, generation_id, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/webp")
    else:
        raise HTTPException(status_code=404, detail="Sub-image not found.")

class ImageSelectionRequest(BaseModel):
    images: List[str]
    selected_index: int
    

@app.post("/generate-code", response_class=JSONResponse)
async def generate_code(
    selected_image: str = Form(...),
    website_name: str = Form(default="My Website"),
    design_philosophy: str = Form(default="modern"),
    color_scheme: str = Form(default="neutral"),
    logo_text: str = Form(default="Logo"),
    toolbar_text: str = Form(default="Home, About, Contact"),
):
    global enhanced_prompt
    # print("enhanced_promptpppppppp",enhanced_prompt)
    global latest_generation
    if isinstance(latest_generation, str):
        segmented_data = json.loads(latest_generation)  # Parse only if it's a string
    else:
        segmented_data = latest_generation  # It's already a dictionary, so use it directly
        print("latest_generation",latest_generation)
        segmentation_details = segmented_data.get('segmentation_details')
        sub_image_data = segmented_data.get('sub_image')
        bb_boxes = segmented_data.get('bb_boxes')
    """Generates code using the segmented image details and all sub-images."""
    try:
      
     
        selected_image_path = os.path.normpath(selected_image.lstrip("/"))
        logger.info(f"Selected image path: {selected_image_path}")
        encoded_image = encode_image(selected_image_path)
        # print("florence_prompt",latest_generation)
        # Extract locations and sizes
        data = latest_generation.get("segmentation_details")


        initial_prompt = segmentation_details
       
        generated_code = generate_code_from_imagee(
            prompt=initial_prompt,
            bb_boxes= bb_boxes,
            encoded_image=encoded_image,
            sub_image_data=sub_image_data,  # Organized sub-images
        )
        
        print(generated_code)
        # Parse the OpenAI response to extract HTML and CSS
        html_code, css_code = parse_html_and_css(generated_code)
       
        
        # Save the generated code to files
        with open("generated_landing_page.html", "w") as html_file:
            html_file.write(html_code)
        
        with open("styles.css", "w") as css_file:
            css_file.write(css_code)
        
        print("Generated code saved to 'generated_landing_page.html' and 'styles.css'")
        return {"html": html_code, "css": css_code}

    except Exception as e:
        logger.error(f"Error in /generate-code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/users/{generation_id}/{image_name}")
async def get_image(generation_id: str, image_name: str):
    """Serve the generated main images."""
    image_path = os.path.join(IMAGE_BASE_DIR, generation_id, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/webp")
    else:
        raise HTTPException(status_code=404, detail="Image not found.")


@app.get("/segment/new/{generation_id}/{image_name}")
async def get_sub_image(generation_id: str, image_name: str):
    """Serve the generated sub-images."""
    image_path = os.path.join(IMAGE_BASE_DIRR, generation_id, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/webp")
    else:
        raise HTTPException(status_code=404, detail="Sub-image not found.")




if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="debug", reload=True)

