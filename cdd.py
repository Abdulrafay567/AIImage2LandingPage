import openai
import logging
import os
import json
from dotenv import load_dotenv
import re
from pydantic import BaseModel
import base64


load_dotenv()
IMAGE_BASE_DIR = "images/users"
os.makedirs(IMAGE_BASE_DIR, exist_ok=True)

# Verify OpenAI API key
openai.api_key = os.environ.get("OPENAI_KEY")
# print("OpenAI API Key2:", openai.api_key)
# if not openai_key:
#     raise ValueError("OPENAI_KEY not found. Please set it in the .env file.")
# openai.api_key = openai_key

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class CodeResponse(BaseModel):
    html_code: str
    css_code: str

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise Exception(f"Image not found at path: {image_path}")

def generate_code_from_imagee(
    prompt: str,
    bb_boxes: dict,
    encoded_image: str,
    sub_image_data: dict = None,
) -> str:
    """
    Generates HTML and CSS code based on the given prompt and image data using OpenAI API.
    """
    try:
   
        user = f"""{prompt}\n
use the following sub-images paths at the correct locations. their backgrounds have been removed so the background colour will be visible and they need to be resized accordingly:\n"{json.dumps(sub_image_data, indent=2)}\n 
**Output Requirements**
- **Return only valid HTML and CSS code**  
- **Ensure all extracted details (backgrounds, fonts, colors, images, spacing) are reflected in the code**  
- **Do not include any explanations, leading/trailing text, or markers like ```html, ```css and ``` outside the specified format.**

**Use Bootstrap for Styling**  
   - Ensure the use of Bootstrap for layout, grids, buttons, and responsive behavior.  
   - Apply Bootstrap's utilities for margins, paddings, typography, and colors.
   - resize icons and logos for example: <img src="..." alt="User Icon" class="mb-3" style="width: 100px; height: auto;">,<img src="..." alt="elite logo" class="logo" style="width: 200px; height: auto;">
### **Expected Output Format**
html
<!-- HTML Code Here -->
css
/* CSS Code Here */"""
#         # Define the messages structure
#         user_prompt = f""""You are an expert frontend developer specializing in HTML5, CSS3, Bootstrap, and modern web design principles. Your task is to generate pixel-perfect HTML and CSS code that replicates the provided landing page image as closely as possible. Ensure the generated code is clean, semantic, and fully responsive. Follow best practices for structure, styling, and accessibility by **identifying the sections present in the landing page**.
# ### **Design Guidelines**
# 1. **Identify Sections and Their Relative Sizes**  
#    - Analyze and define the **exact size and area** occupied by each section (navbar, hero, features, footer, etc.) using the information below.  
#    - Ensure proper spacing between sections. 
#    - Use the following section and image bounding boxes in coco format: {json.dumps(sub_image_data, indent=2)}\n"
#these are the bounding boxes for the some images detected using an ai model, use these locations if the generated images prompts matches these labels detected for accurate placements of the images:\n"{json.dumps(bb_boxes, indent=2)}\n
# 2. **Extract Background Colors, Font Colors, and Font Styles**  
#    - Important: Background color must be specified for each section and it should match the attached image landing page   
#    - Extract **font color, size, and weight** for each text element in each section.  
#    - Ensure fonts match the original image (e.g., 'Roboto', 'Lato', or as specified).  

# 3. **Identify Image Placement and Sizes**  
#    - Identify the **size and positioning** of images used in each section.   
#    - If an image spans an entire section, use it as a CSS `background-image`.  
#    - For inline images, ensure they are responsive with `width: 100%; max-width: Xpx; height: auto`.
#    - use all the sub-images but NOT the 'selected_image' from this data with appropriate sizing and locations as specified including the logo:\n"
# {json.dumps(image_data, indent=2)}\n"  

# 4. **Use Bootstrap for Styling**  
#    - Ensure the use of Bootstrap for layout, grids, buttons, and responsive behavior.  
#    - Apply Bootstrap's utilities for margins, paddings, typography, and colors.  
#    - Use the following Bootstrap links:  
#      ```html
#      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" 
#            rel="stylesheet" 
#            integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" 
#            crossorigin="anonymous">
#      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" 
#              integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" 
#              crossorigin="anonymous"></script>

# ### **Output Requirements**
# - **Return only valid HTML and CSS code**  
# - **Ensure all extracted details (backgrounds, fonts, colors, images, spacing) are reflected in the code**  
# - **Do not include any explanations, leading/trailing text, or markers like ```html, ```css and ``` outside the specified format.**  

# ### **Expected Output Format**
# html
# <!-- HTML Code Here -->
# css
# /* CSS Code Here */
# """
        messages = [
            {
                "role": "system",
                "content": "You are an expert frontend developer specializing in HTML5, CSS3,Bootstrap and modern web design principles."
                "Your task is to generate HTML and CSS code that replicates the landing page image generated using ai as closely as possible."
                "Ensure the generated code is clean, semantic, and fully responsive."
                "Populate the landingpage with relevant text as shown in thw image using the instructions provided by the user"
                "Follow best practices for structure, styling, and accessibility."
            },       
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user  
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
        print("pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp",user)
        # Send the request to OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            #response_format={"type": "json_object"}  # Force JSON response format
        )

        # Return the generated code
        openai_response = response.choices[0].message.content
        return openai_response
    
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}", exc_info=True)
        raise
def parse_html_and_css(text):
    """
    Parse the response text to extract HTML and CSS components.
    
    Args:
        text (str): The raw response text
        
    Returns:
        tuple: (html_part, css_part)
    """
    # Remove unnecessary labels and backticks from the entire text
    cleaned_text = re.sub(r'```html|```css|```|html\s*:\s*|css\s*:\s*', '', text, flags=re.IGNORECASE)
    
    # Split the text into HTML and CSS parts based on the closing </html> tag followed by a newline
    parts = re.split(r'(</html>)\s*\n', cleaned_text, maxsplit=1, flags=re.IGNORECASE)
    
    # Combine the HTML parts (the closing tag is captured in the split)
    if parts and len(parts) >= 2:
        html_part = (parts[0] + parts[1]).strip()
    else:
        html_part = ""
    
    # The CSS part is the remainder after the HTML block
    css_part = parts[2].strip() if len(parts) > 2 else ""
    
    # Remove any extra backticks that might be left behind
    html_part = html_part.replace('```', '').strip()
    css_part = css_part.replace('```', '').strip()
    
    return html_part, css_part
def parse_response(response_text):
    """
    Parse the OpenAI response into HTML and CSS components.
    
    Args:
        response_text (str): The raw response from OpenAI
        
    Returns:
        dict: A dictionary with 'html' and 'css' keys
    """
    try:
        # Try to parse as JSON
        try:
            parsed_json = json.loads(response_text)
            if isinstance(parsed_json, dict) and "html" in parsed_json and "css" in parsed_json:
                return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            
        # If JSON parsing fails, try to extract HTML and CSS using regex
        html_match = re.search(r'"html"\s*:\s*"((?:\\.|[^"\\])*)"', response_text, re.DOTALL)
        css_match = re.search(r'"css"\s*:\s*"((?:\\.|[^"\\])*)"', response_text, re.DOTALL)
        
        if html_match and css_match:
            html_code = html_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            css_code = css_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            return {"html": html_code, "css": css_code}
            
        # If all parsing attempts fail, return empty dictionary
        logger.error("Failed to parse OpenAI response using any method")
        return {"html": "", "css": ""}
    
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}", exc_info=True)
        return {"html": "", "css": ""}
