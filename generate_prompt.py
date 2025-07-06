import openai
import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
class DesignPrompt(BaseModel):
    prompt: str
    website_type: str
    design_philosophy: str
    color_scheme: str
    website_name: str

def enhance_prompt_for_code() -> str:
    """Enhance user prompt with design-specific terminology and structure based on the advanced prompting framework."""
    # Construct the enhanced prompt
    enhanced_prompt = f"""
    Create a landing page that matches as closely as possible to the attached design image'.
    """
    return enhanced_prompt   

def enhance_prompt(basic_prompt) -> str:
    """Enhance user prompt with design-specific terminology."""
    enhanced_prompt = f"""
    Create a {basic_prompt.website_type} website named '{basic_prompt.website_name}' with a {basic_prompt.design_philosophy} design and {basic_prompt.color_scheme} color scheme.
    Objective: {basic_prompt.prompt}
    
    **Landing Page Features:**
    - **Header:** Logo (top left), Navigation menu, Search bar (if needed)
    - **Hero Section:** H1 headline (USP), H2 subheadline, Hero image, CTA button
    - **Content:** Clear hierarchy, Key features, Benefit-driven copy, Relevant visuals
    - **Social Proof:** Testimonials, Partner logos, User reviews
    - **CTA:** Action-driven buttons, Multiple placements
    - **Lead Capture:** Minimalist form, Value proposition, Privacy statement
    - **Footer:** Contact details, Social links, Secondary navigation, Legal info
    
    Ensure **responsiveness, fast performance, and accessibility best practices**.
    """
    return generate_enhanced_prompt(enhanced_prompt)

def generate_enhanced_prompt(basic_prompt):
    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_KEY")
    # print("OpenAI API Key3:", openai.api_key)
    sample_prompts = [
        """Magnificent UI screenshot of a dark-mode AI video editor, Clapper. Features include timeline editor, script editor, video preview, and generative AI. Hero title: 'Clapper. The future of AI cinema, today.' Inspired by Apple products and Dribbble mock-ups.""",
        """Fullscreen front-page of e-commerce site 'Elder Engineering.' Soft grey background, dark grey text. Toolbar includes logo, Products, Contact, About. Features multiple Minimoog synth angles, shopping cart with order summary, and sleek design."""
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""Improve the given prompt by ensuring it includes:
                - Website name and industry
                - Hero section with USP and supporting headline
                - Design philosophy and color scheme
                - Key features (toolbar, search bar, logo info, etc.)
                - Relevant visual and camera view details (if applicable)
                
                If missing, improvise professionally. 
                Structure response as a refined paragraph, following examples: {sample_prompts}
                Avoid unnecessary details and JSON formatting."""},
                {"role": "user", "content": basic_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

   
    



if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    prompt="create a landing page for an e-commerce store"
    website_name = "buy-all"
    website_type="E-commerce"
    design_philosophy="colorfull"
    color_scheme= "light tones"
    design_prompt = DesignPrompt(
            prompt=prompt,
            website_name=website_name,
            website_type=website_type,
            design_philosophy=design_philosophy,
            color_scheme=color_scheme,
        )
    (enhance_prompt(design_prompt))