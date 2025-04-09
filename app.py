import os
import re
import requests
import time
import gradio as gr
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from PIL import Image, ImageDraw, ImageFont, ImageColor
import textwrap
import tempfile
import zipfile
import trafilatura
from io import BytesIO
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Groq API Config
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

CUSTOM_FONTS = {
    "Default": None,
}

# Rate Limit Throttle
class TokenBucket:
    def __init__(self, max_tokens, refill_rate):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        added_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + added_tokens)
        self.last_refill = now

    def consume(self, amount):
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return 0
        else:
            needed = amount - self.tokens
            wait_time = needed / self.refill_rate
            return wait_time

# Instantiate global token bucket for Groq (6000 TPM = 100 TPS)
token_bucket = TokenBucket(max_tokens=6000, refill_rate=100)

def safe_openai_chat_completion(prompt, temperature=0.5, max_tokens=512):
    estimated_input_tokens = len(prompt.split())
    estimated_total_tokens = estimated_input_tokens + max_tokens
    wait_time = token_bucket.consume(estimated_total_tokens)
    if wait_time > 0:
        print(f"[Throttle] Waiting {wait_time:.2f}s to avoid rate limit...")
        time.sleep(wait_time)

    response = openai.ChatCompletion.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"]

# Prompt Template
def summarize_chunk_direct(chunk, tone):
    prompt = f"""
You are a creative Instagram content strategist.
Summarize the content below into a 5-slide carousel post for Instagram. Each slide should be concise, self-contained (max 220 characters), and written in a {tone} tone.
Get straight to the content. DO NOT use something like 'Here is the slides' or things like that. Don't use useless separators. Get straight to the points for the slides. NEVER use hashtags.

Article:
---
{chunk}
---
Now write the 5 slides:
"""
    return safe_openai_chat_completion(prompt, temperature=0.5, max_tokens=512)

# Translation (optional)
def translate_text(text, target_language):
    if target_language == "Default / No Translation":
        return text

    prompt = f"""
You are a native speaker of {target_language} and a skilled copywriter. Paraphrase the following Instagram carousel slides into natural, engaging, and idiomatic {target_language}, as if written by a local social media expert. Avoid literal translation. Maintain the original meaning and tone, but prioritize natural phrasing and flow.
Get straight to the content and don't start with things like 'Here is your result:'. Also don't use useless separators. Get straight to the points for the slides. NEVER use hashtags.

Slides:
{text}
"""
    translated = safe_openai_chat_completion(prompt, temperature=0.7, max_tokens=512).strip()

    if translated.lower().startswith("here are"):
        lines = translated.splitlines()
        lines = [line for line in lines if line.strip() and not line.lower().startswith("here are")]
        translated = "\n".join(lines).strip()

    return translated

# Fetch Text
def fetch_pubmed_abstract(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=text&rettype=abstract"
    response = requests.get(url)
    return response.text.strip() if response.status_code == 200 else "Error fetching abstract."

def extract_text_from_pdf(file):
    loader = PyPDFLoader(file.name)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    result = trafilatura.extract(downloaded)

    if result is None:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                result = '\n'.join([p.get_text() for p in paragraphs if p.get_text().strip()])
        except Exception as e:
            print(f"[Fallback Error] {e}")
            result = None

    return result if result else "Unable to extract content from this URL."

def chunk_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def summarize_chunks(chunks, tone):
    return [summarize_chunk_direct(chunk, tone).strip() for chunk in chunks]

def consolidate_summary(partials, tone, desired_count):
    desired_count = int(desired_count)
    slides = []
    for part in partials:
        lines = [line.strip() for line in part.splitlines() if line.strip()]
        for line in lines:
            cleaned = clean_slide_text(line)
            if cleaned:
                slides.append(cleaned)
            if len(slides) >= desired_count:
                break
        if len(slides) >= desired_count:
            break
    if len(slides) < desired_count:
        remaining = desired_count - len(slides)
        last = slides[-1] if slides else ""
        slides += [last] * remaining
    return slides

def clean_slide_text(slide):
    return re.sub(r"(slide\s*\d+[:\-]?)", "", slide, flags=re.IGNORECASE).strip()

# Image Generator
def parse_color(color_string):
    if color_string.startswith("rgba"):
        numbers = re.findall(r"[\d.]+", color_string)
        return tuple(int(float(n)) for n in numbers[:3])
    elif color_string.startswith("#"):
        return color_string
    else:
        return color_string if color_string in ImageColor.colormap else "#ffffff"

def generate_carousel(pmid_or_text, pdf_file, url_input, tone,
                      bg_color, font_color, logo_file, custom_font_file,
                      translate, image_size, num_slides):

    def extract_dimensions(size_string):
      try:
          size_part = size_string.split("px")[0]
          width, height = map(int, size_part.split("x"))
          return width, height
      except Exception as e:
          raise ValueError(f"Invalid size format: {size_string}") from e

    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
    elif url_input:
        text = extract_text_from_url(url_input)
    elif pmid_or_text.strip().isdigit():
        text = fetch_pubmed_abstract(pmid_or_text.strip())
    else:
        text = pmid_or_text

    if len(text.split()) > 3000:
        return "❌ Error: Input exceeds 3000-word limit. Please shorten your text and try again.", [], None

    chunks = chunk_text(text)
    partials = summarize_chunks(chunks, tone)
    slides = consolidate_summary(partials, tone, int(num_slides))

    if translate != "Default / No Translation":
        slides = translate_text("\n".join(slides), translate).splitlines()

    image_width, image_height = extract_dimensions(image_size)
    image_paths = create_carousel_images(slides, bg_color, font_color, custom_font_file, logo_file, image_width, image_height)
    zip_path = create_zip_from_images(image_paths)

    return "Success!", image_paths, zip_path

# Create Carousel
def create_carousel_images(slides, bg_color, font_color, custom_font_file, logo_file, image_width, image_height):
    font_path = None
    if custom_font_file is not None:
        font_path = custom_font_file
    else:
        try:
            from matplotlib import font_manager
            font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        except Exception as e:
            print(f"[WARN] Fallback font failed: {e}")

    def get_font(size):
        try:
            if font_path:
                return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"[WARN] Failed to load font: {e}")
        return ImageFont.load_default()

    bg_color = parse_color(bg_color)
    font_color = parse_color(font_color)

    slide_images = []

    logo = None
    if logo_file is not None:
        logo = Image.open(logo_file.name).convert("RGBA").resize((120, 120))

    for i, slide_text in enumerate(slides):
        if not slide_text.strip():
            continue


        max_font_size = 75
        min_font_size = 20
        spacing_ratio = 0.25
        width, height = image_width, image_height
        text_area_width = width - 80
        text_area_height = height - 120
        font_size = max_font_size
        final_text = ""

        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        slide_text = slide_text.replace("\\n", "\n")

        while font_size >= min_font_size:
            font = get_font(font_size)
            spacing = int(font_size * spacing_ratio)

            words = slide_text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = (current_line + " " + word).strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] > text_area_width:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

            final_text = "\n".join(lines)
            bbox = draw.multiline_textbbox((0, 0), final_text, font=font, spacing=spacing)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if w <= text_area_width and h <= text_area_height:
                break

            font_size -= 2

        text_x = (width - w) // 2
        text_y = (height - h) // 2

        draw.multiline_text(
            (text_x, text_y),
            final_text,
            fill=font_color,
            font=font,
            spacing=spacing,
            align="center"
        )

        if logo:
            img.paste(logo, (width - 150, 30), logo)

        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        img.save(temp_path)
        slide_images.append(temp_path)

    return slide_images

def create_zip_from_images(image_paths):
    if not image_paths:
        raise ValueError("No images to zip.")
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))
    return zip_path

# Main Function
def generate_carousel(pmid_or_text, pdf_file, url_input, tone,
                      bg_color, font_color, logo_file, custom_font_file,
                      translate, image_size, num_slides):
    
    def extract_dimensions(size_string):
        try:
            size_part = size_string.split("px")[0]
            width, height = map(int, size_part.split("x"))
            return width, height
        except Exception as e:
            raise ValueError(f"Invalid size format: {size_string}") from e

    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
    elif url_input:
        text = extract_text_from_url(url_input)
    elif pmid_or_text.strip().isdigit():
        text = fetch_pubmed_abstract(pmid_or_text.strip())
    else:
        text = pmid_or_text

    if len(text.split()) > 3000:
        return "❌ Error: Input exceeds 3000-word limit. Please shorten your text and try again.", [], None

    chunks = chunk_text(text)
    partials = summarize_chunks(chunks, tone)
    final_text = consolidate_summary(partials, tone, num_slides)

    if translate != "Default / No Translation":
        final_text = translate_text(final_text, translate)

    slides = final_text[:num_slides]
    image_width, image_height = extract_dimensions(image_size)
    image_paths = create_carousel_images(
        slides, bg_color, font_color, custom_font_file, logo_file, image_width, image_height
    )
    zip_path = create_zip_from_images(image_paths)

    return "Success!", image_paths, zip_path

# Gradio UI
tone_choices = ["Academic", "Casual", "Gen Z", "Storytelling"]
color_choices = ["white", "black", "#f5f5f5", "#222", "#1e90ff", "#ff4081"]
language_choices = ["Default / No Translation", "English", "Bahasa Indonesia", "Spanish", "German"]
size_choices = [
    "1080x1080px - Default",
    "1080x1350px - Instagram Feed",
    "1080x1920px - Instagram Story"
]


interface_inputs = [
    gr.Textbox(label="Paste Text or PubMed ID"),
    gr.File(label="Upload PDF (Optional)"),
    gr.Textbox(label="Or paste a URL (Optional)"),
    gr.Dropdown(tone_choices, label="Choose Writing Tone", value="Casual"),
    gr.ColorPicker(label="Background Color", value="#ffffff"),
    gr.ColorPicker(label="Font Color", value="#000000"),
    gr.File(label="Upload Logo (Optional)"),
    gr.File(label="Upload Custom Font File (Optional)"),
    gr.Dropdown(language_choices, label="Translate slides to... (Optional)", value="Default / No Translation"),
    gr.Dropdown(size_choices, label="Image Size", value="1080x1080px - Default"),
    gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Slides")
]

demo = gr.Interface(
    fn=generate_carousel,
    inputs=interface_inputs,
    outputs=[
        gr.Text(label="Status Message"),
        gr.Gallery(label="Instagram Carousel Slides (Download Ready)", columns=5, height=300),
        gr.File(label="Download All Slides as ZIP")
    ],
    title="🧠 Visummary",
    description="Your best ally to summarize PDFs, URLs, or text into Instagram-ready carousel slide images."
)

if __name__ == "__main__":
    demo.launch(debug=True)