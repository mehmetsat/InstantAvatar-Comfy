import gradio as gr
import requests
from PIL import Image
import io
from prompts import prompt_dict
# ===========================
# API Interaction Function
# ===========================

def resize_image(image, size=(512, 512)):
    """Resize an image to fit within the specified size while maintaining aspect ratio."""
    # Calculate the aspect ratio
    aspect = image.width / image.height

    # Calculate new dimensions
    if aspect > 1:
        # Width is greater, so scale based on width
        new_width = size[0]
        new_height = int(new_width / aspect)
    else:
        # Height is greater or equal, so scale based on height
        new_height = size[1]
        new_width = int(new_height * aspect)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def generate_image_final_via_api(
    input_image: Image.Image,
    style: str,
    gender: str
) -> Image.Image:
    """Generates an image by sending a request to the generate-image-final API endpoint."""
    url = "http://localhost:8000/generate-image-final/"
    
    # Convert the input image to bytes
    input_image = resize_image(input_image)
    input_image_bytes = io.BytesIO()
    input_image.save(input_image_bytes, format='JPEG')
    input_image_bytes.seek(0)

    # Prepare the files and data for the request
    files = {
        "input_image": input_image_bytes
    }
    data = {
        "style": style,
        "gender": gender
    }
    
    # Send the POST request
    response = requests.post(url, files=files, data=data)
    
    # Check the response
    if response.status_code == 200:
        # Get the image path from the response
        image_path = response.json().get("image_path")
        if image_path:
            # Open the image from the path
       
            image = Image.open(image_path)
            return image
        else:
            raise Exception("No image path returned in the response.")
    else:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

# ===========================
# Gradio Interface Setup
# ===========================

def main_gradio():
    with gr.Blocks(css="""
        .container { max-width: 800px; margin: auto; }
        .gr-dropdown { font-size: 16px; padding: 10px; }  /* Increase font size and padding */
    """) as demo:
        gr.Markdown("""
        # Instant Avatar
        Transform your photos into stunning avatars with different styles and genders.
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                style = gr.Dropdown(
                    label="Style",
                    choices=prompt_dict["male"].keys(),
                    value="explorer",
                    elem_id="style-dropdown"  # Add an element ID for specific styling
                )
                gender = gr.Dropdown(
                    label="Gender",
                    choices=["female", "male"],
                    value="female"
                )
                generate_button = gr.Button("Generate Image")
            with gr.Column():
                output_image = gr.Image(label="Output Image")
        
        # Connect the generate button to the API interaction function
        generate_button.click(
            fn=generate_image_final_via_api,
            inputs=[
                input_image,
                style,
                gender
            ],
            outputs=output_image
        )

    demo.launch(share=True)

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    main_gradio()
