import requests
from PIL import Image
import io
import random


def test_generate_image():
    url = "http://localhost:8000/generate-image/"
    
    # Open test images
    with open("/home/mehmetsat-extra/personal/InstantAvatar-Comfy/template_images/face_photos/image (30).webp", "rb") as input_image_file, \
         open("/home/mehmetsat-extra/personal/InstantAvatar-Comfy/template_images/init_images/thrillisgon_A_portrait_of_a_young_sorceress_with_deep_violet__ef793e53-1366-4306-82c2-d84d5cdf5923_2.png", "rb") as pose_image_file, \
         open("/home/mehmetsat-extra/personal/InstantAvatar-Comfy/template_images/init_images/thrillisgon_A_celestial_mage_with_soft_glowing_skin_and_silve_9c4973ba-252a-43cb-a211-baa25e2943f5_2.png", "rb") as init_image_file:
        
        # Prepare the files and data for the request
        files = {
            "input_image": input_image_file,
            "pose_image": pose_image_file,
            "init_image": init_image_file
        }
        seed = random.randint(1, 2**64)
        print(seed)
        data = {
            "positive_prompt": "A portrait of a young sorceress with deep violet eyes and flowing raven-black hair. She is dressed in a flowing emerald-green robe that shimmers with magical energy, and a silver pendant glows around her neck. Her hands are raised, conjuring swirling wisps of light. The backdrop shows a moonlit forest, with magical creatures faintly visible in the shadows.",
            "negative_prompt": "Blurry, low-quality",
            "seed": seed,
            "steps": 20,
            "cfg_scale": 2.0,
            "sampler_name": "ddpm",
            "scheduler": "karras"
        }
        
        # Send the POST request
        response = requests.post(url, files=files, data=data, stream=True)
        
        # Check the response
        if response.status_code == 200:
            # Get the image path from the response
            image_path = response.json().get("image_path")
            if image_path:
                print(f"Test passed: Image generated and saved at {image_path}.")
            else:
                print("Test failed: No image path returned.")
        else:
            print(f"Test failed: {response.status_code} - {response.text}")

def test_generate_image_full():
    url = "http://localhost:8000/generate-image-final/"
    
    # Open test images
    with open("/home/mehmetsat-extra/personal/InstantAvatar-Comfy/template_images/face_photos/image (30).webp", "rb") as input_image_file:
        
        # Prepare the files and data for the request
        files = {
            "input_image": input_image_file
        }
        seed = random.randint(1, 2**64)
        print(seed)
        data = {
            "style" : "explorer",
            "gender" : "female"
        }
        
        # Send the POST request
        response = requests.post(url, files=files, data=data, stream=True)
        
        # Check the response
        if response.status_code == 200:
            # Get the image path from the response
            image_path = response.json().get("image_path")
            if image_path:
                print(f"Test passed: Image generated and saved at {image_path}.")
            else:
                print("Test failed: No image path returned.")
        else:
            print(f"Test failed: {response.status_code} - {response.text}")

# Run the test function
if __name__ == "__main__":
    test_generate_image_full()