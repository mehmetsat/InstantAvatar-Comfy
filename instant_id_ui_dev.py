import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from PIL import Image
import gradio as gr
import asyncio
from torchvision import transforms
# ===========================
# Utility Functions
# ===========================

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """Recursively finds the path for a given name starting from the provided path."""
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    """Add 'ComfyUI' to the sys.path."""
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    """Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path."""
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

# ===========================
# Model Manager Class
# ===========================

class ModelManager:
    def __init__(self):

        self.load_models()

    def load_models(self):
        """Initializes and loads all necessary models."""
        print("Initializing models...")
        add_comfyui_directory_to_sys_path()
        add_extra_model_paths()
        self.import_custom_nodes()

        # Initialize NODE_CLASS_MAPPINGS
        from nodes import NODE_CLASS_MAPPINGS
        self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS

        # Load models
        with torch.inference_mode():
            self.checkpointloadersimple = self.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            self.checkpointloadersimple_4 = self.checkpointloadersimple.load_checkpoint(
                ckpt_name="realvisxlV50_v30TurboBakedvae.safetensors"
            )

            self.emptylatentimage = self.NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            self.emptylatentimage_5 = self.emptylatentimage.generate(
                width=1024, height=1024, batch_size=1
            )

            self.instantidmodelloader = self.NODE_CLASS_MAPPINGS["InstantIDModelLoader"]()
            self.instantidmodelloader_11 = self.instantidmodelloader.load_model(
                instantid_file="ip-adapter.bin"
            )

            self.controlnetloader = self.NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            self.controlnetloader_16 = self.controlnetloader.load_controlnet(
                control_net_name="diffusion_pytorch_model.safetensors"
            )

            self.instantidfaceanalysis = self.NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]()
            self.instantidfaceanalysis_38 = self.instantidfaceanalysis.load_insight_face(
                provider="CPU"
            )
            

            self.cliptextencode = self.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            
            self.vaeencode = self.NODE_CLASS_MAPPINGS["VAEEncode"]()
            self.applyinstantid = self.NODE_CLASS_MAPPINGS["ApplyInstantID"]()
            self.ksampler = self.NODE_CLASS_MAPPINGS["KSampler"]()
            self.vaedecode = self.NODE_CLASS_MAPPINGS["VAEDecode"]()
           

        print("Models initialized successfully.")

    def import_custom_nodes(self) -> None:
        """Import and initialize custom nodes."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        import execution
        from nodes import init_extra_nodes
        import server

        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        init_extra_nodes()

# ===========================
# Generate Image Function
# ===========================

def generate_image(
    input_image: Image.Image,
    pose_image: Image.Image,  # New parameter for pose image
    init_image: Image.Image, # Input image for Img2Img
    positive_prompt: str,
    negative_prompt: str,
    seed: int = None,
    steps: int = 15,
    cfg_scale: float = 2.0,
    sampler_name: str = "ddpm",
    scheduler: str = "karras",
    model_manager: ModelManager = None
) -> Image.Image:
    """Generates an image based on the provided inputs using pre-loaded models."""
    if model_manager is None:
        raise ValueError("Model Manager is not initialized.")

    with torch.inference_mode():

        
        # Define a transform to convert PIL images to tensors
        pil_to_tensor = transforms.ToTensor()
        
        # Convert the input images to tensors
        input_image_tensor = pil_to_tensor(input_image)
        pose_image_tensor = pil_to_tensor(pose_image)
        init_image_tensor = pil_to_tensor(init_image)
        

        loadimage_13 = input_image_tensor.unsqueeze(0).permute(0,2,3,1)
        loadimage_67 = pose_image_tensor.unsqueeze(0).permute(0,2,3,1)
        loadimage_70 = init_image_tensor.unsqueeze(0).permute(0,2,3,1)   
  

        # Encode prompts
        checkpoint = get_value_at_index(model_manager.checkpointloadersimple_4, 1)
        cliptextencode_39 = model_manager.cliptextencode.encode(
            text=positive_prompt,
            clip=checkpoint,
        )
        print(f"Positive prompt encoded")
        cliptextencode_40 = model_manager.cliptextencode.encode(
            text=negative_prompt,
            clip=checkpoint
        )
        print(f"Negative prompt encoded")

        
        vaeencode_68 = model_manager.vaeencode.encode(
            pixels=loadimage_70,
            vae=get_value_at_index(model_manager.checkpointloadersimple_4, 2),
        )
        print(f"VAEEncode done")
        # Set seed if provided
        if seed != -1:
            random.seed(seed)
        else:
            seed = random.randint(1, 2**64)

        applyinstantid_60 = model_manager.applyinstantid.apply_instantid(
            weight=0.8,
            start_at=0,
            end_at=1,
            instantid=get_value_at_index(model_manager.instantidmodelloader_11, 0),
            insightface=get_value_at_index(model_manager.instantidfaceanalysis_38, 0),
            control_net=get_value_at_index(model_manager.controlnetloader_16, 0),
            image=loadimage_13,
            model=get_value_at_index(model_manager.checkpointloadersimple_4, 0),
            positive=get_value_at_index(cliptextencode_39, 0),
            negative=get_value_at_index(cliptextencode_40, 0),
            image_kps=loadimage_67,  # Use the pose image here
        )
        print(f"ApplyInstantID done")


        ksampler_3 = model_manager.ksampler.sample(
            seed=seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=0.9,
            model=get_value_at_index(applyinstantid_60, 0),
            positive=get_value_at_index(applyinstantid_60, 1),
            negative=get_value_at_index(applyinstantid_60, 2),
            latent_image=get_value_at_index(vaeencode_68, 0),
        )
        print(f"KSampler done")
        
        vaedecode_8 = model_manager.vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(model_manager.checkpointloadersimple_4, 2),
        )
        print(f"VAEDecode done")
        
        # Convert tensor to image
        tensor = vaedecode_8[0] 
        image_tensor = tensor.squeeze(0)
        image_np = image_tensor.numpy()
        image_np_uint8 = (image_np * 255).astype(np.uint8)
        output_image = Image.fromarray(image_np_uint8)
        print(f"Output image generated")

        return output_image

# ===========================
# Gradio Interface Setup
# ===========================

def main_gradio():
    # Initialize ModelManager once
    model_manager = ModelManager()

    with gr.Blocks() as demo:
        gr.Markdown("# ComfyUI Image Generator")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                pose_image = gr.Image(label="Pose Image", type="pil")  # New Image input for pose
                init_image = gr.Image(label="Init Image", type="pil")  # New Image input for Img2Img
                positive_prompt = gr.Textbox(
                    label="Positive Prompt",
                    value="roman empire photo of a person high details captured with carl zeiss 1um lens"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="deformed eyes, blurry shot, deformed face, Unflattering, distorted, awkward, forced, unnatural, stiff, uncomfortable, insincere, cliched, unrealistic, inaccurate, low-quality, pixelated, blurry, distorted, poorly composed, messy, cluttered, unprofessional."
                )
                seed = gr.Number(label="Seed (Optional)", value=-1)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=15)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=0.1, maximum=20.0, step=0.1, value=2.0)
                sampler_name = gr.Dropdown(
                    label="Sampler Name",
                    choices=["ddpm"],  # add more samplers
                    value="ddpm"
                )
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=["karras"],  # add more schedulers
                    value="karras"
                )
                generate_button = gr.Button("Generate Image")
            with gr.Column():
                output_image = gr.Image(label="Output Image")
        
        # Define a State component to hold the ModelManager
        state = gr.State(model_manager)

        # Define the image generation callback with access to the state
        def generate_image_callback(
            input_image,
            pose_image,  # Include pose_image in the callback
            init_image,
            positive_prompt,
            negative_prompt,
            seed,
            steps,
            cfg_scale,
            sampler_name,
            scheduler,
            state: ModelManager
        ):
            return generate_image(
                input_image=input_image,
                pose_image=pose_image,  # Pass the pose image
                init_image=init_image,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                model_manager=state
            )

        # Connect the generate button to the callback, passing the state
        generate_button.click(
            fn=generate_image_callback,
            inputs=[
                input_image,
                pose_image,  # Add pose_image as input
                init_image,
                positive_prompt,
                negative_prompt,
                seed,
                steps,
                cfg_scale,
                sampler_name,
                scheduler,
                state
            ],
            outputs=output_image
        )

    demo.launch(share=True)

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    main_gradio()
