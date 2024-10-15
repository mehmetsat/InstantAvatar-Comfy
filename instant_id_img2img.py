import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="realvis-sdxl-turbo/realvisxlV30Turbo_v30TurboBakedvae.safetensors"
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_5 = emptylatentimage.generate(
            width=1016, height=1016, batch_size=1
        )

        instantidmodelloader = NODE_CLASS_MAPPINGS["InstantIDModelLoader"]()
        instantidmodelloader_11 = instantidmodelloader.load_model(
            instantid_file="ip-adapter.bin"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_13 = loadimage.load_image(image="IMG_2279.JPG")

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_16 = controlnetloader.load_controlnet(
            control_net_name="diffusion_pytorch_model.safetensors"
        )

        instantidfaceanalysis = NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]()
        instantidfaceanalysis_38 = instantidfaceanalysis.load_insight_face(
            provider="CPU"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_39 = cliptextencode.encode(
            text="A regal portrait of a Roman empress with delicate facial features and an elaborate hairstyle, styled in soft waves and pinned up with ornate golden hairpins. She is dressed in a fine stola, draped with a richly embroidered palla over her shoulder. A golden diadem rests on her head, symbolizing her imperial status. Her expression is serene, with a backdrop of a Roman villa garden.",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_40 = cliptextencode.encode(
            text="flat background", clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        loadimage_67 = loadimage.load_image(
            image="thrillisgon_A_realistic_portrait_of_a_Roman_empress_with_deli_a4d0b1a7-687d-4c27-b5fa-6c84b2728b1d_3.png"
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_68 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_67, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        applyinstantid = NODE_CLASS_MAPPINGS["ApplyInstantID"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        for q in range(1):
            applyinstantid_60 = applyinstantid.apply_instantid(
                weight=0.8,
                start_at=0,
                end_at=1,
                instantid=get_value_at_index(instantidmodelloader_11, 0),
                insightface=get_value_at_index(instantidfaceanalysis_38, 0),
                control_net=get_value_at_index(controlnetloader_16, 0),
                image=get_value_at_index(loadimage_13, 0),
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(cliptextencode_39, 0),
                negative=get_value_at_index(cliptextencode_40, 0),
                image_kps=get_value_at_index(loadimage_67, 0),
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=15,
                cfg=2,
                sampler_name="ddpm",
                scheduler="karras",
                denoise=0.9,
                model=get_value_at_index(applyinstantid_60, 0),
                positive=get_value_at_index(applyinstantid_60, 1),
                negative=get_value_at_index(applyinstantid_60, 2),
                latent_image=get_value_at_index(vaeencode_68, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )


if __name__ == "__main__":
    main()
