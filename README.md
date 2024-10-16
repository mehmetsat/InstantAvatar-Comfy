# 🌟 InstantID Setup Guide

## 🔍 Overview

### 🖥️ Gradio UI with ComfyUI Backend
<span style="color: #4a90e2;">InstantID utilizes a Gradio interface for its user interaction, while leveraging the ComfyUI backend for processing. This combination provides a user-friendly front-end with the powerful processing capabilities of ComfyUI.</span>

ComfyUI is chosen for several reasons:
1. 🛠️ **Versatility**: It offers a highly flexible environment for development, allowing for rapid prototyping and experimentation.
2. 💪 **Power**: As a backend, ComfyUI provides robust processing capabilities, making it suitable for production use.
3. 🎨 **Customization**: ComfyUI allows for fine-tuned control over the generation process, enabling improvements and optimizations.

### 🎭 InstantID
<span style="color: #50c878;">InstantID is a tool that allows for instant identity-preserving generation and editing. It enables users to create or modify images while maintaining the identity of the subject.</span>

### 🚀 Improvements and Optimizations
This implementation includes several enhancements over the default InstantID:

1. 🌈 **Noise Injection**: The default InstantID implementation can sometimes produce overly dark or "burned" images. By injecting noise to the negative embeds, this effect is mitigated, resulting in brighter images and improved facial likenesses. The Apply InstantID node automatically injects 35% noise to achieve this effect.

2. 🖼️ **Init Image Support**: To address the issue of low-quality image generation in ComfyUI's InstantID extension, support for an init image has been added. This init image guides the denoising process at 0.8-0.9 strength, resulting in higher quality output images.

3. 🧠 **Model Integration**: The implementation uses high-quality open or closed models to generate template input images, which then guide the SDXL model. This approach combines the strengths of multiple models to produce superior results. The template images can be generated using Midjourney, Flux1.1 pro or other highend models once and then used as init image here.

### 🎓 Model Choice
<span style="color: #ff6b6b;">For this demo, the realvis 3.0 turbo model is used. This model excels at generating realistic images and scenes. While it's an excellent choice for realistic outputs, it can be replaced with alternative models if different stylistic results are desired.</span>

## 📦 Installation and Setup

### 📥 Clone the Repository
To get started with InstantID, first clone the repository including all submodules:

```bash
git clone --recursive https://github.com/InstantID/InstantID.git
cd InstantID
```

### 🐍 Create and Activate Conda Environment
Create a new Conda environment named "instant_lora" and activate it:

```bash
conda create -n instant_lora python=3.10
conda activate instant_lora
```

### 📚 Install Requirements
Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 🔽 Download Models
Run the script to download required models:

```bash
bash download_models.sh
```

## 🏃‍♂️ Running InstantID

To launch the InstantID user interface, run:

```bash
python instantid_ui.py
```

This will start the Gradio interface, allowing you to interact with InstantID through a web browser.

## 🎮 Using the Gradio UI

<span style="color: #ffa500;">Once you've launched the InstantID UI, you'll see a user-friendly interface with several input fields and options. Here's how to use each component:</span>

1. 📸 **Input Image**: Upload an image of the person whose identity you want to preserve.

2. 🕺 **Pose Image**: Upload an image that represents the pose or composition you want for the output image.

3. 🎨 **Init Image**: Upload an initial image for the img2img process. This serves as a starting point for the generation and helps improve output quality.

4. ✅ **Positive Prompt**: Enter a description of what you want to see in the generated image. Be as detailed as possible for better results.

5. ❌ **Negative Prompt**: Enter descriptions of elements you don't want to see in the generated image. This helps refine the output.

6. 🎲 **Seed**: Enter a number for reproducible results, or leave at -1 for a random seed.

7. 🔢 **Steps**: Adjust the number of denoising steps. More steps can lead to better quality but take longer.

8. 🎚️ **CFG Scale**: Adjust the Classifier Free Guidance scale. Higher values adhere more strictly to your prompt.

9. 🧮 **Sampler Name**: Choose the sampling method. Currently, only "ddpm" is available.

10. ⏱️ **Scheduler**: Choose the scheduler for the sampling process. Currently, only "karras" is available.

11. 🖱️ **Generate Image**: Click this button to start the image generation process.

12. 🖼️ **Output Image**: The generated image will appear here once the process is complete.

💡 Tips for best results:
- Ensure your input image clearly shows the person's face.
- Choose a pose image that matches the style and composition you want.
- Provide a high-quality init image to guide the generation process.
- Experiment with different prompts and settings to achieve your desired output.
- If you're not satisfied with the result, try adjusting the CFG scale or number of steps, or use a different seed.
- Remember that the realvis 3.0 turbo model excels at realistic scenes. If you need a different style, consider using an alternative model.

## 🔜 Next Steps

<span style="color: #9370db;">After setting up and familiarizing yourself with the UI, you can start experimenting with different inputs and settings to generate and edit images. Refer to the project's documentation for more advanced usage instructions and tips on getting the best results.</span>

## 📜 License

MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.