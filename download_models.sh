
# Download the insightface model
#mkdir -p ./models/insightface/models/
#wget "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip?download=true" -O ./models/insightface/models/antelopev2.zip --content-disposition
#unzip ./models/insightface/models/antelopev2.zip -d ./models/insightface/models/
#
## Download the instantid model 
#mkdir -p ./models/instantid
#wget "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true" -O ./models/instantid/ip-adapter.bin --content-disposition

# Download the instantid controlnet
wget "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true" -O ./models/controlnet/diffusion_pytorch_model.safetensors --content-disposition

# download the realvis 3.0 turbo sdxl model
wget "https://civitai.com/api/download/models/272378?type=Model&format=SafeTensor&size=pruned&fp=fp16" -O ./models/checkpoints/realvisxlV50_v30TurboBakedvae.safetensors --content-disposition