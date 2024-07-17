import json
from typing import Any, Dict, List
import os

import torch
from tqdm import tqdm
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain_core.embeddings import Embeddings

# from transform import image_transform_v2
import sys
sys.path.append('models')
from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip import image_transform, tokenize

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu" # 实际情况无GPU

class CNCLIPEmbeddings(BaseModel, Embeddings):
    """OpenCLIP Embeddings model."""

    model: Any
    preprocess: Any
    tokenizer: Any
    # Select model: https://github.com/mlfoundations/open_clip
    vision_model: str = "ViT-B-16"
    text_model: str = "RoBERTa-wwm-ext-base-chinese"
    checkpoint: str = "models/pretrained_weights/clip_cn_vit-b-16.pt" #"laion2b_s32b_b79k"
    precision: str = "amp"

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:

            # Fall back to class defaults if not provided
            vision_model = values.get("vision_model", cls.__fields__["vision_model"].default)
            text_model = values.get("text_model", cls.__fields__["text_model"].default)
            checkpoint = values.get("checkpoint", cls.__fields__["checkpoint"].default)
            precision = values.get("precision", cls.__fields__["precision"].default)

            # Initialize the model.
            vision_model_config_file = f"models/cn_clip/clip/model_configs/{vision_model.replace('/', '-')}.json"
            print('Loading clip vision model config from', vision_model_config_file)
            assert os.path.exists(vision_model_config_file)
            
            text_model_config_file = f"models/cn_clip/clip/model_configs/{text_model.replace('/', '-')}.json"
            print('Loading clip text model config from', text_model_config_file)
            assert os.path.exists(text_model_config_file)

            with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
                model_info = json.load(fv)
                if isinstance(model_info['vision_layers'], str):
                    model_info['vision_layers'] = eval(model_info['vision_layers'])        
                for k, v in json.load(ft).items():
                    model_info[k] = v

            model = CLIP(**model_info).to(device)
            convert_weights(model)

            #gpu = 0 实际应用无GPU
            #torch.cuda.set_device(gpu)
            if precision == "amp" or precision == "fp32":
                convert_models_to_fp32(model)
            #model.cuda(gpu)
            if precision == "fp16":
                convert_weights(model)

            # Resume from a checkpoint.
            print("Begin to load clip model checkpoint from {}.".format(checkpoint))
            assert os.path.exists(checkpoint), "The checkpoint file {} not exists!".format(checkpoint)
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(checkpoint, map_location='cpu')
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
            model.load_state_dict(sd)

            # preprocess
            preprocess_val = image_transform(image_size=224)
            preprocess = preprocess_val

            tokenizer = tokenize
            values["model"] = model
            values["preprocess"] = preprocess
            values["tokenizer"] = tokenizer

        except ImportError:
            raise ImportError(
                "Please ensure both open_clip and torch libraries are installed. "
                "pip install open_clip_torch torch"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = []
        for text in texts:
            # Tokenize the text
            tokenized_text = self.tokenizer(text).to(device)

            # Encode the text to get the embeddings
            embeddings_tensor = self.model.encode_text(tokenized_text)

            # Normalize the embeddings
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert normalized tensor to list and add to the text_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            text_features.append(embeddings_list)

        return text_features


    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")

        # Open images directly as PIL images
        pil_images = [_PILImage.open(uri) for uri in uris]

        image_features = []
        for pil_image in tqdm(pil_images, desc='嵌入进度'):
            # Preprocess the image for the model
            preprocessed_image = self.preprocess(pil_image).unsqueeze(0).to(device)

            # Encode the image to get the embeddings
            embeddings_tensor = self.model.encode_image(preprocessed_image)

            # Normalize the embeddings tensor
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert tensor to list and add to the image_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()

            image_features.append(embeddings_list)

        return image_features