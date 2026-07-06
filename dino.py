from pathlib import Path

import numpy as np


DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def get_dino_embedding(
    image,
    model_name=DEFAULT_DINO_MODEL,
    device=None,
    normalize=True,
):
    """Return a DINOv3 embedding for an image.

    image can be a file path, URL, PIL image, or numpy array.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from transformers.image_utils import load_image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = getattr(get_dino_embedding, "_cache", {})
    cache_key = (model_name, device)
    if cache_key not in cache:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        cache[cache_key] = (processor, model)
        get_dino_embedding._cache = cache

    processor, model = cache[cache_key]

    if isinstance(image, (str, Path)):
        image = load_image(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    embedding = getattr(outputs, "pooler_output", None)
    if embedding is None:
        embedding = outputs.last_hidden_state[:, 0, :]

    if normalize:
        embedding = F.normalize(embedding, p=2, dim=-1)

    return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
