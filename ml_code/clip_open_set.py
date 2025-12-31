import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

TEXT_PROMPTS = [
    "a photo of human skin",
    "a close-up photo of a skin condition",
    "a dermatology image",
    "a medical photo of skin disease",
    "a non-skin image",
    "an object or scenery photo",
]

text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)


@torch.no_grad()
def clip_open_set_score(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)

    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)
    probs = similarity.softmax(dim=0).cpu().numpy()

    return dict(zip(TEXT_PROMPTS, probs))


def is_in_distribution(scores):
    skin_scores = [
        scores["a photo of human skin"],
        scores["a close-up photo of a skin condition"],
        scores["a dermatology image"],
        scores["a medical photo of skin disease"],
    ]

    non_skin_scores = [
        scores["a non-skin image"],
        scores["an object or scenery photo"],
    ]

    # Relative comparison (THIS IS THE KEY FIX)
    return max(skin_scores) > max(non_skin_scores)
