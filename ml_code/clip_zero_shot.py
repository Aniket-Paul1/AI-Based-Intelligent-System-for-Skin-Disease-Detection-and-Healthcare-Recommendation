import torch
import clip
from PIL import Image

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load CLIP ----------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ---------------- Disease Space ----------------
# EXACT disease names (mentor requirement)
ZERO_SHOT_DISEASES = [
    "Acne",
    "Rosacea",
    "Herpes",
    "Dermatitis",
    "Psoriasis",
    "Vitiligo",
    "Basal Cell Carcinoma",
    "Melanoma",
]

# ---------------- FACE-AWARE PROMPTS ----------------
# This is the MOST IMPORTANT improvement
text_prompts = [
    "a close-up facial skin photo showing acne",
    "a close-up facial skin photo showing rosacea",
    "a close-up facial skin photo showing herpes skin infection",
    "a close-up facial skin photo showing dermatitis rash",
    "a close-up skin photo showing psoriasis plaques",
    "a close-up skin photo showing vitiligo depigmentation",
    "a close-up skin lesion photo showing basal cell carcinoma",
    "a close-up skin lesion photo showing melanoma",
]

text_tokens = clip.tokenize(text_prompts).to(device)


@torch.no_grad()
def clip_zero_shot_predict(image: Image.Image):
    """
    Always returns the MOST visually similar disease.
    """

    image_input = preprocess(image).unsqueeze(0).to(device)

    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)
    probs = similarity.softmax(dim=0).cpu().numpy()

    results = {
        ZERO_SHOT_DISEASES[i]: float(probs[i])
        for i in range(len(ZERO_SHOT_DISEASES))
    }

    # Sort by confidence
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1], reverse=True)
    )

    best_label = next(iter(sorted_results))
    best_conf = sorted_results[best_label]

    return best_label, best_conf, sorted_results
