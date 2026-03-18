import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_model(model_name: str, device: str):
    print(f"[CLIP] Loading model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    print(f"[CLIP] Model loaded on {device}.")
    return processor, model


def compute_similarity(image_path: str, texts: list[str], processor, model, device: str):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        # CLIP 原始输出 logits，已经包含 learnable logit_scale
        logits_per_image = outputs.logits_per_image  # shape: [1, num_texts]
        probs = logits_per_image.softmax(dim=1)[0]

        # 也手动算一份 cosine similarity，便于你观察
        image_embeds = outputs.image_embeds  # [1, D]
        text_embeds = outputs.text_embeds    # [N, D]

        image_embeds = image_embeds#  / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds#  / text_embeds.norm(dim=-1, keepdim=True)

        cosine_sims = (image_embeds @ text_embeds.T)[0]  # [N]

    return cosine_sims.detach().cpu(), probs.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP similarity between one image and multiple texts.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        required=True,
        help='Candidate texts, e.g. --texts "a dog" "a cat" "a car"',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processor, model = load_model(args.model, args.device)
    cosine_sims, probs = compute_similarity(
        str(image_path), args.texts, processor, model, args.device
    )

    print("\n=== Results ===")
    for text, sim, prob in zip(args.texts, cosine_sims.tolist(), probs.tolist()):
        print(f"Text: {text}")
        print(f"  Cosine similarity: {sim:.6f}")
        print(f"  Softmax probability: {prob:.6f}")

    best_idx = torch.argmax(probs).item()
    print("\n=== Best Match ===")
    print(f"Text: {args.texts[best_idx]}")
    print(f"Cosine similarity: {cosine_sims[best_idx].item():.6f}")
    print(f"Softmax probability: {probs[best_idx].item():.6f}")


if __name__ == "__main__":
    main()