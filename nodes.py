import torch
import random
from tqdm import tqdm
from comfy.model_base import BaseModel
from comfy.sd import load_checkpoint_guess_config
from server import PromptServer

MEDIUMS = [
    "painting", "drawing", "photograph", "HD photo", "illustration", "portrait",
    "sketch", "3d render", "digital painting", "concept art", "screenshot",
    "canvas painting", "watercolor art", "print", "mosaic", "sculpture",
    "cartoon", "comic art", "anime",
]

SUBJECTS = [
    "dog", "cat", "horse", "cow", "pig", "sheep", "lion", "elephant", "monkey",
    "bird", "chicken", "eagle", "parrot", "penguin", "fish", "shark", "dolphin",
    "whale", "octopus", "bee", "butterfly", "ant", "ladybug", "person", "man",
    "woman", "child", "baby", "boy", "girl", "car", "boat", "airplane", "bicycle",
    "motorcycle", "train", "building", "house", "bridge", "castle", "temple",
    "monument", "tree", "flower", "mountain", "lake", "river", "ocean", "beach",
    "fruit", "vegetable", "meat", "bread", "cake", "soup", "coffee", "toy", "book",
    "phone", "computer", "TV", "camera", "musical instrument", "furniture", "road",
    "park", "garden", "forest", "city", "sunset", "clouds",
]

class CLIPSliderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "target_word": ("STRING", {"default": "happy"}),
                "opposite": ("STRING", {"default": "sad"}),
                "scales": ("STRING", {"default": "1.0"}),
                "prompt": ("STRING", {"default": "a photo of a person"}),
                "iterations": ("INT", {"default": 300, "min": 1, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "target_word_2nd": ("STRING", {"default": ""}),
                "opposite_2nd": ("STRING", {"default": ""}),
                "scales_2nd": ("STRING", {"default": "0.0"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_clip_slider"
    CATEGORY = "conditioning"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_latent_direction(self, clip, target_word, opposite, iterations=300):
        with torch.no_grad():
            positives = []
            negatives = []
            for _ in tqdm(range(iterations)):
                medium = random.choice(MEDIUMS)
                subject = random.choice(SUBJECTS)
                pos_prompt = f"a {medium} of a {target_word} {subject}"
                neg_prompt = f"a {medium} of a {opposite} {subject}"
                pos_toks = clip.tokenize(pos_prompt)
                neg_toks = clip.tokenize(neg_prompt)
                pos = clip.encode_from_tokens(pos_toks)
                neg = clip.encode_from_tokens(neg_toks)
                positives.append(pos)
                negatives.append(neg)

        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        diffs = positives - negatives
        avg_diff = diffs.mean(0, keepdim=True)
        return avg_diff

    def apply_clip_slider(self, model, clip, target_word, opposite, scales, prompt, iterations, seed,
                          target_word_2nd="", opposite_2nd="", scales_2nd="0.0"):
        torch.manual_seed(seed)

        avg_diff = self.find_latent_direction(clip, target_word, opposite, iterations)
        avg_diff_2nd = None
        if target_word_2nd and opposite_2nd:
            avg_diff_2nd = self.find_latent_direction(clip, target_word_2nd, opposite_2nd, iterations)

        # Convert scales from string to list of floats
        scales = [float(s.strip()) for s in scales.split(',')]
        scales_2nd = [float(s.strip()) for s in scales_2nd.split(',')]

        # Ensure scales_2nd has the same length as scales
        scales_2nd = scales_2nd + [0.0] * (len(scales) - len(scales_2nd))

        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        positive_conditionings = []
        negative_conditionings = []

        for scale, scale_2nd in zip(scales, scales_2nd):
            # Apply the CLIP slider effect for positive conditioning
            positive_cond = cond + avg_diff * scale
            if avg_diff_2nd is not None:
                positive_cond = positive_cond + avg_diff_2nd * scale_2nd

            # Apply the inverse CLIP slider effect for negative conditioning
            negative_cond = cond - avg_diff * scale
            if avg_diff_2nd is not None:
                negative_cond = negative_cond - avg_diff_2nd * scale_2nd

            positive_conditionings.append([positive_cond, {"pooled_output": pooled}])
            negative_conditionings.append([negative_cond, {"pooled_output": pooled}])

        return (positive_conditionings, negative_conditionings)

NODE_CLASS_MAPPINGS = {
    "CLIPSlider": CLIPSliderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSlider": "CLIP Slider"
}
