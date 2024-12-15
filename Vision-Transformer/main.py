import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Define the Vision Transformer-based Image Search Engine
class ViTImageSearchEngine:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.image_embeddings = {}

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def encode_image(self, image_path):
        inputs = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token representation as the image embedding
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def add_image(self, image_id, image_path):
        embedding = self.encode_image(image_path)
        self.image_embeddings[image_id] = embedding

    def search_similar(self, query_image_path, top_k=5):
        query_embedding = self.encode_image(query_image_path)
        similarities = {}

        for image_id, embedding in self.image_embeddings.items():
            similarity = cosine_similarity(query_embedding, embedding).flatten()
            similarities[image_id] = similarity[0]

        # Sort by similarity in descending order and return top_k results
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# Example Usage
def main():
    search_engine = ViTImageSearchEngine()

    # Add images to the search engine
    search_engine.add_image("image1", "logo.png")
    search_engine.add_image("image2", "pdfgpt.png")
    search_engine.add_image("image3", "Gaussian.png")

    # Perform a search
    results = search_engine.search_similar("pdfgpt.png")

    print("Search Results:")
    for image_id, similarity in results:
        print(f"Image ID: {image_id}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
