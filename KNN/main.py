import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from sklearn.neighbors import NearestNeighbors
import numpy as np


# Define the KNN-based Image Search Engine with ViT
class KNNImageSearchEngine:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.image_embeddings = []
        self.image_ids = []
        self.knn = None

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def encode_image(self, image_path):
        inputs = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def add_image(self, image_id, image_path):
        embedding = self.encode_image(image_path)
        self.image_embeddings.append(embedding)
        self.image_ids.append(image_id)

    def build_index(self):
        if not self.image_embeddings:
            raise ValueError("No images have been added to the search engine.")
        embeddings_matrix = np.vstack(self.image_embeddings)
        self.knn = NearestNeighbors(
            n_neighbors=min(5, len(self.image_embeddings)), metric="cosine"
        ).fit(embeddings_matrix)

    def search_similar(self, query_image_path, top_k=1):
        if not self.knn:
            raise ValueError(
                "KNN index has not been built. Call build_index() after adding images."
            )

        query_embedding = self.encode_image(query_image_path)
        distances, indices = self.knn.kneighbors(query_embedding, n_neighbors=top_k)

        results = [
            (self.image_ids[idx], 1 - distances[0][i])
            for i, idx in enumerate(indices[0])
        ]
        return results


# Example Usage
def main():
    search_engine = KNNImageSearchEngine()

    search_engine.add_image("image1", "logo.png")
    search_engine.add_image("image2", "pdfgpt.png")
    search_engine.add_image("image3", "Gaussian.png")

    search_engine.build_index()

    results = search_engine.search_similar("logo.png")

    print("Search Results:")
    for image_id, similarity in results:
        print(f"Image ID: {image_id}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
