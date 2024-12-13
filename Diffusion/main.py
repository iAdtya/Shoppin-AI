import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from PIL import Image
import numpy as np

# Diffusion Model Feature Extractor
class DiffusionFeatureExtractor:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def extract_features(self, image_path, transform):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze(0).cpu().numpy()

# Cosine Similarity Search Engine
class CosineSimilaritySearchEngine:
    def __init__(self):
        self.feature_db = {}

    def add_image(self, image_id, feature):
        self.feature_db[image_id] = feature

    def search(self, query_feature, top_k=5):
        ids, features = zip(*self.feature_db.items())
        features = np.array(features)

        # Normalize features
        query_norm = query_feature / np.linalg.norm(query_feature)
        db_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

        similarities = np.dot(db_norm, query_norm)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(ids[idx], similarities[idx]) for idx in top_indices]
        return results

# Mock Diffusion Model
class MockDiffusionModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(MockDiffusionModel, self).__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        x = self.fc(x)
        return x

# Main
if __name__ == "__main__":
    # Setup
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to a consistent size
        transforms.ToTensor(),       # Convert to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the data
    ])

    output_dim = 128  # Output feature dimension
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model = MockDiffusionModel(output_dim=output_dim).to(device)
    feature_extractor = DiffusionFeatureExtractor(model=diffusion_model, device=device)

    search_engine = CosineSimilaritySearchEngine()

    # Feature Extraction and Indexing
    image_paths = {"image1": "logo.png", "image2": "pdfgpt.png", "image3": "Gaussian.png"}

    for image_id, path in image_paths.items():
        feature = feature_extractor.extract_features(path, transform)
        if feature is not None:
            search_engine.add_image(image_id, feature)

    # Query
    query_feature = feature_extractor.extract_features("logo.png", transform)
    if query_feature is not None:
        results = search_engine.search(query_feature, top_k=3)

        print("Search Results:")
        for image_id, similarity in results:
            print(f"Image ID: {image_id}, Similarity: {similarity:.4f}")
    else:
        print("Failed to extract query features.")
