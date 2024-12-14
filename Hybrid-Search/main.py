import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid


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


# Mock Diffusion Model
class MockDiffusionModel(torch.nn.Module):
    def __init__(self, output_dim):
        super(MockDiffusionModel, self).__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Setup Qdrant client
client = QdrantClient(
    url="https://5ae5348b-98a2-4b84-9b6d-0920ccd86376.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="6MB3dXeiBwjVMF9nQGMBQyB4jWWTkHjiO_i88pmLKDfZG0g3HTuasA",
)

# client.create_collection(
#     collection_name="shoppin",
#     vectors_config=VectorParams(size=128, distance=Distance.COSINE),
# )

if __name__ == "__main__":
    # Setup
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    output_dim = 128  # Output feature dimension
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model = MockDiffusionModel(output_dim=output_dim).to(device)
    feature_extractor = DiffusionFeatureExtractor(model=diffusion_model, device=device)

    image_paths = {
        "image1": "logo.png",
        "image2": "pdfgpt.png",
        "image3": "Gaussian.png",
    }

    for image_id, path in image_paths.items():
        feature = feature_extractor.extract_features(path, transform)
        if feature is not None:
            client.upsert(
                collection_name="shoppin",
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=feature.tolist(),
                        payload={"image_id": image_id},
                    )
                ],
            )

    query_feature = feature_extractor.extract_features("logo.png", transform)
    if query_feature is not None:
        hits = client.search(
            collection_name="shoppin",
            query_vector=query_feature.tolist(),
            limit=3,
        )

        print("Search Results:")
        for hit in hits:
            if hit.payload is not None:
                print(
                    f"Image ID: {hit.payload['image_id']}, Similarity: {hit.score:.4f}"
                )
        else:
            print("Hit payload is None.")

    else:
        print("Failed to extract query features.")
