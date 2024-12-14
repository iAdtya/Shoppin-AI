import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Define the Autoencoder (from your provided code)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Define the Autoencoder-Based Image Search Engine
class AutoencoderImageSearchEngine:
    def __init__(self, autoencoder, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.autoencoder = autoencoder.to(self.device)
        self.autoencoder.eval()  # Set model to evaluation mode
        self.image_embeddings = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for the Autoencoder model.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def encode_image(self, image_path):
        """
        Encode an image into an embedding using the Autoencoder model.
        """
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            encoded, _ = self.autoencoder(image_tensor)
        # Flatten the encoded output
        return encoded.view(encoded.size(0), -1).cpu().numpy()

    def add_image(self, image_id, image_path):
        """
        Add an image to the search engine by encoding and storing its embedding.
        """
        embedding = self.encode_image(image_path)
        self.image_embeddings[image_id] = embedding

    def search_similar(self, query_image_path, top_k=5):
        """
        Search for similar images given a query image.
        """
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
    # Load a pre-trained Autoencoder or initialize a new one
    autoencoder = Autoencoder()

    search_engine = AutoencoderImageSearchEngine(autoencoder)

    # Add images to the search engine
    search_engine.add_image("image1", "logo.png")
    search_engine.add_image("image2", "pdfgpt.png")
    search_engine.add_image("image3", "Gaussian.png")

    # Perform a search
    results = search_engine.search_similar("logo.png")

    print("Search Results:")
    for image_id, similarity in results:
        print(f"Image ID: {image_id}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
