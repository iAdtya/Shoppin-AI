# Multiple Methods of Image Similarity Search

## Comparison

| Method             | Precision | Recall    | Retrieval Accuracy | Scalability |
| ------------------ | --------- | --------- | ------------------ | ----------- |
| Hybrid-Search      | Very High | Very High | Very High          | Very High   |
| Autoencoder        | Medium    | Medium    | Medium             | Medium      |
| Vision Transformer | High      | High      | High               | High        |
| KNN with ViT       | High      | High      | High               | Very High   |
| Diffusion Models   | Very High | Very High | Very High          | Low         |

## Run on Colab

```sh
!git clone https://github.com/iAdtya/Shoppin-AI
```

```sh
!pip install -r requirements.txt
```

```sh
!python Hybrid-Search\main.py
```

## 0. Best approach (Hybrid-Search)

- hybrid approach that leverages the strengths of diffusion models for feature extraction and a vector database for efficient similarity search.

## Advantages of Hybrid-Search Approach

- High precision and recall due to the combination of multiple methods.
- Great for real-time applications.
- Scalable and adaptable to different types of image retrieval tasks.

## 1. Autoencoder-Based Approach

- Use an autoencoder to encode input images into a latent space.
- During a query, compare the latent vector of the query image with those of stored images using distance metrics like cosine similarity or nucleus sampling, Euclidean distance.

### Advantages of Autoencoder-Based Approach

- Simple and unsupervised.
- Effective for dimensionality reduction.

---

## 2. Vision Transformer (ViT)-Based Approach

- Use a Vision Transformer (ViT) to encode input images into sequence-based embeddings.
- Utilize self-attention mechanisms to capture global image relationships.

### Advantages of ViT-Based Approach

- Captures global context effectively.
- Scalable for large and diverse datasets.
- Works well for highly complex or structured datasets.

---

## 3. KNN-Based Approach with ViT

- Use a Vision Transformer (ViT) to encode input images into embeddings.
- Build a K-Nearest Neighbors (KNN) index for the embeddings.
- Perform similarity search using KNN to find the closest embeddings.

### Advantages of KNN-Based Approach with ViT

- Efficient for large datasets with pre-built indices.
- Can handle dynamic updates to the dataset.

---

## 4. Diffusion Model-Based Approach

- Use a diffusion model to extract features from images.
- Compare the features using cosine similarity to find similar images.

### Advantages of Diffusion Model-Based Approach

- High precision and recall.
- Effective for complex image representations.
- Ideal for high-accuracy.

---
