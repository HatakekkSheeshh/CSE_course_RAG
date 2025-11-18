"""
Script to PROVE whether chunks in PCA clusters are actually similar.
Regenerates embeddings, performs PCA, identifies clusters, and shows actual content.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from models.embedding import Embedding

# Load metadata
metadata_path = Path("data/indices/Advanced_Programming/metadata.json")
print("="*80)
print("STEP 1: Loading metadata")
print("="*80)

with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata_map = json.load(f)

print(f"Loaded {len(metadata_map)} chunks")

# Extract chunk texts and IDs in consistent order
chunk_ids = sorted(metadata_map.keys())
chunk_texts = [metadata_map[chunk_id]['text'] for chunk_id in chunk_ids]

print(f"\nSample chunk lengths:")
for i in range(min(5, len(chunk_texts))):
    print(f"  Chunk {i}: {len(chunk_texts[i])} chars, {len(chunk_texts[i].split())} words")

# Regenerate embeddings
print("\n" + "="*80)
print("STEP 2: Regenerating embeddings (this may take a moment...)")
print("="*80)

embedding_model = Embedding()
batch_size = 32
all_embeddings = []

for i in range(0, len(chunk_texts), batch_size):
    batch = chunk_texts[i:i + batch_size]
    batch_embeddings = embedding_model.embed_batch(batch)
    all_embeddings.extend(batch_embeddings)
    if (i // batch_size + 1) % 10 == 0:
        print(f"  Processed {i + len(batch)}/{len(chunk_texts)} chunks...")

embeddings_array = np.array(all_embeddings, dtype='float32')
print(f"\nEmbeddings shape: {embeddings_array.shape}")
print(f"  → {embeddings_array.shape[0]} chunks, {embeddings_array.shape[1]} dimensions")

# Perform PCA
print("\n" + "="*80)
print("STEP 3: Performing PCA")
print("="*80)

pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings_array)

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by PC1: {explained_variance[0]:.2%}")
print(f"Explained variance by PC2: {explained_variance[1]:.2%}")
print(f"Total explained variance: {sum(explained_variance):.2%}")

# Identify the dense cluster (upper-right quadrant: PC1 > 0, PC2 > 0)
print("\n" + "="*80)
print("STEP 4: Identifying clusters")
print("="*80)

# Define cluster regions based on the image description
# Dense cluster: PC1 between 0.0 and 0.45, PC2 between 0.0 and 0.45
dense_cluster_mask = (embeddings_2d[:, 0] >= 0.0) & (embeddings_2d[:, 0] <= 0.45) & \
                     (embeddings_2d[:, 1] >= 0.0) & (embeddings_2d[:, 1] <= 0.45)

dense_cluster_indices = np.where(dense_cluster_mask)[0]
print(f"Dense cluster (upper-right): {len(dense_cluster_indices)} chunks")

# Smaller cluster around (-0.3, -0.1)
small_cluster_mask = (embeddings_2d[:, 0] >= -0.4) & (embeddings_2d[:, 0] <= -0.2) & \
                     (embeddings_2d[:, 1] >= -0.2) & (embeddings_2d[:, 1] <= 0.0)

small_cluster_indices = np.where(small_cluster_mask)[0]
print(f"Small cluster (~-0.3, -0.1): {len(small_cluster_indices)} chunks")

# Outlier around (-0.75, 0.25)
outlier_mask = (embeddings_2d[:, 0] <= -0.7) & (embeddings_2d[:, 0] >= -0.8) & \
               (embeddings_2d[:, 1] >= 0.2) & (embeddings_2d[:, 1] <= 0.3)

outlier_indices = np.where(outlier_mask)[0]
print(f"Outlier (~-0.75, 0.25): {len(outlier_indices)} chunks")

# Analyze what's in the dense cluster
print("\n" + "="*80)
print("STEP 5: PROOF - Analyzing dense cluster content")
print("="*80)

if len(dense_cluster_indices) > 0:
    print(f"\nShowing {min(10, len(dense_cluster_indices))} chunks from dense cluster:")
    print("-" * 80)
    
    # Get document types
    doc_types_in_cluster = {}
    for idx in dense_cluster_indices[:20]:  # Check first 20
        chunk_id = chunk_ids[idx]
        doc_type = metadata_map[chunk_id].get('metadata', {}).get('doc_type', 'unknown')
        doc_types_in_cluster[doc_type] = doc_types_in_cluster.get(doc_type, 0) + 1
    
    print(f"\nDocument type distribution in dense cluster (first 20):")
    for doc_type, count in sorted(doc_types_in_cluster.items(), key=lambda x: -x[1]):
        print(f"  {doc_type}: {count}")
    
    # Show actual chunk texts
    print(f"\n\nACTUAL CHUNK CONTENT FROM DENSE CLUSTER:")
    print("=" * 80)
    for i, idx in enumerate(dense_cluster_indices[:5]):  # Show first 5
        chunk_id = chunk_ids[idx]
        chunk_data = metadata_map[chunk_id]
        text = chunk_data['text']
        doc_type = chunk_data.get('metadata', {}).get('doc_type', 'unknown')
        pc1, pc2 = embeddings_2d[idx]
        
        print(f"\n--- Chunk {i+1} (Index {idx}, PC1={pc1:.3f}, PC2={pc2:.3f}) ---")
        print(f"ID: {chunk_id}")
        print(f"Type: {doc_type}")
        print(f"Text (first 500 chars):")
        print(text[:500] + "..." if len(text) > 500 else text)
        print()

# Compare with chunks from other regions
print("\n" + "="*80)
print("STEP 6: COMPARISON - Chunks from other regions")
print("="*80)

# Get some scattered points (left half, negative PC1)
scattered_mask = embeddings_2d[:, 0] < 0.0
scattered_indices = np.where(scattered_mask)[0]

if len(scattered_indices) > 0:
    print(f"\nShowing {min(3, len(scattered_indices))} chunks from scattered region (PC1 < 0):")
    print("-" * 80)
    for i, idx in enumerate(scattered_indices[:3]):
        chunk_id = chunk_ids[idx]
        chunk_data = metadata_map[chunk_id]
        text = chunk_data['text']
        doc_type = chunk_data.get('metadata', {}).get('doc_type', 'unknown')
        pc1, pc2 = embeddings_2d[idx]
        
        print(f"\n--- Scattered Chunk {i+1} (Index {idx}, PC1={pc1:.3f}, PC2={pc2:.3f}) ---")
        print(f"ID: {chunk_id}")
        print(f"Type: {doc_type}")
        print(f"Text (first 500 chars):")
        print(text[:500] + "..." if len(text) > 500 else text)
        print()

# Check outlier
if len(outlier_indices) > 0:
    print("\n" + "="*80)
    print("STEP 7: OUTLIER ANALYSIS")
    print("="*80)
    idx = outlier_indices[0]
    chunk_id = chunk_ids[idx]
    chunk_data = metadata_map[chunk_id]
    text = chunk_data['text']
    doc_type = chunk_data.get('metadata', {}).get('doc_type', 'unknown')
    pc1, pc2 = embeddings_2d[idx]
    
    print(f"\nOutlier chunk (PC1={pc1:.3f}, PC2={pc2:.3f}):")
    print(f"ID: {chunk_id}")
    print(f"Type: {doc_type}")
    print(f"Full text:")
    print(text)
    print()

# Calculate similarity within dense cluster
print("\n" + "="*80)
print("STEP 8: SIMILARITY METRICS")
print("="*80)

if len(dense_cluster_indices) >= 2:
    # Get embeddings for dense cluster
    dense_embeddings = embeddings_array[dense_cluster_indices]
    
    # Calculate cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(dense_embeddings)
    
    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(len(dense_embeddings), k=1)
    cluster_similarities = similarities[triu_indices]
    
    print(f"\nSimilarity within dense cluster:")
    print(f"  Mean cosine similarity: {np.mean(cluster_similarities):.4f}")
    print(f"  Std cosine similarity: {np.std(cluster_similarities):.4f}")
    print(f"  Min similarity: {np.min(cluster_similarities):.4f}")
    print(f"  Max similarity: {np.max(cluster_similarities):.4f}")
    
    # Compare with random pairs from entire dataset
    n_samples = min(100, len(embeddings_array))
    random_indices = np.random.choice(len(embeddings_array), n_samples, replace=False)
    random_embeddings = embeddings_array[random_indices]
    random_similarities = cosine_similarity(random_embeddings)
    random_triu = random_similarities[np.triu_indices(n_samples, k=1)]
    
    print(f"\nSimilarity across random chunks:")
    print(f"  Mean cosine similarity: {np.mean(random_triu):.4f}")
    print(f"  Std cosine similarity: {np.std(random_triu):.4f}")
    
    print(f"\n→ Dense cluster is {np.mean(cluster_similarities) - np.mean(random_triu):.4f} more similar on average")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("Each point in the PCA plot represents ONE CHUNK (not a token).")
print("Each chunk contains ~512 tokens worth of text.")
print(f"Total chunks visualized: {len(chunk_ids)}")
print("\nThe similarity analysis above shows whether chunks in the dense cluster")
print("are actually more similar to each other than random chunks.")





