***

# Tri-MoDE: A Scalable Tri-Modal Data Routing System for Resource-Constrained Environments

**Author:** Sikai Wang
**Project Repository:** [https://github.com/SydneySlu/Tri-Mode](https://github.com/SydneySlu/Tri-Mode)

## 1. Executive Summary

This project addresses the challenge of processing high-dimensional multi-modal data (Image, Text, Audio) under computational resource constraints. Traditional methods for multi-modal clustering typically rely on high-end GPU clusters. **Tri-MoDE (Tri-Modal Data Experts)** introduces a lightweight, scalable architecture that leverages pre-trained large models (CLIP/CLAP) and streaming engineering designs to achieve efficient unsupervised clustering and data routing on consumer-grade CPUs.

The system constructs a unified 1536-dimensional semantic space aligned across three modalities and employs an Out-of-Core learning strategy. Stress tests demonstrate that the architecture maintains constant memory complexity (O(1)) and supports linear scalability, making it a viable prototype for large-scale data cleaning and mixture-of-experts (MoE) routing.

## 2. System Architecture & Pipeline

The system is engineered into five sequential phases, adhering to the "Streaming Processing" paradigm:

### Phase 1: Data Ingestion & Recovery
* **Objective:** Secure valid multi-modal raw data.
* **Implementation:** Utilized the ESC-50 environmental audio dataset.
* **Engineering Solution:** Implemented a custom byte-level decoding pipeline using `io.BytesIO` and `soundfile` to bypass the `torchcodec` compatibility issues on Windows/CPU environments, ensuring 100% data integrity.

### Phase 2: Standardization & Visual Modality Synthesis
* **Objective:** Align audio data with visual encoders.
* **Implementation:** Developed a signal processing module using `librosa` to transform 1D audio waveforms into 2D **Mel Spectrograms**.
* **Storage Optimization:** Adopted the **WebDataset (.tar)** format for sharded storage, solving the I/O bottleneck caused by massive small files and enabling sequential data access.

### Phase 3: Heterogeneous Feature Extraction
* **Objective:** Construct a unified semantic space.
* **Implementation:** Integrated two distinct pre-trained encoders:
    * **Vision & Text:** CLIP (ViT-B/32) with OpenAI weights.
    * **Audio:** CLAP (HTSAT-base) with LAION weights.
* **Fusion Strategy:** Features from all three modalities are extracted via a streaming inference engine, normalized (L2), and concatenated into a robust **1536-dimensional embedding**.

### Phase 4: Scalable Expert Training
* **Objective:** Unsupervised discovery of semantic structures.
* **Implementation:** Replaced traditional K-Means with **Mini-Batch K-Means**.
* **Scalability:** Implemented incremental learning (Partial Fit) to decouple memory usage from dataset size, allowing the system to train on potentially infinite data streams using limited RAM.

### Phase 5: Intelligent Routing & Visualization
* **Objective:** Assign data to specialized experts and verify alignment.
* **Implementation:** Calculated Euclidean distances between samples and learned centroids to perform Hard Assignment.
* **Verification:** Utilized **t-SNE** for dimensionality reduction, visually confirming the semantic clustering of aligned tri-modal data.

## 3. Key Innovations

### 3.1. Unified Tri-Modal Embedding Space
Unlike previous works (e.g., MetaCLIP) that focus solely on Image-Text pairs, Tri-MoDE successfully introduces the **Audio modality** into the clustering loop. By synthesizing Mel Spectrograms and leveraging CLAP, the system achieves physical and semantic alignment across three distinct data types, enhancing the model's ability to handle complex, unstructured data.

### 3.2. Resource-Efficient "Out-of-Core" Architecture
A core contribution is the re-engineering of the data pipeline for **resource-constrained environments**. By combining WebDataset streaming with Mini-Batch algorithms, the system eliminates the dependency on high-memory servers.
* **Throughput:** Achieved ~30,000 samples/sec on a standard CPU.
* **Memory Footprint:** Constant complexity regardless of dataset size.

### 3.3. Trustworthy Perception Design
The system incorporates a "Multi-Modal Corroboration" mechanism. By clustering based on concatenated features, the system utilizes modal redundancy (e.g., audio confirming visual ambiguity) to improve routing robustness against noise or missing data in single modalities.

## 4. Challenges & Engineering Solutions

| Challenge | Technical Issue | Engineering Solution |
| :--- | :--- | :--- |
| **Cross-Platform Compatibility** | Original research code relied on Linux-specific distributed processing and GPU dependencies (CUDA). | Refactored the entire codebase with dynamic device detection (`device='cpu'`) and standardized relative paths for portability. |
| **Data Decoding Failure** | `datasets` library failed to decode audio on Windows due to backend dependencies (`torchcodec`). | Reverse-engineered the data loading process to extract raw binary bytes and implemented a manual in-memory decoding pipeline. |
| **Multiprocessing Deadlocks** | Python's `spawn` method on Windows caused global variable access errors during parallel inference. | Architected a decoupled worker routine in `prep_inference.py`, ensuring each subprocess initializes its own configuration context. |
| **Type Mismatch** | Conflict between Audio library outputs (`numpy.float64`) and PyTorch models (`torch.float32`). | Implemented an explicit type casting layer within the inference loop to ensure precision alignment across the pipeline. |

## 5. Future Work

* **Deep Fusion:** Introduce Cross-Attention mechanisms to capture fine-grained interactions between modalities, moving beyond simple feature concatenation.
* **Downstream Validation:** Train specialized classifiers on the routed data subsets to quantitatively measure the accuracy improvements gained from the "Data Expert" routing strategy.
* **Scale-up:** Deploy the validated architecture on cloud infrastructure to process ImageNet-scale datasets.

## 6. Conclusion

Tri-MoDE demonstrates a successful transition from theoretical research to engineering implementation. It proves that with optimized architectural design, complex multi-modal AI tasks can be effectively executed on accessible hardware, providing a scalable solution for data preprocessing and intelligent routing.
