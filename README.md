# Optimized-Architectures-for-Real-Time-Anomaly-Detection-via-Knowledge-Distillation
Implementation of a Hybrid Transformer-GAN for network anomaly detection. Serves as the high-performance "Teacher" model for knowledge distillation in edge-level security.

### 2. Optimized Model (Distill-Guard)
**GitHub Repository Description:**
> Edge-optimized cyber anomaly detection using Knowledge Distillation and INT8 Quantization. Optimized for real-time inference on Raspberry Pi.

**README.md File:**
```markdown
# Distill-Guard: Optimized Edge-Level Anomaly Detection

Distill-Guard is an optimized version of a Hybrid Transformer-GAN, developed for real-time deployment on resource-constrained edge devices like Raspberry Pi and IoT gateways.

## ⚡ Optimization Features
- **Knowledge Distillation:** Transfers "dark knowledge" from a large Transformer-GAN (Teacher) to a lightweight Student.
- **Linear Attention:** Replaces standard self-attention to reduce computational complexity from $O(n^2)$ to $O(n)$.
- **INT8 Quantization:** Quantization-Aware Training (QAT) to compress the model for 8-bit inference.

## 🛠️ Architecture
- **Student Model:** A 1-2 layer linearized Transformer.
- **Framework:** PyTorch & TorchAO.
- **Deployment:** ONNX Runtime for edge compatibility.

## 📦 Setup & Installation
1. **Environment:**
   ```bash
   conda create -n distill_guard python=3.10
   conda activate distill_guard
   pip install torch pandas scikit-learn torchao onnxruntime
