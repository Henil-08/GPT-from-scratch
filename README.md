# GPT-from-scratch

This project implements a GPT-style transformer model from scratch using PyTorch for English-to-Italian translation, trained on the OPUS Books dataset.

## Features

- Custom transformer implementation (encoder-decoder architecture)
- Tokenizer training using WordLevel tokenization
- End-to-end ML pipeline: data preprocessing → training → evaluation
- Metrics: BLEU, Word Error Rate (WER), Character Error Rate (CER)
- Optimized decoding with greedy and beam search
- Real-time inference via Streamlit application
- Containerized deployment with Docker
- GPU-backed inference on AWS EC2

## Model Details

- Dataset: OPUS Books (EN–IT translation)
- Sequence Length: 350
- Model Dimension: 512
- Training: Multi-epoch GPU training on Google Colab
- Evaluation Metrics:
  - BLEU Score
  - WER (Word Error Rate)
  - CER (Character Error Rate)

## Training and Inference

Training is performed on Google Colab using GPU acceleration via notebook `train.ipynb` 

For inferenc, run streamlit app using:
```bash
streamlit run app.py
```

You can also perform inference via notebook `notebooks/inference.ipynb`

## Deployment

### Docker

Build Image:
```bash
docker build -t transformer-app .
```

Run container:
```bash
docker run -p 8501:8501 transformer-app
```

### AWS

- Launch GPU-enabled EC2 instance
- Install Docker
- Run container

```bash
docker run -d -p 8501:8501 transformer-app
```