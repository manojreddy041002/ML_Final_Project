# Vision LLM MNIST ML Final Project

> This project compares MNIST digit recognition using a trained CNN versus zero-shot Vision-LLM inference via LLaVA on Ollama. The system classifies digits locally from a prompt and image, showing that while CNNs reach ~99% accuracy, Vision-LLMs provide flexible, human-readable, training-free predictions.


## 1. Set up environment

```bash
cd ML_Final_Project_organized
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the MNIST CNN model

```bash
python src/mnist_cnn.py
```

## 3. Run the Vision/LLM experiment

```bash
python src/mnist_vision_llm.py
```

## 4. Run comparison utility

```bash
python src/compare_any.py
```

## 5. Inputs & outputs

* MNIST data files → `data/`
* Raw digit images → `images/raw/`
* Generated outputs (plots, predictions) → `images/`
