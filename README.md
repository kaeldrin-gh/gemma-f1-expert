# Gemma F1 Expert 🏎️

A fine-tuned Formula 1 knowledge assistant based on Google's Gemma-3 model, trained on racing data from the Jolpica-F1 API and official press releases.

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![GPU](https://img.shields.io/badge/GPU-8--16GB-green.svg)

## 🎯 Features

- **Factual F1 Questions**: Race winners, lap records, driver statistics
- **Rule Explanations**: DRS, tyre strategies, technical regulations
- **Race Summaries**: Concise 2-3 sentence Grand Prix summaries
- **Live Data**: Real-time standings via Jolpica-F1 API
- **Chat Interface**: Streamlit web app for interactive queries

The model is trained on ~3,000 prompt-answer pairs built from the Jolpica-F1 API and official press releases.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (8-16GB VRAM) for training
- ~10 GB disk space
- Internet connection for data collection

### Installation

```bash
git clone https://github.com/yourusername/gemma-f1-expert.git
cd gemma-f1-expert

# Quick setup (recommended for beginners)
python setup.py

# Manual setup
pip install -r requirements.txt
```

### 1. Fetch and Prepare Data

```bash
# Fetch F1 data from Jolpica API (respects rate limits: 200 req/hour)
python data/fetch_jolpica.py

# Scrape press releases from FIA and team RSS feeds
python data/scrape_press.py

# Build training dataset (~3k Q-A pairs)
python data/build_dataset.py
```

### 2. Train the Model

```bash
# Fine-tune Gemma-3n with LoRA (takes ~30-45 minutes on 8GB GPU)
python src/train_lora.py
```

### 3. Test the Model

```bash
# CLI interface
python src/generate.py "Who won the 2023 Monaco Grand Prix?"

# Run evaluation
python src/evaluate.py

# Launch Streamlit web app
streamlit run src/webapp.py
```

## 📁 Project Structure

```
gemma-f1-expert/
├── data/                     # Data collection and processing
│   ├── fetch_jolpica.py     # Jolpica-F1 API client
│   ├── scrape_press.py      # RSS feed scraper
│   └── build_dataset.py     # Dataset builder
├── src/                      # Core implementation
│   ├── prepare_dataset.py   # Data formatting utilities
│   ├── train_lora.py        # LoRA fine-tuning script
│   ├── evaluate.py          # Model evaluation
│   ├── generate.py          # CLI question answering
│   └── webapp.py            # Streamlit web interface
├── notebooks/               # Jupyter notebooks
│   └── 00_train.ipynb      # End-to-end Colab tutorial
├── tests/                   # Unit tests
│   └── test_generate.py    # Generation tests
├── models/                  # Model storage (gitignored)
└── requirements.txt         # Python dependencies
```

## 🤖 Model Details

- **Base Model**: `google/gemma-3n` (quantized to 4-bit)
- **Fine-tuning**: LoRA with rank=4, alpha=16, dropout=0.05
- **Training**: 2 epochs, learning rate=2e-4, batch size=8
- **Context Length**: 256 tokens
- **Memory**: Fits comfortably on 8-16 GB GPU

## 📊 Data Sources

### Jolpica-F1 API
- **Base URL**: `https://api.jolpi.ca/ergast/f1`
- **Rate Limit**: 200 requests/hour (unauthenticated)
- **Coverage**: Seasons 2000-present
- **Data**: Race results, fastest laps, driver/constructor standings

### Press Releases
- FIA official communications
- Team newsrooms (last 2 seasons)
- Race weekend summaries

## 🔧 Development

### Run Tests
```bash
pytest -v
```

### Code Quality
```bash
ruff check .
ruff format .
```

### Data Refresh
The GitHub Action automatically refreshes data weekly using the Jolpica API.

## 📝 Example Usage

### CLI Examples
```bash
# Factual questions
python src/generate.py "Who holds the fastest lap record at Silverstone?"

# Rule explanations
python src/generate.py "How does DRS work in Formula 1?"

# Race summaries
python src/generate.py "Summarize the 2023 Abu Dhabi Grand Prix"
```

### Web Interface
Launch the Streamlit app for an interactive chat experience:
```bash
streamlit run src/webapp.py
```

Features:
- Real-time question answering
- Live standings from Jolpica API
- Formatted responses with markdown
- Chat history

## 🎮 Demo

![F1 Expert Demo](demo.gif)

## ⚠️ Rate Limits

The Jolpica-F1 API has a rate limit of 200 requests per hour for unauthenticated users. Our scripts automatically respect this limit with appropriate delays (`time.sleep(0.2)` between requests).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Jolpica-F1 API](https://jolpi.ca/) for providing comprehensive F1 data
- [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA training
- [Google](https://ai.google.dev/gemma) for the Gemma model family
- The Formula 1 community for inspiration

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{gemma_f1_expert,
  title={Gemma F1 Expert: Formula 1 Knowledge Assistant},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gemma-f1-expert}
}
```
