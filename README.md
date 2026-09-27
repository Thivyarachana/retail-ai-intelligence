# Retail AI Intelligence

A real-world retail data project for applied learning.

## Features
- Executive sales dashboard
- Interactive sales explorer
- Customer RFM + K-Means segmentation
- Random Forest demand prediction
- AI chatbot using Gemini when `GEMINI_API_KEY` is configured
- Fallback project-aware chatbot without an API key
- Automatic download of the UCI Online Retail dataset

## Run in Google Colab

```bash
pip install -r requirements.txt
```

Optional Gemini key:
```python
import os
os.environ["GEMINI_API_KEY"] = "YOUR_KEY"
```

Run:
```bash
streamlit run app.py &>/content/streamlit.log &
```

For a public Colab URL, use a tunnel such as Cloudflare Tunnel or localtunnel.

## Project title
Retail AI Intelligence: Real-World Sales Analytics, Customer Segmentation & Demand Prediction
