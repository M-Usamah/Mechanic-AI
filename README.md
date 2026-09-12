# Mechanic AI

Vehicle diagnosis assistant that reads workshop spreadsheets and answers engine-fault questions with an LLM. Includes Streamlit, Gradio, and desktop GUI front ends.

## What it does

- Loads manufacturer diagnosis tables (Ford, Audi, Hyundai, and others)
- Asks a Groq-hosted LLM to interpret symptoms against those tables
- Serves the same agent from Streamlit, Gradio, or a local GUI

## Stack

Python, Groq LLM, Streamlit, Gradio, pandas

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (do not commit it):

```
GROQ_API_KEY=your_key_here
```

Then run one of:

```bash
python vehicle_diagnosis_streamlit.py
python vehicle_diagnosis_gradio.py
python vehicle_diagnosis_gui_ultra_modern.py
```
