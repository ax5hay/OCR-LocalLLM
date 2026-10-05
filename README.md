# OCR + Local LLM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-text_extraction-4a4a52?style=flat-square)
![Local LLM](https://img.shields.io/badge/LLM-local_·_private-8c2f24?style=flat-square)

A minimal, **single-file Streamlit app** that reads a PDF, pulls its text with
[PyMuPDF](https://pymupdf.readthedocs.io/), and lets you ask a **local LLM** questions grounded
in that document. Nothing leaves your machine: the model runs locally, so private documents
stay private.

> This is the small prototype. The production, distributed version lives in
> **[OCR-LLM-DIST](https://github.com/ax5hay/OCR-LLM-DIST)** (FastAPI + Next.js, Ollama/LMStudio, streaming, Dockerized).

## What it does

1. **Upload** a PDF in the Streamlit UI.
2. **Extract** text page-by-page with PyMuPDF.
3. **Ask**: your question plus the extracted text is sent to a locally-running LLM endpoint.
4. **Read** the model's answer, grounded in the document.

## Run it

```bash
pip install -r requirements.txt    # streamlit, requests, pymupdf, pandas
# start your local LLM server first (e.g. Ollama or LMStudio), then:
streamlit run updatedstreamlitapp.py
```

Point the app at your local model endpoint, drop in a PDF, and start asking.
