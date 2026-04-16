# GeoPulse 프로젝트 컨텍스트

## 프로젝트 개요
AI 기반 지정학 분쟁 분석 플랫폼

## 기술 스택
- DL: Keras (Embedding + GlobalAveragePooling)
- RAG: FAISS + SentenceTransformer
- LLM: Gemini 2.0 Flash (google-generativeai==0.8.6)
- UI: Gradio 4.37.2

## 파일 구조
- GeoPulse_DL.ipynb  → DL 분류기 학습
- GeoPulse_RAG.ipynb → RAG + Gemini 리포트
- app.py             → Gradio UI
- data/GeoPulse_Final_Dataset_KOREAN.csv

## 현재 진행 상황
- DL 노트북 완성
- RAG 노트북 완성
- Gradio app.py 작성 중