# ============================================================
# GeoPulse - Gradio 앱 (심플 버전 - 입력창 하나)
# 실행: python app.py
# ============================================================

import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()

# ── 설정 ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN        = 30
TOP_K          = 5
INDEX_PATH     = "rag/faiss_index.bin"
META_PATH      = "rag/metadata.pkl"
CAUSE_REV      = {0: "영토", 1: "정부/권력", 2: "복합"}
RISK_REV       = {0: "소규모", 1: "전면전"}

# ════════════════════════════════════════════════════════════
# 모델 로드
# ════════════════════════════════════════════════════════════
print("🚀 GeoPulse 초기화 중...")

try:
    cause_model = load_model("models/cause_classifier.keras")
    risk_model  = load_model("models/risk_classifier.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    DL_READY = True
    print("✅ DL 모델 로드 완료")
except Exception as e:
    DL_READY = False
    print(f"⚠️ DL 모델 미로드: {e}")

try:
    print("📥 임베딩 모델 로딩 중...")
    embedder = SentenceTransformer(EMBED_MODEL)
    index    = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    RAG_READY = True
    print(f"✅ RAG 로드 완료: {index.ntotal}개 벡터")
except Exception as e:
    RAG_READY = False
    print(f"⚠️ RAG 미로드: {e}")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    GEMINI_READY = True
    print("✅ Gemini 초기화 완료")
except Exception as e:
    GEMINI_READY = False
    print(f"⚠️ Gemini 미초기화: {e}")


# ════════════════════════════════════════════════════════════
# 핵심 함수
# ════════════════════════════════════════════════════════════

def predict_dl(query_text: str) -> tuple:
    if not DL_READY:
        return ({"label": "미분류", "confidence": 0.0},
                {"label": "미분류", "confidence": 0.0})
    seq    = tokenizer.texts_to_sequences([query_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    c_prob = cause_model.predict(padded, verbose=0)[0]
    r_prob = risk_model.predict(padded, verbose=0)[0]
    c_idx  = int(np.argmax(c_prob))
    r_idx  = int(np.argmax(r_prob))
    return (
        {"label": CAUSE_REV[c_idx], "confidence": float(c_prob[c_idx])},
        {"label": RISK_REV[r_idx],  "confidence": float(r_prob[r_idx])},
    )


def search_rag(query: str) -> list:
    if not RAG_READY:
        return []
    q_vec = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.search(q_vec, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            item = dict(metadata[idx])
            item["similarity"] = float(score)
            results.append(item)
    return results


def generate_report(query: str, similar_cases: list,
                    cause_result: dict, risk_result: dict) -> str:
    if not GEMINI_READY:
        return "❌ Gemini API 키를 .env 파일에 설정해주세요.\nGEMINI_API_KEY=your_key_here"

    cases_text = "\n".join([
        f"  {i+1}. {c.get('발생지','')} ({c.get('연도','')}) | "
        f"원인: {c.get('분쟁원인','')} | "
        f"강도: {c.get('전쟁강도','')} | "
        f"사망자: {c.get('사망자_추정치','')}"
        for i, c in enumerate(similar_cases[:5])
    ])

    prompt = f"""당신은 국제정치 및 지정학 전문 분석가입니다.
사용자가 "{query}"에 대해 분석을 요청했습니다.
아래 데이터를 바탕으로 GeoPulse 종합 분쟁 분석 리포트를 작성하세요.

=== AI 딥러닝 분석 결과 ===
분쟁원인 분류: {cause_result['label']} (신뢰도 {cause_result['confidence']*100:.1f}%)
전쟁강도 분류: {risk_result['label']} (신뢰도 {risk_result['confidence']*100:.1f}%)

=== 역사적 유사 사례 (RAG 검색) ===
{cases_text}

다음 7개 섹션으로 리포트를 작성하세요:

1. 📋 분쟁 개요 (어떤 분쟁인지 먼저 설명)
2. 🔍 공식 명분 vs 숨겨진 이유 (자원/지정학/패권/경제 관점)
3. 🌐 주변국 & 동맹국 관계망
   - 지원국: 이유 + 예상 이득
   - 반대국: 이유 + 예상 손해
   - 중립국: 전략적 입장
4. 📜 역사적 패턴 분석 (유사 사례 비교)
5. ⚡ 전쟁 가능성 & 시나리오
   - 전면전 가능성: XX%
   - 시나리오 A/B/C
6. 💰 피해 & 이득 분석
   - 인적/경제적 피해
   - 예상 쟁취물 & 이후 활용
7. 🔮 종합 전망 (핵심 결론 3줄)

⚠️ 본 리포트는 AI 추론 기반 분석으로, 공식 입장과 다를 수 있습니다.
"""
    response = gemini_model.generate_content(prompt)
    return response.text


# ════════════════════════════════════════════════════════════
# Gradio 메인 함수
# ════════════════════════════════════════════════════════════

def analyze(query: str):
    """단일 입력으로 전체 분석 실행"""
    if not query.strip():
        return "", "", "❌ 분석할 분쟁을 입력해주세요.\n예: 우크라이나, 중동 분쟁, 수단 내전"

    try:
        # 1) RAG 검색
        similar_cases = search_rag(query)

        # 2) DL 예측
        cause_result, risk_result = predict_dl(query)

        # 3) DL 결과 텍스트
        dl_text = (
            f"🤖 딥러닝 분류 결과\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"분쟁원인: {cause_result['label']} "
            f"({cause_result['confidence']*100:.1f}%)\n"
            f"전쟁강도: {risk_result['label']} "
            f"({risk_result['confidence']*100:.1f}%)\n"
            f"{'✅ DL 모델 정상' if DL_READY else '⚠️ DL 모델 미로드'}"
        )

        # 4) 유사 사례 텍스트
        if similar_cases:
            similar_text = f"📚 '{query}' 관련 유사 사례 TOP {len(similar_cases)}\n"
            similar_text += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for i, c in enumerate(similar_cases[:5], 1):
                similar_text += (
                    f"{i}. {c.get('발생지','')} ({c.get('연도','')})\n"
                    f"   원인: {c.get('분쟁원인','')} | "
                    f"강도: {c.get('전쟁강도','')} | "
                    f"유사도: {c.get('similarity',0):.3f}\n"
                )
        else:
            similar_text = "⚠️ RAG 인덱스 미로드"

        # 5) Gemini 리포트
        report = generate_report(query, similar_cases, cause_result, risk_result)

        return dl_text, similar_text, report

    except Exception as e:
        return f"❌ 오류: {str(e)}", "", ""


# ════════════════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════════════════

with gr.Blocks(title="GeoPulse - 지정학 분쟁 분석") as app:

    gr.Markdown("""
# 🌍 GeoPulse
### AI 기반 지정학 분쟁 분석 플랫폼
**DL 분류 + RAG 검색 + Gemini 리포트**
---
""")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="🔍 분석할 분쟁을 입력하세요",
                placeholder="예: 우크라이나   /   중동 분쟁   /   수단 내전   /   가자지구",
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            analyze_btn = gr.Button(
                "🔍 분석 시작",
                variant="primary",
                size="lg",
            )

    gr.Markdown("---")

    with gr.Row():
        dl_output = gr.Textbox(
            label="🤖 DL 분류 결과",
            lines=5,
            interactive=False,
        )
        similar_output = gr.Textbox(
            label="📚 유사 사례 (RAG 검색)",
            lines=5,
            interactive=False,
        )

    report_output = gr.Textbox(
        label="📋 GeoPulse 종합 분석 리포트",
        lines=30,
        interactive=False,
    )

    gr.Markdown("""
---
⚠️ 본 플랫폼은 AI 추론 기반 분석으로, 공식 입장 및 학술 연구와 다를 수 있습니다.
📊 데이터 출처: UCDP (Uppsala Conflict Data Program)
""")

    # 버튼 클릭 & 엔터 모두 지원
    analyze_btn.click(
        fn=analyze,
        inputs=[query_input],
        outputs=[dl_output, similar_output, report_output],
    )
    query_input.submit(
        fn=analyze,
        inputs=[query_input],
        outputs=[dl_output, similar_output, report_output],
    )


# ════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🌍 GeoPulse 시작!")
    print(f"   DL  모델: {'✅' if DL_READY else '❌'}")
    print(f"   RAG 인덱스: {'✅' if RAG_READY else '❌'}")
    print(f"   Gemini API: {'✅' if GEMINI_READY else '❌'}")
    print()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )