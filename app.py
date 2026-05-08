# ============================================================
# GeoPulse app.py
# DL 2개 분류기 + RAG + Gemini 통합 Gradio 앱
# 실행: python app.py
#
# 실행 전 확인:
#   1. GeoPulse_DL.ipynb 실행 완료
#      → models/cause_classifier.keras
#      → models/risk_classifier.keras
#      → models/tokenizer.pkl
#   2. GeoPulse_RAG.ipynb 실행 완료
#      → rag/faiss_index.bin
#      → rag/metadata.pkl
#   3. .env 파일에 GEMINI_API_KEY 설정
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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
EMBED_MODEL    = 'paraphrase-multilingual-MiniLM-L12-v2'
MAX_LEN        = 30
TOP_K          = 5
INDEX_PATH     = 'rag/faiss_index.bin'
META_PATH      = 'rag/metadata.pkl'

# 라벨 역변환 (숫자 → 한글)
CAUSE_REV = {0: '영토', 1: '정부/권력', 2: '복합'}
RISK_REV  = {0: '소규모', 1: '전면전'}


# ── 모델 로드 ─────────────────────────────────────────────────
print('🚀 GeoPulse 초기화 중...')

# DL 모델 로드
try:
    cause_model = load_model('models/cause_classifier.keras')
    risk_model  = load_model('models/risk_classifier.keras')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # 강도 분류기용 토크나이저 (분리된 경우)
    try:
        with open('models/tokenizer_risk.pkl', 'rb') as f:
            tokenizer_risk = pickle.load(f)
    except:
        tokenizer_risk = tokenizer  # 없으면 동일 토크나이저 사용
    DL_READY = True
    print('✅ DL 모델 로드 완료')
except Exception as e:
    DL_READY = False
    print(f'❌ DL 모델 로드 실패: {e}')
    print('   GeoPulse_DL.ipynb 먼저 실행하세요')

# RAG 로드
try:
    print('📥 임베딩 모델 로딩 중...')
    embedder    = SentenceTransformer(EMBED_MODEL)
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    RAG_READY = True
    print(f'✅ RAG 로드 완료: {faiss_index.ntotal}개 벡터')
except Exception as e:
    RAG_READY = False
    print(f'❌ RAG 로드 실패: {e}')
    print('   GeoPulse_RAG.ipynb 먼저 실행하세요')

# Gemini 로드
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_READY = True
    print('✅ Gemini 초기화 완료')
except Exception as e:
    GEMINI_READY = False
    print(f'❌ Gemini 초기화 실패: {e}')


# ── 핵심 함수 3개 ─────────────────────────────────────────────

def predict_dl(query: str) -> tuple:
    """DL 2개 분류기로 분쟁 원인/강도 예측"""
    if not DL_READY:
        return (
            {'label': '미분류', 'confidence': 0.0},
            {'label': '미분류', 'confidence': 0.0}
        )

    # 강도 분류기용: 키워드로 사망자 토큰 추가
    keywords_large = ['전쟁', '전면전', '내전', '침공', '폭격']
    keywords_small = ['충돌', '국경', '테러', '소규모']
    if any(k in query for k in keywords_large):
        death_token = '대규모전쟁 전면전'
    elif any(k in query for k in keywords_small):
        death_token = '소규모충돌'
    else:
        death_token = '중규모전쟁'
    query_risk = f'{query} {death_token}'

    # 원인 분류기 예측
    seq_c  = tokenizer.texts_to_sequences([query])
    pad_c  = pad_sequences(seq_c, maxlen=MAX_LEN, padding='post')
    prob_c = cause_model.predict(pad_c, verbose=0)[0]
    idx_c  = int(np.argmax(prob_c))

    # 강도 분류기 예측
    seq_r  = tokenizer_risk.texts_to_sequences([query_risk])
    pad_r  = pad_sequences(seq_r, maxlen=MAX_LEN, padding='post')
    prob_r = risk_model.predict(pad_r, verbose=0)[0]
    idx_r  = int(np.argmax(prob_r))

    return (
        {'label': CAUSE_REV[idx_c], 'confidence': float(prob_c[idx_c])},
        {'label': RISK_REV[idx_r],  'confidence': float(prob_r[idx_r])}
    )


def search_rag(query: str, cause_label: str = '', risk_label: str = '') -> list:
    # DL 결과를 쿼리에 추가 ← 핵심!
    enhanced_query = f"{query} {cause_label} {risk_label}"
    
    q_vec = embedder.encode(
        [enhanced_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    scores, indices = faiss_index.search(q_vec, TOP_K * 3)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            item = dict(metadata[idx])
            item['유사도'] = float(score)
            
            # DL 결과와 일치하는 사례 우선 배치
            if (item.get('분쟁원인') == cause_label and
                item.get('전쟁강도') == risk_label):
                item['매칭'] = '✅ DL 일치'
                results.insert(0, item)
            else:
                item['매칭'] = '📌 유사'
                results.append(item)
    
    return results[:TOP_K]


def generate_report(query: str, cause_result: dict,
                    risk_result: dict, similar: list) -> str:
    """Gemini로 7개 섹션 종합 분석 리포트 생성"""
    if not GEMINI_READY:
        return '❌ Gemini API 키를 .env 파일에 설정해주세요.\nGEMINI_API_KEY=your_key_here'

    cases_text = '\n'.join([
        f"  {i+1}. {c.get('발생지','')} ({c.get('연도','')}) | "
        f"원인: {c.get('분쟁원인','')} | "
        f"강도: {c.get('전쟁강도','')} | "
        f"사망자: {c.get('사망자_추정치','')} | "
        f"유사도: {c.get('유사도',0):.3f}"
        for i, c in enumerate(similar[:5])
    ])

    prompt = f"""당신은 국제정치 및 지정학 전문 분석가입니다.
사용자가 "{query}"에 대해 분석을 요청했습니다.
아래 데이터를 바탕으로 GeoPulse 종합 분쟁 분석 리포트를 작성하세요.

=== AI 딥러닝 분석 결과 ===
분쟁원인 분류: {cause_result['label']} (신뢰도 {cause_result['confidence']*100:.1f}%)
전쟁강도 분류: {risk_result['label']} (신뢰도 {risk_result['confidence']*100:.1f}%)

=== 역사적 유사 사례 (RAG 검색 TOP 5) ===
{cases_text}

다음 7개 섹션으로 리포트를 작성하세요:

1. 📋 분쟁 개요
2. 🔍 공식 명분 vs 숨겨진 이유 (자원/지정학/패권/경제)
3. 🌐 주변국 & 동맹국 관계망 (지원국/반대국/중립국)
4. 📜 역사적 패턴 분석 (유사 사례 비교)
5. ⚡ 전쟁 가능성 & 시나리오 (% + A/B/C)
6. 💰 피해 & 이득 분석
7. 🔮 종합 전망 (3줄 요약)

⚠️ 본 리포트는 AI 추론 기반 분석으로, 공식 입장과 다를 수 있습니다."""

    return gemini_model.generate_content(prompt).text


# ── Gradio 메인 함수 ──────────────────────────────────────────

def analyze(query: str):
    """분석 실행: DL → RAG → Gemini"""
    if not query.strip():
        return (
            '',
            '',
            '❌ 분석할 분쟁을 입력해주세요.\n예: 우크라이나 / 중동 분쟁 / 수단 내전'
        )
    try:
        # 1) DL 예측
        cause_result, risk_result = predict_dl(query)

        # 2) RAG 검색
        similar = search_rag(query)

        # 3) DL 결과 텍스트
        dl_text  = '🤖 딥러닝 분류 결과\n'
        dl_text += '━' * 28 + '\n'
        dl_text += f"⚡ 분쟁원인: {cause_result['label']} ({cause_result['confidence']*100:.1f}%)\n"
        dl_text += f"💥 전쟁강도: {risk_result['label']} ({risk_result['confidence']*100:.1f}%)\n"
        dl_text += f"\n{'✅ DL 정상' if DL_READY else '⚠️ DL 미로드'}"

        # 4) RAG 결과 텍스트
        if similar:
            rag_text  = f"📚 '{query}' 유사 사례 TOP {len(similar)}\n"
            rag_text += '━' * 28 + '\n'
            for i, c in enumerate(similar, 1):
                rag_text += f"{i}. {c.get('발생지','')} ({c.get('연도','')})\n"
                rag_text += f"   원인: {c.get('분쟁원인','')} | 강도: {c.get('전쟁강도','')}\n"
                rag_text += f"   사망자: {c.get('사망자_추정치','')} | 유사도: {c.get('유사도',0):.3f}\n"
        else:
            rag_text = '⚠️ RAG 인덱스 미로드\nGeoPulse_RAG.ipynb 먼저 실행하세요'

        # 5) Gemini 리포트
        report = generate_report(query, cause_result, risk_result, similar)

        return dl_text, rag_text, report

    except Exception as e:
        return f'❌ 오류: {str(e)}', '', ''


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title='분쟁 분석') as app:

    gr.Markdown("""
### AI 기반 지정학 분쟁 분석 플랫폼
**딥러닝 분류 + RAG 검색 + Gemini 리포트**
---
""")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label='🔍 분석할 분쟁을 입력하세요',
                placeholder='예: 우크라이나   /   중동 분쟁   /   수단 내전   /   미얀마 내전',
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            analyze_btn = gr.Button('🔍 분석 시작', variant='primary', size='lg')

    gr.Markdown('---')

    with gr.Row():
        dl_output  = gr.Textbox(label='🤖 DL 분류 결과', lines=6, interactive=False)
        rag_output = gr.Textbox(label='📚 유사 사례 (RAG)', lines=6, interactive=False)

    report_output = gr.Textbox(
        label='📋 GeoPulse 종합 분석 리포트',
        lines=30, interactive=False
    )

    gr.Markdown("""
---
⚠️ 본 플랫폼은 AI 추론 기반 분석으로, 공식 입장과 다를 수 있습니다.
📊 데이터: UCDP (Uppsala Conflict Data Program, 1989~2024)
""")

    analyze_btn.click(
        fn=analyze,
        inputs=[query_input],
        outputs=[dl_output, rag_output, report_output]
    )
    query_input.submit(
        fn=analyze,
        inputs=[query_input],
        outputs=[dl_output, rag_output, report_output]
    )


# ── 실행 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\n🌍 GeoPulse 시작!')
    print(f'   DL  모델:   {"✅" if DL_READY else "❌"}')
    print(f'   RAG 인덱스: {"✅" if RAG_READY else "❌"}')
    print(f'   Gemini API: {"✅" if GEMINI_READY else "❌"}')
    print()
    app.launch(
        share=True,
        server_name='0.0.0.0',
        server_port=7860,
        show_error=True
    )