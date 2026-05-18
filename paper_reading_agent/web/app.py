from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path when run directly (e.g., via streamlit)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

import re

from paper_reading_agent.agent.core import PaperReadingAgent
from paper_reading_agent.config import AppConfig


_MATH_CMD_LIST = [
    "mathbf", "mathcal", "mathbb", "mathit", "mathrm", "boldsymbol", "bm",
    "hat", "bar", "tilde", "vec", "dot", "ddot", "widehat", "widetilde",
    "overline", "underline", "boxed", "operatorname", "mathop", "text",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    "partial", "infty", "nabla", "approx", "sim", "propto", "equiv",
    "times", "cdot", "sum", "prod", "int", "oint", "bigcup", "bigcap",
    "sqrt", "frac", "geq", "leq", "neq", "mapsto", "rightarrow", "Rightarrow",
    "left", "right", "langle", "rangle", "ldots", "cdots", "vdots", "ddots",
    "in", "notin", "subset", "supset", "subseteq", "supseteq",
    "forall", "exists", "emptyset", "varnothing", "top", "bot",
    "quad", "qquad",
    "Big", "Bigg", "big", "bigg", "Bigl", "Bigr", "bigl", "bigr",
    "oplus", "otimes", "odot", "circ", "star", "bullet",
    "triangle", "square", "diamond",
    "lVert", "rVert", "vert", "Vert",
]

# Regex to split text into math regions ($...$, $$...$$) and non-math regions
_MATH_REGION = re.compile(r"(\$\$.*?\$\$|\$[^$]+\$)", re.DOTALL)

# Build pattern that matches \cmd{...}, \cmd_x, \cmd^x, \cmd_{...}, \cmd^{...}
def _build_bare_latex_re() -> re.Pattern:
    cmds = "|".join(c for c in _MATH_CMD_LIST)
    # Bal: matches { ... } with up to one level of nested braces (e.g., {\text{abc}})
    _BAL = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    # Loop: after \cmd, repeatedly match: {arg}, _{arg}, _x, ^{arg}, or ^x
    suffix = rf"(?:{_BAL}|_(?:{_BAL}|[a-zA-Z0-9]+)|\^(?:{_BAL}|[a-zA-Z0-9]+))*"
    pattern = r"(?<!\w)(\\(" + cmds + r"))" + suffix
    return re.compile(pattern)

_BARE_LATEX_CMD = _build_bare_latex_re()


def _normalize_latex(text: str) -> str:
    """Convert LaTeX delimiters to $...$ / $$...$$ for Streamlit, protecting existing math."""
    # Step 1: Convert \begin{env}...\end{env} blocks to $$...$$ first (before splitting)
    for env in ("equation", "align", "aligned", "gather", "eqnarray"):
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}",
            r"$$\1$$", text, flags=re.DOTALL,
        )

    # Step 2: Split into math and non-math segments, protect existing $...$ / $$...$$
    parts: list[str] = []
    last_end = 0
    for m in _MATH_REGION.finditer(text):
        non_math = text[last_end:m.start()]
        parts.append(_normalize_non_math(non_math))
        parts.append(m.group(0))
        last_end = m.end()
    parts.append(_normalize_non_math(text[last_end:]))
    return "".join(parts)


def _normalize_non_math(text: str) -> str:
    """Normalize LaTeX in non-math text: convert delimiters, wrap bare commands."""
    # \[...\] → $$...$$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # \(...\) → $...$
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
    # Wrap bare LaTeX commands in $...$
    text = _BARE_LATEX_CMD.sub(r"$\g<0>$", text)
    return text


def init_agent() -> PaperReadingAgent:
    if "agent" not in st.session_state:
        config = AppConfig.from_env()
        st.session_state.agent = PaperReadingAgent(config)
    return st.session_state.agent


def init_state() -> None:
    defaults = {
        "paper_loaded": False,
        "report_generated": False,
        "messages": [],
        "section_list": [],
        "paper_title": "",
        "current_file_name": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


PIXEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

/* ---- global ---- */
.stApp { background: #f5f0e8; }
body, .stMarkdown, .stText, p, li, label {
    font-family: 'VT323', 'Courier New', monospace !important;
    font-size: 1.15rem;
    color: #2d2d2d !important;
}
h1 { font-family: 'Press Start 2P', monospace !important; font-size: 1.4rem !important; color: #c84c09 !important; }
h2 { font-family: 'Press Start 2P', monospace !important; font-size: 1.0rem !important; color: #c84c09 !important;
     border-bottom: 3px double #c84c09; padding-bottom: 0.25rem; }
h3 { font-family: 'Press Start 2P', monospace !important; font-size: 0.75rem !important; color: #2d2d2d !important; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #ede4d3;
    border-right: 3px solid #c84c09;
    image-rendering: pixelated;
}
[data-testid="stSidebar"] h3 { color: #c84c09 !important; }

/* ---- Nintendo-style banner ---- */
.nes-banner {
    background: linear-gradient(180deg, #0d1b3e 0%, #162852 50%, #0d1b3e 100%);
    border: 4px solid #f0c040;
    padding: 20px 24px 16px;
    margin-bottom: 18px;
    position: relative;
    box-shadow:
        6px 6px 0 #00000050,
        inset 0 0 0 2px #0d1b3e,
        inset 0 0 0 4px #c84040;
    image-rendering: pixelated;
    text-align: center;
}
.nes-banner::before {
    content: "";
    position: absolute; top: 6px; left: 6px; right: 6px; bottom: 6px;
    border: 1px dashed #f0c04030;
    pointer-events: none;
}
.nes-banner h1 {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.85rem !important;
    color: #f0c040 !important;
    text-shadow: 3px 3px 0 #00000080, 0 0 12px #f0c04040;
    margin: 4px 0 8px;
    letter-spacing: 3px;
    line-height: 1.6;
}
.nes-banner .sub {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.42rem;
    color: #aabbcc;
    letter-spacing: 2px;
    line-height: 1.8;
}
.nes-stars {
    font-size: 0.7rem;
    color: #f0c040;
    letter-spacing: 8px;
    text-shadow: 2px 2px 0 #00000060;
    margin-bottom: 2px;
}

/* ---- buttons ---- */
.stButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55rem !important;
    background: #faf6ef !important;
    color: #2d2d2d !important;
    border: 3px solid #2d2d2d !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 #c84c0940;
    transition: all 0.1s;
    image-rendering: pixelated;
}
.stButton > button:hover {
    background: #c84c09 !important;
    color: #fff !important;
    border-color: #c84c09 !important;
    box-shadow: 2px 2px 0 #c84c0940;
    transform: translate(2px, 2px);
}

/* ---- file uploader ---- */
[data-testid="stFileUploader"] section {
    border: 3px dashed #2d2d2d !important;
    background: #faf6ef !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #c84c09 !important;
    background: #fff8f0 !important;
}

/* ---- chat ---- */
[data-testid="stChatMessage"] {
    background: #faf6ef !important;
    border: 2px solid #2d2d2d;
    border-radius: 0;
    box-shadow: 4px 4px 0 #d5c4a1;
}
.stChatInput textarea {
    font-family: 'VT323', monospace !important;
    font-size: 1.1rem !important;
    background: #faf6ef !important;
    border: 2px solid #2d2d2d !important;
    border-radius: 0 !important;
    color: #2d2d2d !important;
}
.stChatInput textarea:focus {
    border-color: #c84c09 !important;
    box-shadow: 3px 3px 0 #c84c0930;
}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.5rem !important;
    color: #999 !important;
    border: 2px solid transparent;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #c84c09 !important;
    background: #faf6ef !important;
    border-bottom: 3px solid #c84c09 !important;
}

/* ---- alerts ---- */
.stAlert { border-radius: 0 !important; font-family: 'VT323', monospace !important; box-shadow: 3px 3px 0 #00000015; }
div[data-testid="stInfo"]    { background: #e8f0fe; border: 2px solid #5a8ec0; }
div[data-testid="stSuccess"] { background: #e6f4e6; border: 2px solid #5a9e5a; }
div[data-testid="stError"]   { background: #fbeae5; border: 2px solid #c84c09; }

/* ---- code ---- */
code {
    font-family: 'VT323', monospace !important;
    background: #faf6ef !important;
    border: 1px solid #d5c4a1;
    border-radius: 0 !important;
    color: #c84c09 !important;
    padding: 1px 6px;
}

/* ---- scrollbar ---- */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #f5f0e8; }
::-webkit-scrollbar-thumb { background: #d5c4a1; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #c84c09; }
</style>
"""

PIXEL_BANNER = """
<div class="nes-banner">
  <div class="nes-stars">★ ★ ★ ★ ★</div>
  <h1>📖 论 文 阅 读 助 手</h1>
  <div class="sub">
    ◆ AI 学术论文解读 ◆ 解析 ◆ 分析 ◆ 问答 ◆<br>
    上传 PDF，让 AI 为你深度解读论文
  </div>
</div>
"""


def main() -> None:
    st.set_page_config(page_title="论文阅读助手", page_icon="📖", layout="wide")
    st.markdown(PIXEL_CSS, unsafe_allow_html=True)
    st.markdown(PIXEL_BANNER, unsafe_allow_html=True)

    init_state()
    agent = init_agent()

    with st.sidebar:
        st.markdown("### ▸ 加载论文")
        uploaded_file = st.file_uploader("[ 拖放 PDF 文件 ]", type=["pdf"])

        if uploaded_file is not None:
            # Detect when user uploads a different file
            is_new_file = uploaded_file.name != st.session_state.current_file_name
            if not st.session_state.paper_loaded or is_new_file:
                with st.spinner(">>> 正在解析和索引论文..."):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        agent.load_paper(tmp_path)
                        paper = agent.paper
                        st.session_state.paper_loaded = True
                        st.session_state.current_file_name = uploaded_file.name
                        st.session_state.paper_title = paper.title or uploaded_file.name
                        st.session_state.report = ""
                        st.session_state.report_generated = False
                        st.session_state.messages = []
                        st.session_state.section_list = [
                            {"title": s.title, "level": s.level, "pages": f"{s.start_page+1}-{s.end_page+1}"}
                            for s in paper.sections
                        ]
                        st.success(f">> 已加载：{paper.title or uploaded_file.name}")
                        st.info(f">> 页数：{paper.metadata.get('page_count', '?')}  |  章节数：{len(paper.sections)}")
                    except Exception as e:
                        st.error(f"加载失败：{e}")
                        st.session_state.paper_loaded = False
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

        if st.session_state.paper_loaded:
            if st.button("▸ 生成报告", use_container_width=True):
                with st.spinner(">>> 正在生成解读报告..."):
                    try:
                        report = agent.generate_report()
                        st.session_state.report = report
                        st.session_state.report_generated = True
                        st.success(">> 报告已生成！")
                    except Exception as e:
                        st.error(f"报告生成失败：{e}")

            st.divider()
            st.markdown("### ▸ 论文目录")
            for sec in st.session_state.section_list:
                indent = "  " * (sec["level"] - 1)
                st.text(f"{indent}• {sec['title']} (第{sec['pages']}页)")

            if st.button("▸ 清除并加载新论文", use_container_width=True):
                agent.reset_conversation()
                for key in list(st.session_state.keys()):
                    if key != "agent":
                        del st.session_state[key]
                st.rerun()

    # Main content area
    tab1, tab2 = st.tabs(["▸ 解读报告", "▸ 问答"])

    with tab1:
        if st.session_state.report_generated:
            st.markdown(_normalize_latex(st.session_state.get("report", "")))
        else:
            st.info("> 等待输入。请上传论文并生成报告。")

    with tab2:
        if not st.session_state.paper_loaded:
            st.info("> 请先加载论文以开始问答。")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(_normalize_latex(msg["content"]))

            if prompt := st.chat_input("> 询问论文相关内容..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        try:
                            response = agent.ask_question(prompt)
                            st.markdown(_normalize_latex(response))
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"错误：{e}")


if __name__ == "__main__":
    main()
