import streamlit as st
import html
from ml_model import analyze_email_with_ai


def _score_color(score_percent):
    if score_percent > 50:
        return "#ff4b4b"
    if score_percent >= 25:
        return "#ffd43b"
    return "#00ff00"

st.set_page_config(page_title="Phishing Detector", layout="wide")

st.markdown("""
<style>
    .main .block-container {
        max-width: none;
        width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .risk-inner {
        width: 90px; height: 90px;
        border-radius: 50%;
        background-color: #262c3a;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .badge-safe { background-color: rgba(0, 255, 0, 0.1); color: #00ff00; padding: 2px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #00ff00;}
    .badge-suspicious { background-color: rgba(255, 212, 59, 0.12); color: #ffd43b; padding: 2px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #ffd43b;}
    .badge-flagged { background-color: rgba(255, 0, 0, 0.1); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #ff4b4b;}
    .url-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; align-items: center;}
    .url-text { word-break: break-all; margin-right: 10px; color: #ccc;}
    div[data-testid="stButton"] > button[kind="primary"] {
        border: 1px solid rgba(80, 120, 255, 0.55) !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        border-color: #4f8bff !important;
        box-shadow: 0 0 0 1px rgba(79, 139, 255, 0.25), 0 0 10px rgba(79, 139, 255, 0.18) !important;
    }
    div[data-testid="stButton"] > button:not([kind="primary"]):hover {
        border-color: #4f8bff !important;
        box-shadow: 0 0 0 1px rgba(79, 139, 255, 0.25), 0 0 10px rgba(79, 139, 255, 0.18) !important;
    }
    div[data-testid="stTextArea"] {
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        margin-top: -2px;
        position: relative;
        z-index: 1;
        overflow: visible !important;
    }
    div[data-testid="stTextArea"]:focus-within {
        z-index: 25;
        box-shadow: none !important;
    }
    div[data-testid="stTextArea"] > div {
        overflow: visible !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextArea"] [data-baseweb="textarea"] {
        box-shadow: none !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
    }
    div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within {
        box-shadow: none !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        outline: none !important;
    }
    div[data-testid="stTextArea"] textarea {
        min-height: 374px !important;
        height: 374px !important;
        resize: none !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        background: #111723 !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextArea"] textarea:focus-visible {
        outline: none !important;
        border-color: rgba(255, 255, 255, 0.14) !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

if "step" not in st.session_state:
    st.session_state.step = "input"

if st.session_state.step == "input":
    with st.container(border=True):
        st.markdown("<h3 style='margin-bottom: 0;'>Phishing Detector</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #888; font-size: 14px; margin-top: 0;'>Analyze suspicious email content.</p>", unsafe_allow_html=True)

        email_content = st.text_area("Email Input Area", placeholder="Paste Email Here...", height=374, label_visibility="collapsed")
        _, center_btn_col, _ = st.columns([2, 1, 2])
        with center_btn_col:
            analyze_now = st.button("ANALYZE NOW", type="primary", use_container_width=True)

        if analyze_now:
            if email_content:
                st.session_state.raw_email = email_content
                st.session_state.step = "analyzing"
                st.rerun()
            else:
                st.warning("Please paste an email first.")

elif st.session_state.step == "analyzing":
    st.write("<br><br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Phishing Detector</h3>", unsafe_allow_html=True)
        st.write("<br><br>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing email content, URLs, screenshots, NLP, and vision model..."):
            results = analyze_email_with_ai(st.session_state.raw_email)
            st.session_state.final_data = results
            st.session_state.step = "results"
            
        st.write("<br><br>", unsafe_allow_html=True)
        st.rerun()

elif st.session_state.step == "results":
    data = st.session_state.final_data
    
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Phishing Detector</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 12px; letter-spacing: 2px;'>RESULT SECTION</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("⚠️ **RISK ASSESSMENT**")

                classification = data.get('classification', data['risk_level'])
                stacked_scores = data.get('stacked_scores', {})
                circle_css = f"""
                <p style="text-align: left; color: #ccc; font-size: 14px; margin-bottom: 8px;">Logistic Stacking Result:</p>
                <div style="display: flex; align-items: center; justify-content: center; margin-top: 10px;">
                    <div style="width: 120px; height: 120px; border-radius: 50%; background: conic-gradient({data['color']} {data['risk_score']}%, #333 0); display: flex; align-items: center; justify-content: center; border: 5px solid #1e232f; margin: 0 auto;">
                        <div class="risk-inner">
                            <span style="font-size: 24px; font-weight: bold; color: {data['color']};">{data['risk_score']}%</span>
                        </div>
                    </div>
                </div>
                """
                st.markdown(circle_css, unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: {data['color']}; font-weight: bold; margin-top: 10px;'>{classification}</p>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='font-size: 12px; color: #ccc; margin-bottom: 2px; text-align: center;'>BERT NLP: {stacked_scores.get('bert_nlp_risk_score', 0):.2f}% | MobileNetV2: {stacked_scores.get('mobilenet_vision_risk_score', 0):.2f}%</p>",
                    unsafe_allow_html=True,
                )

        with col2:
            with st.container(border=True):
                st.markdown("☑️ **NLP ANALYSIS**")
                nlp_summary = data.get('nlp_summary', {})
                nlp_score = nlp_summary.get('bert_nlp_risk_score', 0.0)
                nlp_assessment = nlp_summary.get('assessment', 'SAFE')
                nlp_assessment_color = "#ff4b4b" if nlp_assessment == "DANGER" else "#ffd43b" if nlp_assessment == "SUSPICIOUS" else "#00ff00"
                st.markdown(
                    f"<p style='font-size: 15px; color: #ccc; margin-bottom: 4px;'>BERT NLP Risk Score: <span style='font-weight:700; color:{nlp_assessment_color};'>{nlp_score:.2f}%</span></p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 15px; color: #ccc; margin-bottom: 8px;'>Assessment: <span style='font-weight:700; color:{nlp_assessment_color};'>{nlp_assessment}</span></p>",
                    unsafe_allow_html=True,
                )
                findings = data.get('nlp_findings', [])
                sentiment_finding = None

                for finding in findings:
                    finding_text = str(finding).strip()
                    if "suspicious url structure score" in finding_text.lower():
                        continue
                    if "sentiment" in finding_text.lower():
                        sentiment_finding = finding_text
                        continue

                    color = "#ff4b4b" if "Urgent" in str(finding) or "Suspicious" in str(finding) else "#ccc"
                    st.markdown(
                        f"<p style='color: {color}; margin-bottom: 5px; font-size: 16px;'>• {finding}</p>",
                        unsafe_allow_html=True,
                    )

                # Intentionally suppressing explicit Sentiment display in NLP ANALYSIS tile.

        with col1:
            with st.container(border=True):
                st.markdown("🔗 **EXTRACTED URLs ANALYSIS**")
                url_summary = data.get('url_summary', {})
                url_score = float(url_summary.get('url_analyzer_risk_score', 0.0))
                url_assessment = url_summary.get('assessment', 'SAFE')
                url_color = "#ff4b4b" if url_assessment == "DANGER" else "#ffd43b" if url_assessment == "SUSPICIOUS" else "#00ff00"

                st.markdown(
                    f"<p style='font-size: 14px; color: #ccc; margin-bottom: 2px;'>URL Analyzer Risk Score: <span style='font-weight:700; color:{url_color};'>{url_score:.2f}%</span></p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 14px; color: #ccc; margin-bottom: 8px;'>Assessment: <span style='font-weight:700; color:{url_color};'>{url_assessment}</span></p>",
                    unsafe_allow_html=True,
                )

                with st.container(height=181):
                    if not data['extracted_urls']:
                        st.markdown("<p style='color: #888; font-size: 14px;'>No URLs found in email.</p>", unsafe_allow_html=True)
                    else:
                        for i, url_obj in enumerate(data['extracted_urls']):
                            indiv_url_score = float(url_obj.get('url_risk_score', 0.0))
                            indiv_url_score_color = _score_color(indiv_url_score)
                            safe_display_url = html.escape(url_obj['url'])
                            # Keep URLs as plain text so none are clickable in this panel.
                            plain_url_text = safe_display_url.replace("://", "://&#8203;").replace(".", ".&#8203;")

                            col_url, col_score = st.columns([3, 1])
                            with col_url:
                                st.markdown(
                                    f"<span style='color: #888; word-break: break-all;'>{i+1}. {plain_url_text}</span>",
                                    unsafe_allow_html=True
                                )
                            with col_score:
                                st.markdown(
                                    f"<span style='color:#aaa; font-size:14px;'>Risk Score: <span style='font-weight:700; color:{indiv_url_score_color};'>{indiv_url_score:.2f}%</span></span>",
                                    unsafe_allow_html=True,
                                )

        with col2:
            with st.container(border=True):
                st.markdown("👁️ **VISION ANALYSIS**")
                vision_summary = data.get('vision_summary', {})
                avg_vision_risk = vision_summary.get('avg_vision_risk', 0)
                vision_assessment = vision_summary.get('assessment', 'SAFE')

                captured_items = [
                    obj for obj in data.get('extracted_urls', []) if obj.get('screenshot_path')
                ]

                if captured_items:
                    carousel_key = "vision_carousel_index"
                    if carousel_key not in st.session_state:
                        st.session_state[carousel_key] = 0

                    current_index = st.session_state[carousel_key]
                    if current_index >= len(captured_items):
                        current_index = len(captured_items) - 1
                    if current_index < 0:
                        current_index = 0
                    st.session_state[carousel_key] = current_index

                    current_item = captured_items[current_index]
                    risk_color = "#ff4b4b" if vision_assessment == "DANGER" else "#ffd43b" if vision_assessment == "SUSPICIOUS" else "#00ff00"
                    current_vision_score = float(current_item.get('vision_risk_score', avg_vision_risk))
                    current_vision_assessment = "DANGER" if current_vision_score > 50 else ("SUSPICIOUS" if current_vision_score >= 25 else "SAFE")
                    current_vision_notes = str(current_item.get('vision_notes', '') or '').lower()
                    is_ssl_privacy_warning = (
                        "browser certificate/privacy warning detected" in current_vision_notes
                        or "ssl/privacy warning interstitial" in current_vision_notes
                        or current_item.get('vision_label') == 'SSL/Privacy warning interstitial'
                        or current_vision_score >= 100
                    )

                    with st.container(height=373):
                        st.markdown(
                            f"<p style='font-size: 14px; color: #ccc; margin-bottom: 2px;'>MobileNetV2 Vision Risk Score: <span style='font-weight:700; color:{risk_color};'>{avg_vision_risk:.2f}%</span> <span style='color: #888;'>({len(captured_items)} screenshots)</span></p>",
                            unsafe_allow_html=True,
                        )
                        assessment_label = (
                            "<span style='display:inline-flex; align-items:center; gap:8px; font-size:14px; color:#ccc; margin-bottom:6px;'>"
                            f"Assessment: <span style='font-weight:700; color:{risk_color};'>{vision_assessment}</span>"
                        )
                        if is_ssl_privacy_warning:
                            assessment_label += (
                                "<span style='display:inline-block; padding:3px 9px; border-radius:999px; background:rgba(255,75,75,0.14); color:#ff6b6b; border:1px solid rgba(255,75,75,0.35); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.7px;'>"
                                "Certificate / Privacy Warning Detected - 100%"
                                "</span>"
                            )
                        assessment_label += "</span>"
                        st.markdown(assessment_label, unsafe_allow_html=True)
                        nav_left, nav_center, nav_right = st.columns([1, 6, 1], vertical_alignment="center")
                        with nav_left:
                            prev_disabled = current_index <= 0
                            if st.button("◀", key="vision_prev", disabled=prev_disabled, use_container_width=True):
                                st.session_state[carousel_key] = max(0, current_index - 1)
                                st.rerun()
                        with nav_center:
                            current_source_url = html.escape(current_item.get('url', 'Unknown URL'))
                            st.image(
                                current_item['screenshot_path'],
                                width="content",
                            )
                            st.markdown(
                                f"<p style='font-size: 12px; color: #aaa; margin-top: 2px; text-align: center;'>{current_source_url} | {current_index + 1} of {len(captured_items)}</p>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<p style='font-size: 14px; color: #ccc; margin-top: 2px; text-align: center;'>Individual Vision Score: <span style='font-weight:700; color:{'#ff4b4b' if current_vision_assessment == 'DANGER' else '#ffd43b' if current_vision_assessment == 'SUSPICIOUS' else '#00ff00'};'>{current_vision_score:.2f}%</span> &nbsp; Assessment: <span style='font-weight:700;'>{current_vision_assessment}</span></p>",
                                unsafe_allow_html=True,
                            )
                        with nav_right:
                            next_disabled = current_index >= len(captured_items) - 1
                            if st.button("▶", key="vision_next", disabled=next_disabled, use_container_width=True):
                                st.session_state[carousel_key] = min(len(captured_items) - 1, current_index + 1)
                                st.rerun()
                else:
                    with st.container(height=430):
                        st.markdown("<p style='color: #888; font-size: 14px; text-align: center;'>No screenshot available</p>", unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    if st.button("← Analyze Another Email", use_container_width=False):
        st.session_state.step = "input"
        st.rerun()