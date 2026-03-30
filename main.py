import streamlit as st
import html
from ml_model import analyze_email_with_ai

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
    .badge-flagged { background-color: rgba(255, 0, 0, 0.1); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #ff4b4b;}
    .url-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; align-items: center;}
    .url-text { word-break: break-all; margin-right: 10px; color: #ccc;}
</style>
""", unsafe_allow_html=True)

if "step" not in st.session_state:
    st.session_state.step = "input"

if st.session_state.step == "input":
    st.write("<br><br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Phishing Detector</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 14px;'>Analyze suspicious email content.</p>", unsafe_allow_html=True)
        
        email_content = st.text_area("Email Input Area", placeholder="Paste Email Here...", height=200, label_visibility="collapsed")
        
        if st.button("ANALYZE NOW", type="primary", use_container_width=True):
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
                circle_css = f"""
                <div style="display: flex; align-items: center; justify-content: space-around; margin-top: 10px;">
                    <div style="width: 120px; height: 120px; border-radius: 50%; background: conic-gradient({data['color']} {data['risk_score']}%, #333 0); display: flex; align-items: center; justify-content: center; border: 5px solid #1e232f; margin: 0 auto;">
                        <div class="risk-inner">
                            <span style="font-size: 24px; font-weight: bold; color: {data['color']};">{data['risk_score']}%</span>
                        </div>
                    </div>
                    <div>
                        <h4 style="margin:0; color: white;">{classification}</h4>
                    </div>
                </div>
                """
                st.markdown(circle_css, unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: {data['color']}; font-weight: bold; margin-top: 10px;'>{classification}</p>", unsafe_allow_html=True)

        with col2:
            with st.container(border=True):
                st.markdown("☑️ **NLP ANALYSIS FINDINGS**")
                findings = data.get('nlp_findings', [])
                sentiment_finding = None

                for finding in findings:
                    finding_text = str(finding).strip()
                    if "sentiment" in finding_text.lower():
                        sentiment_finding = finding_text
                        continue

                    color = "#ff4b4b" if "Urgent" in str(finding) or "Suspicious" in str(finding) else "#ccc"
                    st.markdown(
                        f"<p style='color: {color}; margin-bottom: 5px; font-size: 14px;'>• {finding}</p>",
                        unsafe_allow_html=True,
                    )

                if sentiment_finding:
                    raw_sentiment = sentiment_finding.split(":", 1)[1].strip() if ":" in sentiment_finding else sentiment_finding
                    raw_lower = raw_sentiment.lower()
                    finding_lower = sentiment_finding.lower()

                    if any(token in raw_lower for token in ["danger", "fear"]) or "high sentiment" in finding_lower:
                        sentiment_value = "DANGER"
                    elif any(token in raw_lower for token in ["caution", "medium", "moderate"]):
                        sentiment_value = "CAUTION"
                    elif any(token in raw_lower for token in ["safe", "normal", "low"]):
                        sentiment_value = "SAFE"
                    else:
                        sentiment_value = raw_sentiment.upper()

                    sentiment_color = {
                        "SAFE": "#00ff00",
                        "CAUTION": "#ffd43b",
                        "DANGER": "#ff4b4b",
                    }.get(sentiment_value, "#ccc")
                    st.markdown(
                        f"<p style='color: #ccc; margin-bottom: 5px; font-size: 14px;'>• Sentiment: <span style='color: {sentiment_color}; font-weight: 700;'>{sentiment_value or sentiment_finding}</span></p>",
                        unsafe_allow_html=True,
                    )

        with col1:
            with st.container(border=True):
                st.markdown("🔗 **EXTRACTED URLs**")
                
                if not data['extracted_urls']:
                    st.markdown("<p style='color: #888; font-size: 14px;'>No URLs found in email.</p>", unsafe_allow_html=True)
                else:
                    for i, url_obj in enumerate(data['extracted_urls']):
                        badge_class = "badge-flagged" if url_obj['status'] == "DANGER" else "badge-safe"
                        status_text = url_obj['status']
                        full_url = url_obj.get('full_url', url_obj['url'])
                        safe_display_url = html.escape(url_obj['url'])
                        # Break auto-link detection for DANGER while keeping visual text readable.
                        danger_plain_text = safe_display_url.replace("://", "://&#8203;").replace(".", ".&#8203;")
                        
                        col_url, col_status = st.columns([3, 1])
                        with col_url:
                            if url_obj['status'] == "SAFE":
                                st.markdown(
                                    f"<a href='{full_url}' target='_blank' style='color: #00ff00; text-decoration: underline; word-break: break-all;'>{i+1}. {url_obj['url']}</a>",
                                    unsafe_allow_html=True
                                )
                            elif url_obj['status'] == "DANGER":
                                st.markdown(
                                    f"<span style='color: #888; word-break: break-all;'>{i+1}. {danger_plain_text}</span>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<span style='color: #888; text-decoration: line-through; word-break: break-all;'>{i+1}. {url_obj['url']}</span>",
                                    unsafe_allow_html=True
                                )
                        with col_status:
                            st.markdown(f"<span class='{badge_class}' style='display: inline-block;'>{status_text}</span>", unsafe_allow_html=True)

        with col2:
            with st.container(border=True):
                st.markdown("👁️ **VISION ANALYSIS**")
                vision_summary = data.get('vision_summary', {})
                similarity = vision_summary.get('similarity', 50)

                screenshot_path = None
                screenshot_caption = "Screenshot"
                vision_caption = "Brand Match"
                phishing_label = "LINKED WEBSITE"
                legitimate_label = "KNOWN LEGITIMATE WEBSITE"
                
                for url_obj in data.get('extracted_urls', []):
                    if url_obj.get('screenshot_path'):
                        screenshot_path = url_obj.get('screenshot_path')
                        if url_obj.get('status') == 'DANGER' or url_obj.get('status') == 'UNSAFE':
                            phishing_label = "PHISHING WEBSITE"
                            legitimate_label = url_obj.get('page_title') or "KNOWN LEGITIMATE WEBSITE"
                        break

                if screenshot_path:
                    similarity_description = "very low" if similarity < 20 else "low" if similarity < 40 else "moderate" if similarity < 70 else "high"
                    match_color = "#ff4b4b" if similarity < 40 else "#ffd43b" if similarity < 70 else "#00ff00"
                    st.markdown(
                        f"<p style='font-size: 12px; color: {match_color};'><em>Match: {similarity}% ({similarity_description})</em></p>",
                        unsafe_allow_html=True
                    )

                img_col1, img_col2, img_col3 = st.columns([2, 1, 2])
                with img_col1:
                    if screenshot_path:
                        st.image(screenshot_path, caption=phishing_label)
                    else:
                        st.image("https://via.placeholder.com/150x100/1e232f/ffffff?text=Image+1", caption=phishing_label)
                with img_col2:
                    st.markdown("<h3 style='text-align:center; margin-top:20px;'>VS</h3>", unsafe_allow_html=True)
                with img_col3:
                    if screenshot_path:
                        st.image(screenshot_path, caption=legitimate_label)
                    else:
                        st.image("https://via.placeholder.com/150x100/1e232f/ffffff?text=Image+2", caption=legitimate_label)

    st.write("<br>", unsafe_allow_html=True)
    if st.button("← Analyze Another Email", use_container_width=False):
        st.session_state.step = "input"
        st.rerun()