import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="EduNavigator — AI Career Guidance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2a3a 0%, #2E86AB 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: rgba(255,255,255,0.3) !important;
    }
    
    /* Header */
    h1 { 
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8em !important;
        font-weight: 700 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 24px 16px;
        border-radius: 16px;
        text-align: center;
        border-top: 4px solid #2E86AB;
        box-shadow: 0 4px 16px rgba(46,134,171,0.12);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-number { 
        font-size: 2.4em; 
        font-weight: 700; 
        color: #2E86AB;
        line-height: 1.1;
    }
    .metric-label { 
        color: #6c757d; 
        font-size: 0.82em; 
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a6b8a 0%, #2E86AB 50%, #A23B72 100%);
        color: white;
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(46,134,171,0.25);
        position: relative;
        overflow: hidden;
    }
    .result-box::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
    }
    .result-title { 
        font-size: 0.9em; 
        opacity: 0.85; 
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .result-career { 
        font-size: 2em; 
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Section headers */
    h3 {
        color: #1a2a3a !important;
        border-left: 4px solid #2E86AB;
        padding-left: 12px;
        margin: 24px 0 16px !important;
    }
    
    /* Info box */
    .stInfo {
        background: #e8f4f8 !important;
        border-left: 4px solid #2E86AB !important;
        border-radius: 8px !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2E86AB, #A23B72) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(46,134,171,0.3) !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(46,134,171,0.4) !important;
    }

    /* Divider */
    hr { border-color: #dee2e6 !important; margin: 32px 0 !important; }

    /* Footer */
    .footer-box {
        background: #1a2a3a;
        color: #8b9bb4;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        font-size: 0.85em;
        line-height: 2;
    }
    .footer-box a { color: #5bbcdc; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA
# ============================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'Math & Physics':        np.random.randint(40, 100, n),
        'Biology & Chemistry':   np.random.randint(40, 100, n),
        'History & Geography':   np.random.randint(40, 100, n),
        'Languages & Literature':np.random.randint(40, 100, n),
        'Arts & Design':         np.random.randint(30, 100, n)
    })

    careers = {
        'Math & Physics':         'IT & Engineering',
        'Biology & Chemistry':    'Medicine & Biology',
        'History & Geography':    'Law & Social Sciences',
        'Languages & Literature': 'Philology & Education',
        'Arts & Design':          'Design & Creative Arts'
    }
    df['Career'] = df.idxmax(axis=1).map(careers)
    return df

df = load_data()

FEATURES = ['Math & Physics', 'Biology & Chemistry',
            'History & Geography', 'Languages & Literature', 'Arts & Design']

CAREER_ICONS = {
    'IT & Engineering':       '💻',
    'Medicine & Biology':     '🏥',
    'Law & Social Sciences':  '⚖️',
    'Philology & Education':  '📚',
    'Design & Creative Arts': '🎨'
}

CAREER_DESC = {
    'IT & Engineering':       'Software Developer, Data Scientist, Engineer, Architect',
    'Medicine & Biology':     'Doctor, Pharmacist, Biologist, Veterinarian',
    'Law & Social Sciences':  'Lawyer, Historian, Economist, Diplomat',
    'Philology & Education':  'Teacher, Translator, Journalist, Writer',
    'Design & Creative Arts': 'Graphic Designer, Animator, Musician, Artist'
}

# ============================================
# MODEL
# ============================================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df['Career']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, accuracy = train_model(df)

# ============================================
# PDF GENERATOR
# ============================================
def generate_pdf(student_name, scores, prediction, probabilities):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', fontSize=20, fontName='Helvetica-Bold',
                                  textColor=colors.HexColor('#2E86AB'),
                                  spaceAfter=6, alignment=1)
    info_style = ParagraphStyle('info', fontSize=11, fontName='Helvetica',
                                 textColor=colors.grey, alignment=1)
    result_style = ParagraphStyle('result', fontSize=16, fontName='Helvetica-Bold',
                                   textColor=colors.HexColor('#A23B72'),
                                   alignment=1, spaceAfter=6)
    header_style = ParagraphStyle('header', fontSize=13, fontName='Helvetica-Bold',
                                   textColor=colors.HexColor('#2E86AB'), spaceAfter=8)

    story.append(Paragraph("EduNavigator", title_style))
    story.append(Paragraph("AI-Powered Career Guidance Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Student: {student_name} | Date: {datetime.date.today().strftime('%B %d, %Y')}",
        info_style))
    story.append(Spacer(1, 20))

    icon = CAREER_ICONS.get(prediction, '🎯')
    story.append(Paragraph("Recommended Career Path:", info_style))
    story.append(Paragraph(f"{icon}  {prediction}", result_style))
    story.append(Paragraph(f"Possible roles: {CAREER_DESC.get(prediction, '')}",
                            ParagraphStyle('roles', fontSize=10,
                                           textColor=colors.grey, alignment=1)))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Academic Profile", header_style))
    score_data = [['Subject Area', 'Score', 'Level']] + [
        [subj, str(score),
         'High' if score >= 70 else 'Medium' if score >= 50 else 'Low']
        for subj, score in zip(FEATURES, scores)
    ]
    table = Table(score_data, colWidths=[3*inch, 1.2*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Career Match Probabilities", header_style))
    prob_data = [['Career Path', 'Match %']] + [
        [f"{CAREER_ICONS.get(c,'')} {c}", f"{p*100:.1f}%"]
        for c, p in sorted(zip(model.classes_, probabilities),
                           key=lambda x: x[1], reverse=True)
    ]
    prob_table = Table(prob_data, colWidths=[3.5*inch, 1.2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))

    footer_style = ParagraphStyle('footer', fontSize=9,
                                   textColor=colors.grey, alignment=1)
    story.append(Paragraph(
        "Generated by EduNavigator | Orleu National Centre for Professional Development, Kazakhstan",
        footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 🎓 EduNavigator")
    st.markdown("*AI Career Guidance System*")
    st.markdown("---")
    student_name = st.text_input("👤 Student name:", placeholder="Enter full name...")
    st.markdown("### 📊 Academic Scores (1–100)")
    math_score = st.slider("💻 Math & Physics", 1, 100, 75)
    bio_score  = st.slider("🏥 Biology & Chemistry", 1, 100, 50)
    hist_score = st.slider("⚖️ History & Geography", 1, 100, 40)
    lit_score  = st.slider("📚 Languages & Literature", 1, 100, 60)
    art_score  = st.slider("🎨 Arts & Design", 1, 100, 30)
    st.markdown("---")
    st.markdown(f"**Model accuracy:** `{accuracy*100:.0f}%`")
    st.markdown(f"**Algorithm:** `Random Forest (200 trees)`")

# ============================================
# MAIN PAGE
# ============================================
st.markdown("# 🎓 EduNavigator")
st.markdown("### AI-Powered Career Guidance System for School Students")
st.markdown("*Capstone project · Orleu National Centre for Professional Development · Kazakhstan*")
st.markdown("---")

# Metrics
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in zip(
    [c1, c2, c3, c4],
    [len(df), f"{accuracy*100:.0f}%", len(FEATURES), "RF"],
    ["Training samples", "Model accuracy", "Feature areas", "Algorithm"]
):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-number">{num}</div>
        <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction
scores = [math_score, bio_score, hist_score, lit_score, art_score]
prediction = model.predict([scores])[0]
probabilities = model.predict_proba([scores])[0]
icon = CAREER_ICONS.get(prediction, '🎯')

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### 🔮 Career Prediction")
    st.markdown(f"""<div class="result-box">
        <div class="result-title">Recommended career path</div>
        <div class="result-career">{icon} {prediction}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"**Possible roles:** {CAREER_DESC.get(prediction, '')}")
    st.markdown("<br>", unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        'Career': [f"{CAREER_ICONS.get(c,'')} {c}" for c in model.classes_],
        'Probability': np.round(probabilities * 100, 1)
    }).sort_values('Probability', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ['#2E86AB' if p == prob_df['Probability'].max()
                  else '#A8DADC' for p in prob_df['Probability']]
    bars = ax.barh(prob_df['Career'], prob_df['Probability'], color=bar_colors)
    ax.set_xlabel('Match Probability (%)')
    ax.set_title('Career Path Match Analysis', fontweight='bold', pad=12)
    for bar, val in zip(bars, prob_df['Probability']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=10)
    ax.set_xlim(0, 110)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

with col_r:
    st.markdown("### 📊 Academic Profile")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    labels = ['Math &\nPhysics', 'Biology &\nChemistry',
              'History &\nGeography', 'Languages &\nLiterature', 'Arts &\nDesign']
    bar_cols = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    bars2 = ax2.bar(labels, scores, color=bar_cols, edgecolor='white',
                    linewidth=1.5, width=0.6)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel('Score')
    ax2.set_title("Student's Academic Profile", fontweight='bold', pad=12)
    ax2.axhline(70, color='green', linestyle='--', alpha=0.4, label='High (70+)')
    ax2.axhline(50, color='orange', linestyle='--', alpha=0.4, label='Medium (50+)')
    for bar, val in zip(bars2, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig2)

# Analytics
st.markdown("---")
st.markdown("### 📈 Dataset Analytics")
ca, cb = st.columns(2)

with ca:
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    counts = df['Career'].value_counts()
    ax3.pie(counts,
            labels=[f"{CAREER_ICONS.get(c,'')} {c}" for c in counts.index],
            autopct='%1.1f%%',
            colors=['#2E86AB','#A23B72','#F18F01','#C73E1D','#3B1F2B'],
            startangle=90)
    ax3.set_title('Career Distribution in Dataset', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)

with cb:
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax4, linewidths=0.5)
    ax4.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig4)

# PDF
st.markdown("---")
st.markdown("### 📄 Download Report")
if student_name:
    pdf = generate_pdf(student_name, scores, prediction, probabilities)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf,
        file_name=f"EduNavigator_{student_name.replace(' ','_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
else:
    st.info("👈 Enter student name in the sidebar to download the PDF report")

st.markdown("---")
st.markdown("""
<div class='footer-box'>
    🎓 <strong style='color:white'>EduNavigator</strong> · AI-Powered Career Guidance System<br>
    Built with Python · Scikit-learn · Streamlit · ReportLab<br>
    Orleu National Centre for Professional Development · Kazakhstan 🇰🇿<br>
    <a href='https://github.com/mikushkan/edunavigator'>GitHub</a> ·
    <a href='https://mikushkan.github.io/edunavigator/'>Landing page</a> ·
    <a href='https://mektep.streamlit.app'>Live app</a>
</div>
""", unsafe_allow_html=True)
