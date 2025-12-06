"""
Created By Page - Team Information
"""

import streamlit as st
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Created By - Team",
    page_icon="👥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .team-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .team-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .team-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 0;
    }
    
    .team-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .team-card:hover {
        transform: translateY(-10px);
        border-color: #667eea;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
    }
    
    .member-avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        margin: 0 auto 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .member-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .member-role {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.7);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .member-desc {
        text-align: center;
        color: rgba(255,255,255,0.8);
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1rem;
    }
    
    .skill-badge {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .contribution-card {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .contribution-card h3 {
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stats-card h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .stats-card p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .timeline-item {
        border-left: 3px solid #667eea;
        padding-left: 1.5rem;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 0;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #667eea;
    }
    
    .timeline-item h4 {
        color: #667eea;
        margin-bottom: 0.3rem;
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .social-btn {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .social-btn:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="team-header">
    <h1>👥 Meet Our Team</h1>
    <p>The minds behind the Energy Consumption Prediction System</p>
</div>
""", unsafe_allow_html=True)

# Team introduction
st.markdown("""
<div style="text-align: center; margin: 2rem 0; font-size: 1.2rem; color: rgba(255,255,255,0.8);">
    Our project combines advanced machine learning techniques with real-world applications 
    to address India's growing energy management needs.
</div>
""", unsafe_allow_html=True)

# Project statistics
st.markdown("## 📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stats-card">
        <h2>50K+</h2>
        <p>Data Points</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-card">
        <h2>96.21%</h2>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-card">
        <h2>28</h2>
        <p>States Covered</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stats-card">
        <h2>3000+</h2>
        <p>Lines of Code</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Team members
st.markdown("## 🌟 Team Members")

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="team-card">
        <div class="member-avatar">👨‍💻</div>
        <div class="member-name">Ekansh Singh</div>
        <div class="member-role">Reg. No. 102215107 (4NC2)</div>
        
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="team-card">
        <div class="member-avatar">👨‍💼</div>
        <div class="member-name">Chirag Bansal</div>
        <div class="member-role">Reg. No. 102215109 (4NC2)</div>
        
    </div>
    """, unsafe_allow_html=True)

# Row 2
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="team-card">
        <div class="member-avatar">👩‍🎨</div>
        <div class="member-name">Pratima</div>
        <div class="member-role">Reg. No. 102215324 (4NC2)</div>
        
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="team-card">
        <div class="member-avatar">👨‍🔬</div>
        <div class="member-name">Kanchan Saini</div>
        <div class="member-role">Reg. No. 102215322 (4NC2)</div>
        
        
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# Technologies used
st.markdown("## 🛠️ Technologies & Tools Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Deep Learning
    - TensorFlow 2.15
    - Keras
    - Scikit-learn
    - NumPy
    - Pandas
    """)

with col2:
    st.markdown("""
    ### Visualization
    - Streamlit
    - Plotly
    - Matplotlib
    - Seaborn
    """)

with col3:
    st.markdown("""
    ### Development
    - Python 3.8+
    - Jupyter Notebooks
    - Git & GitHub
    - VS Code
    """)

st.markdown("---")

# Acknowledgments
st.markdown("## 🙏 Acknowledgments")

st.markdown("""
<div style="background: rgba(102, 126, 234, 0.1); padding: 2rem; border-radius: 12px; border-left: 4px solid #667eea;">
    <h3 style="color: #667eea;">Special Thanks To:</h3>
    <ul style="color: rgba(255,255,255,0.8); line-height: 2;">
        <li><strong>Grid Controller of India Ltd.</strong> and <strong>Ministry of Power</strong> for providing the comprehensive energy consumption dataset</li>
        <li><strong>Our College Faculty (Dr. Gaganpreet Kaur & Dr. Deepak Rakesh Kumar)</strong> for guidance and support throughout the project</li>
        <li><strong>Open Source Community</strong> for amazing tools and libraries</li>
        <li><strong>Research Papers Authors</strong> whose work inspired our methodology</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Contact information
st.markdown("## 📞 Get In Touch")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 12px;">
        <h3>📧 Email</h3>
        <p style="color: #667eea;">esingh3_be22@thapar.edu</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 12px;">
        <h3>🔗 GitHub</h3>
        <p style="color: #667eea;">github.com/singh-ekansh</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 12px;">
        <h3>💼 LinkedIn</h3>
        <p style="color: #667eea;">linkedin.com/in/ekanshsinghin</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.6); margin-top: 3rem;'>
    <h3 style="color: #667eea;">⭐ If you like our work, please star us on GitHub!</h3>
    <p>Built with ❤️ by the Energy Prediction Team</p>
    <p>December 2025 | Deep Learning Project</p>
</div>
""", unsafe_allow_html=True)