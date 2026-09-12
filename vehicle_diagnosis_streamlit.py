import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from pathlib import Path
import time
from vehicle_diagnosis_agent import VehicleDiagnosisAgent

# Dark theme colors matching the original design
DARK_COLORS = {
    'background': '#0f172a',
    'card': '#1e293b', 
    'primary': '#3b82f6',
    'primary_light': '#60a5fa',
    'secondary': '#64748b',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'border': '#334155',
    'input_bg': '#334155',
    'chat_bg': '#1e293b'
}

# Configure Streamlit page
st.set_page_config(
    page_title="AI Vehicle Diagnosis Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown(f"""
<style>
    .main {{
        background-color: {DARK_COLORS['background']};
        color: {DARK_COLORS['text']};
    }}
    .stApp {{
        background-color: {DARK_COLORS['background']};
    }}
    .stTextInput > div > div > input {{
        background-color: {DARK_COLORS['input_bg']};
        color: {DARK_COLORS['text']};
        border: 1px solid {DARK_COLORS['border']};
    }}
    .stSelectbox > div > div > select {{
        background-color: {DARK_COLORS['input_bg']};
        color: {DARK_COLORS['text']};
        border: 1px solid {DARK_COLORS['border']};
    }}
    .stButton > button {{
        background-color: {DARK_COLORS['primary_light']};
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: {DARK_COLORS['primary']};
    }}
    .stFileUploader > div {{
        background-color: {DARK_COLORS['card']};
        border: 1px solid {DARK_COLORS['border']};
        border-radius: 5px;
    }}
    .stMarkdown {{
        color: {DARK_COLORS['text']};
    }}
    .header {{
        background: linear-gradient(135deg, {DARK_COLORS['primary']}, {DARK_COLORS['primary_light']});
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .card {{
        background-color: {DARK_COLORS['card']};
        padding: 20px;
        border-radius: 8px;
        border: 1px solid {DARK_COLORS['border']};
        margin: 10px 0;
    }}
</style>
""", unsafe_allow_html=True)

class StreamlitVehicleDiagnosis:
    def __init__(self):
        if 'agent' not in st.session_state:
            st.session_state.agent = VehicleDiagnosisAgent()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'diagnosis_complete' not in st.session_state:
            st.session_state.diagnosis_complete = False
        if 'show_plots' not in st.session_state:
            st.session_state.show_plots = False
        if 'diagnosis_result' not in st.session_state:
            st.session_state.diagnosis_result = None
        if 'current_vehicle' not in st.session_state:
            st.session_state.current_vehicle = None
    
    def create_plots(self, df):
        """Create matplotlib plots with dark theme matching Tkinter version"""
        if df is None or df.empty:
            return None
            
        try:
            # Set dark theme for matplotlib - match Tkinter exactly
            plt.style.use('dark_background')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if 'Unnamed' not in str(col)]
            
            if len(numeric_cols) == 0:
                # No numeric data, show empty dark background
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#1e293b')  # card color
                ax.set_facecolor('#1e293b')
                ax.axis('off')
                ax.text(0.5, 0.5, 'No numeric data to visualize', 
                       ha='center', va='center', fontsize=14, color='#94a3b8')
                plt.tight_layout()
                return fig
            
            # Show up to 12 plots (6 rows x 2 cols) - matching Tkinter
            cols = 2
            rows = min(6, (len(numeric_cols) + 1) // 2)
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 30))
            fig.patch.set_facecolor('#1e293b')  # card color
            
            # Create plots for all available numeric columns (up to 12)
            for idx, col in enumerate(numeric_cols[:12], 1):
                if rows == 1:
                    ax = axes[idx-1] if cols > 1 else axes
                else:
                    row = (idx - 1) // cols
                    col_idx = (idx - 1) % cols
                    ax = axes[row, col_idx]
                
                ax.set_facecolor('#0f172a')  # background color
                data = df[col].dropna()
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           fontsize=10, color='#94a3b8')
                    ax.set_title(col[:35], fontsize=9, fontweight='bold', pad=6, color='#f1f5f9')
                    ax.axis('off')
                    continue
                
                if len(data) > 15:
                    # Dark theme line plot - enhanced visibility
                    x = np.arange(len(data))
                    y = data.values
                    ax.plot(x, y, color='#60a5fa', linewidth=4.0, alpha=0.9)
                    ax.fill_between(x, y, alpha=0.3, color='#3b82f6')
                else:
                    # Dark theme bar chart - enhanced visibility
                    ax.bar(range(len(data)), data.values, color='#3b82f6', alpha=0.8, edgecolor='#60a5fa', linewidth=2.0)
                
                # Dark theme styling - enhanced for better visibility
                ax.set_title(col[:35], fontsize=16, fontweight='bold', pad=12, color='#f1f5f9')
                ax.set_xlabel('Sample', fontsize=14, color='#94a3b8')
                ax.set_ylabel('Value', fontsize=14, color='#94a3b8')
                ax.tick_params(labelsize=12, colors='#94a3b8')
                ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#475569')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#475569')
                ax.spines['bottom'].set_color('#475569')
            
            # Hide unused subplots
            if rows > 1:
                for idx in range(len(numeric_cols) + 1, rows * cols + 1):
                    row = (idx - 1) // cols
                    col_idx = (idx - 1) % cols
                    if row < rows and col_idx < cols:
                        axes[row, col_idx].set_visible(False)
            
            plt.tight_layout(pad=5.0)  # Maximum padding for better visibility
            return fig
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            return None
    
    def format_diagnosis_response(self, response):
        """Format AI response with proper styling"""
        if not response:
            return "No diagnosis available."
            
        # Split response into sections
        sections = response.split('\n')
        formatted = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if section.startswith('**') and section.endswith('**'):
                # Bold headers
                formatted += f"**{section[2:-2]}**\n\n"
            elif section.startswith('- ') or section.startswith('• '):
                # Bullet points
                formatted += f"• {section[2:]}\n"
            else:
                formatted += f"{section}\n"
        
        return formatted
    
    def add_chat_message(self, message, is_ai=True):
        """Add message to chat history"""
        st.session_state.chat_history.append({
            'message': message,
            'is_ai': is_ai,
            'timestamp': time.time()
        })
    
    def display_chat_history(self):
        """Display chat history"""
        for msg in st.session_state.chat_history:
            if msg['is_ai']:
                with st.chat_message("assistant"):
                    st.write(msg['message'])
            else:
                with st.chat_message("user"):
                    st.write(msg['message'])
    
    def run(self):
        # Header
        st.markdown("""
        <div class="header">
            <h1 style="color: white; margin: 0; font-size: 28px;">AI Vehicle Diagnosis Assistant</h1>
            <p style="color: #93c5fd; margin: 10px 0 0 0; font-size: 14px;">Powered by Advanced AI • Real-time Analysis • Expert Insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top Section - Controls (Full Width) - Matching Tkinter exactly
        st.markdown(f"""
        <div class="card">
            <h3 style="color: {DARK_COLORS['text']}; margin-top: 0;">Vehicle Selection & Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Controls row
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            # Status and file info
            st.markdown("**Status:** Online ✅")
            if 'current_file' in st.session_state and st.session_state.current_file:
                st.markdown(f"**File:** {st.session_state.current_file}")
            else:
                st.markdown("**File:** No file selected - Click browse to get started")
        
        with col2:
            # File upload
            uploaded_file = st.file_uploader(
                "Browse Files",
                type=['xlsx', 'xls', 'csv'],
                help="Upload vehicle diagnostic data file",
                key="file_uploader"
            )
        
        with col3:
            # Diagnosis button
            diagnosis_btn = st.button("🔍 Run AI Diagnosis", type="primary", disabled=uploaded_file is None)
        
        with col4:
            # Clear chat button
            clear_btn = st.button("🗑️ Clear Chat")
        
        # Handle file upload
        if uploaded_file:
            vehicle_name = Path(uploaded_file.name).stem
            
            # Clear chat and diagnosis if new file is uploaded
            if st.session_state.current_vehicle != vehicle_name:
                st.session_state.chat_history = []
                st.session_state.diagnosis_result = None
                st.session_state.diagnosis_complete = False
                st.session_state.current_vehicle = vehicle_name
            
            st.session_state.current_file = vehicle_name
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.current_data = df
                st.session_state.show_plots = True  # Enable plots display
                st.success(f"✅ File loaded: {vehicle_name} ({df.shape[0]} rows × {df.shape[1]} columns)")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        # Handle diagnosis
        if diagnosis_btn and uploaded_file:
            with st.spinner("🤖 Analyzing vehicle diagnostic data... This may take 30-60 seconds..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Perform diagnosis
                    diagnosis = st.session_state.agent.diagnose_vehicle(temp_path)
                    
                    if diagnosis:
                        # Save diagnosis to session state
                        st.session_state.diagnosis_result = diagnosis
                        
                        formatted_diagnosis = self.format_diagnosis_response(diagnosis)
                        
                        # Add to chat
                        self.add_chat_message(f"DIAGNOSIS REPORT: {vehicle_name.upper()}\n" + "-"*50 + "\n\n" + formatted_diagnosis, is_ai=True)
                        self.add_chat_message("Analysis complete. You can now ask me specific questions about your vehicle's condition.", is_ai=True)
                        
                        st.session_state.diagnosis_complete = True
                        st.success("✅ Diagnosis complete!")
                    else:
                        self.add_chat_message("❌ Failed to generate diagnosis. Please check your API key and connection.", is_ai=True)
                        st.error("Failed to generate diagnosis")
                    
                except Exception as e:
                    self.add_chat_message(f"❌ Error analyzing file: {str(e)}", is_ai=True)
                    st.error(f"Error: {str(e)}")
            
            # Rerun to show updated chat
            st.rerun()
        
        # Handle clear chat
        if clear_btn:
            st.session_state.chat_history = []
            st.session_state.diagnosis_result = None
            st.session_state.diagnosis_complete = False
            st.rerun()
        
        # Bottom Section - Left: Plots, Right: Chat (Matching Tkinter exactly)
        col_left, col_right = st.columns([1, 2])  # Chat is 2x wider
        
        with col_left:
            # Visualization Panel (matches Tkinter left panel)
            st.markdown(f"""
            <div class="card">
                <h3 style="color: {DARK_COLORS['text']}; margin-top: 0;">Data Visualization</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a container for plots that persists
            plot_container = st.container()
            
            with plot_container:
                if st.session_state.current_data is not None and st.session_state.show_plots:
                    # Always show plots if data exists and plots are enabled
                    try:
                        fig = self.create_plots(st.session_state.current_data)
                        if fig:
                            # Use st.pyplot with proper configuration for better visibility
                            st.pyplot(fig, clear_figure=True, width='stretch', dpi=150)
                            plt.close(fig)
                        else:
                            st.warning("Unable to generate plots from the data.")
                    except Exception as e:
                        st.error(f"Error creating plots: {str(e)}")
                        st.markdown("""
                        <div style="text-align: center; padding: 50px; color: #94a3b8;">
                            <h4>[CHART ERROR]</h4>
                            <p>Unable to visualize data</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 50px; color: #94a3b8;">
                        <h4>[CHART]</h4>
                        <p>Select a vehicle file<br>to visualize data</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_right:
            # Chat Section (matches Tkinter right panel)
            st.markdown(f"""
            <div class="card">
                <h3 style="color: {DARK_COLORS['text']}; margin-top: 0;">AI Chat Assistant</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display chat history
            if st.session_state.chat_history:
                self.display_chat_history()
            else:
                st.markdown("Hello! I'm your AI Vehicle Diagnosis Assistant. Upload a vehicle data file to get started!")
                st.markdown("AI system is ready! I can analyze vehicle diagnostic data and answer your questions.")
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about vehicle diagnostics..."):
                # Add user message
                self.add_chat_message(prompt, is_ai=False)
                
                # Get AI response
                with st.spinner("Thinking..."):
                    try:
                        # Build context with diagnosis and previous chat
                        context = ""
                        
                        # Add diagnosis if available
                        if st.session_state.diagnosis_result:
                            vehicle = st.session_state.current_vehicle or "the vehicle"
                            context += f"VEHICLE DIAGNOSIS FOR {vehicle.upper()}:\n{st.session_state.diagnosis_result}\n\n"
                        
                        # Add recent chat history (last 5 exchanges)
                        recent_messages = st.session_state.chat_history[-10:] if len(st.session_state.chat_history) > 0 else []
                        if recent_messages:
                            context += "PREVIOUS CONVERSATION:\n"
                            for msg in recent_messages:
                                role = "User" if not msg['is_ai'] else "Assistant"
                                # Skip the full diagnosis report in history
                                if "DIAGNOSIS REPORT:" not in msg['message']:
                                    context += f"{role}: {msg['message'][:200]}\n"
                            context += "\n"
                        
                        # Add current question
                        full_prompt = f"{context}USER'S CURRENT QUESTION: {prompt}\n\nPlease provide a helpful, specific response based on the diagnosis and conversation context above."
                        
                        response = st.session_state.agent.chat_with_ai(full_prompt)
                        formatted_response = self.format_diagnosis_response(response)
                        self.add_chat_message(formatted_response, is_ai=True)
                    except Exception as e:
                        self.add_chat_message(f"Sorry, I encountered an error: {str(e)}", is_ai=True)
                
                st.rerun()

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitVehicleDiagnosis()
    app.run()

if __name__ == "__main__":
    main()