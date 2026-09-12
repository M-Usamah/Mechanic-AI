import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import json
import requests
from pathlib import Path
import time
import threading
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

class GradioVehicleDiagnosis:
    def __init__(self):
        self.agent = VehicleDiagnosisAgent()
        self.chat_history = []
        
    def create_plots(self, df):
        """Create matplotlib plots with dark theme matching Tkinter version"""
        if df is None or df.empty:
            return None
            
        # Set dark theme for matplotlib - match Tkinter exactly
        plt.style.use('dark_background')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if 'Unnamed' not in col]
        
        if len(numeric_cols) == 0:
            # No numeric data, show empty dark background
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#1e293b')  # card color
            ax.set_facecolor('#1e293b')
            ax.axis('off')
        else:
            # Show up to 12 plots (6 rows x 2 cols) - matching Tkinter
            cols = 2
            rows = min(6, (len(numeric_cols) + 1) // 2)
            
            fig, axes = plt.subplots(rows, cols, figsize=(8, 12))
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
                
                if len(data) > 15:
                    # Dark theme line plot - exact Tkinter colors
                    x = np.arange(len(data))
                    y = data.values
                    ax.plot(x, y, color='#60a5fa', linewidth=2.5, alpha=0.9)
                    ax.fill_between(x, y, alpha=0.3, color='#3b82f6')
                else:
                    # Dark theme bar chart - exact Tkinter colors
                    ax.bar(range(len(data)), data.values, color='#3b82f6', alpha=0.8, edgecolor='#60a5fa')
                
                # Dark theme styling - exact Tkinter styling
                ax.set_title(col[:35], fontsize=9, fontweight='bold', pad=6, color='#f1f5f9')
                ax.set_xlabel('Sample', fontsize=7, color='#94a3b8')
                ax.set_ylabel('Value', fontsize=7, color='#94a3b8')
                ax.tick_params(labelsize=6, colors='#94a3b8')
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
        
        plt.tight_layout(pad=1.5)  # Match Tkinter padding
        
        # Convert to base64 for Gradio
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#1e293b', 
                   edgecolor='none', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
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
    
    
    def on_file_selected(self, file):
        """Handle file selection"""
        if file is None:
            return gr.Button(visible=False), None, "No file selected - Click browse to get started", []
        
        try:
            # Load and plot data
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            else:
                df = pd.read_excel(file.name)
            
            # Generate plots
            plot_image = self.create_plots(df)
            
            # Update file info
            vehicle_name = Path(file.name).stem
            file_info = f"File: {vehicle_name} ({df.shape[0]} rows × {df.shape[1]} columns)"
            
            # Welcome message
            welcome_msg = [(None, f"File loaded: {vehicle_name}\nClick 'Run AI Diagnosis' to analyze this vehicle.")]
            
            return gr.Button(visible=True), plot_image, file_info, welcome_msg
            
        except Exception as e:
            return gr.Button(visible=False), None, f"Error loading file: {str(e)}", []
    
    def diagnose_vehicle(self, file):
        """Main diagnosis function"""
        if file is None:
            return [], None, "Please upload a vehicle data file first."
        
        try:
            # Load data
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            else:
                df = pd.read_excel(file.name)
            
            # Generate plots
            plot_image = self.create_plots(df)
            
            # Perform diagnosis
            diagnosis = self.agent.diagnose_vehicle(file.name)
            formatted_diagnosis = self.format_diagnosis_response(diagnosis)
            
            # Format as chat messages
            vehicle_name = Path(file.name).stem
            messages = [
                (f"Please analyze {vehicle_name}", None),
                (None, f"Analyzing vehicle diagnostic data..."),
                (None, "Processing data and running AI analysis. This may take 30-60 seconds..."),
                (None, f"DIAGNOSIS REPORT: {vehicle_name.upper()}\n" + "-"*50 + "\n\n" + formatted_diagnosis),
                (None, "Analysis complete. You can now ask me specific questions about your vehicle's condition, recommended actions, or any concerns you may have.")
            ]
            
            return messages, plot_image, "✅ Diagnosis complete"
            
        except Exception as e:
            error_msg = [(None, f"❌ Error analyzing file: {str(e)}")]
            return error_msg, None, f"❌ Error: {str(e)}"
    
    def chat_with_ai(self, message, history):
        """Handle chat interactions"""
        if not message.strip():
            return history, ""
            
        # Add user message to history
        history.append((message, None))
        
        # Get AI response (simplified for chat)
        try:
            # For chat, we'll use a simpler prompt
            response = self.agent.query_ollama(f"User question: {message}\n\nPlease provide a helpful response about vehicle diagnostics.")
            formatted_response = self.format_diagnosis_response(response)
            
            # Update history
            history[-1] = (message, formatted_response)
            
        except Exception as e:
            history[-1] = (message, f"Sorry, I encountered an error: {str(e)}")
        
        return history, ""
    
    def clear_chat(self):
        """Clear chat and show welcome message"""
        return [(None, "Hello! I'm your AI Vehicle Diagnosis Assistant. Upload a vehicle data file to get started!")]
    
    def create_interface(self):
        """Create the Gradio interface matching Tkinter layout exactly"""
        
        # Custom CSS for dark theme - exact Tkinter colors
        css = f"""
        .gradio-container {{
            background: {DARK_COLORS['background']} !important;
            color: {DARK_COLORS['text']} !important;
            font-family: 'Segoe UI', sans-serif !important;
        }}
        .gr-button {{
            background: {DARK_COLORS['primary_light']} !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 5px !important;
        }}
        .gr-button:hover {{
            background: {DARK_COLORS['primary']} !important;
        }}
        .gr-textbox, .gr-upload {{
            background: {DARK_COLORS['input_bg']} !important;
            border: 1px solid {DARK_COLORS['border']} !important;
            color: {DARK_COLORS['text']} !important;
            border-radius: 5px !important;
        }}
        .gr-panel {{
            background: {DARK_COLORS['card']} !important;
            border: 1px solid {DARK_COLORS['border']} !important;
            border-radius: 8px !important;
        }}
        .chatbot {{
            background: {DARK_COLORS['chat_bg']} !important;
        }}
        """
        
        with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue")) as interface:
            # Header - matching Tkinter gradient style
            gr.Markdown(
                f"""
                <div style="text-align: center; padding: 30px; background: {DARK_COLORS['primary']}; border-radius: 0px; margin: -20px -20px 20px -20px;">
                    <h1 style="color: white; margin: 0; font-size: 28px; font-weight: bold;">AI Vehicle Diagnosis Assistant</h1>
                    <p style="color: #93c5fd; margin: 10px 0 0 0; font-size: 11px;">Powered by Advanced AI  •  Real-time Analysis  •  Expert Insights</p>
                </div>
                """,
                elem_id="header"
            )
            
            # Main container matching Tkinter layout
            with gr.Row():
                # Left side - Controls (Top) and Visualization (Bottom)
                with gr.Column(scale=1):
                    # Top Panel - Controls (matching Tkinter top panel)
                    gr.Markdown(f"### Vehicle Selection & Analysis", elem_id="control_title")
                    
                    # Status and file info row
                    with gr.Row():
                        status_output = gr.Textbox(
                            label="Status: Checking...",
                            interactive=False,
                            elem_id="status",
                            scale=1
                        )
                        
                        file_info = gr.Textbox(
                            label="File Info",
                            value="No file selected - Click browse to get started",
                            interactive=False,
                            elem_id="file_info",
                            scale=3
                        )
                    
                    # Buttons row
                    with gr.Row():
                        file_input = gr.File(
                            label="Browse Files",
                            file_types=[".xlsx", ".xls", ".csv"],
                            elem_id="file_input",
                            scale=1
                        )
                        
                        diagnose_btn = gr.Button(
                            "[D] Run AI Diagnosis", 
                            variant="primary",
                            elem_id="diagnose_btn",
                            scale=1,
                            visible=False
                        )
                        
                        clear_btn = gr.Button(
                            "[C] Clear Chat",
                            elem_id="clear_btn",
                            scale=1
                        )
                    
                    # Visualization Panel (matches Tkinter left panel)
                    gr.Markdown(f"### Data Visualization", elem_id="viz_title")
                    plot_output = gr.Image(
                        label="Vehicle Data Charts",
                        elem_id="plot_output",
                        height=400
                    )
                
                # Right side - Chat interface (matches Tkinter right panel)
                with gr.Column(scale=2):  # 2x wider like Tkinter
                    gr.Markdown(f"### AI Chat Assistant", elem_id="chat_title")
                    
                    chatbot = gr.Chatbot(
                        label="AI Assistant",
                        height=600,
                        elem_id="chatbot"
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask me anything about vehicle diagnostics...",
                            label="Your message",
                            elem_id="chat_input",
                            lines=2,
                            scale=4
                        )
                        chat_btn = gr.Button("Send\n[S]", variant="primary", elem_id="chat_btn", scale=1)
            
            # Event handlers
            file_input.change(
                fn=self.on_file_selected,
                inputs=[file_input],
                outputs=[diagnose_btn, plot_output, file_info, chatbot]
            )
            
            diagnose_btn.click(
                fn=self.diagnose_vehicle,
                inputs=[file_input],
                outputs=[chatbot, plot_output, status_output]
            )
            
            chat_btn.click(
                fn=self.chat_with_ai,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            chat_input.submit(
                fn=self.chat_with_ai,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot]
            )
        
        return interface

def main():
    """Main function to launch the Gradio interface"""
    app = GradioVehicleDiagnosis()
    interface = app.create_interface()
    
    print("🚗 Starting AI Vehicle Diagnosis Assistant...")
    print("📊 Gradio interface will open in your browser")
    print("🔗 If it doesn't open automatically, check the terminal for the local URL")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
