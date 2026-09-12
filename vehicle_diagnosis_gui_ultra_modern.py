import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from vehicle_diagnosis_agent import OllamaManager, VehicleDiagnosisAgent
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time
from PIL import Image, ImageTk
import os


class ChatMessage(tk.Frame):
    """Dark theme chat message bubble with robot icons"""
    def __init__(self, parent, message, is_ai=True, bg_color='#1e293b', ai_avatar_photo=None, **kwargs):
        super().__init__(parent, bg=bg_color, **kwargs)
        
        if is_ai:
            # AI message (left aligned, blue)
            bubble_frame = tk.Frame(self, bg=bg_color)
            bubble_frame.pack(anchor='w', padx=20, pady=5)
            
            # Avatar - use robot icon if available, otherwise text
            if ai_avatar_photo:
                avatar = tk.Label(bubble_frame, image=ai_avatar_photo, bg=bg_color)
                avatar.pack(side='left', padx=(0, 10))
            else:
                avatar = tk.Label(bubble_frame, text="[AI]", font=('Segoe UI', 12, 'bold'), bg=bg_color, fg='#60a5fa')
                avatar.pack(side='left', padx=(0, 10))
            
            # Message bubble
            message_frame = tk.Frame(bubble_frame, bg='#1e40af', bd=0, relief='flat')
            message_frame.pack(side='left')
            
            message_label = tk.Label(
                message_frame,
                text=message,
                font=('Segoe UI', 11),
                bg='#1e40af',
                fg='white',
                wraplength=600,
                justify='left',
                padx=20,
                pady=15
            )
            message_label.pack()
        else:
            # User message (right aligned, gray)
            bubble_frame = tk.Frame(self, bg=bg_color)
            bubble_frame.pack(anchor='e', padx=20, pady=5)
            
            # Message bubble
            message_frame = tk.Frame(bubble_frame, bg='#475569', bd=0, relief='flat')
            message_frame.pack(side='right')
            
            message_label = tk.Label(
                message_frame,
                text=message,
                font=('Segoe UI', 11),
                bg='#475569',
                fg='#f1f5f9',
                wraplength=600,
                justify='left',
                padx=20,
                pady=15
            )
            message_label.pack()
            
            # Avatar
            avatar = tk.Label(bubble_frame, text="[U]", font=('Segoe UI', 12, 'bold'), bg=bg_color, fg='#94a3b8')
            avatar.pack(side='right', padx=(10, 0))


class UltraModernVehicleDiagnosisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vehicle Diagnosis - Ultra Modern")
        self.root.geometry("1500x950")
        
        # Dark theme color scheme
        self.colors = {
            'primary': '#3b82f6',
            'primary_light': '#60a5fa',
            'secondary': '#8b5cf6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'background': '#0f172a',      # Dark navy
            'card': '#1e293b',            # Dark gray
            'text': '#f1f5f9',            # Light text
            'text_light': '#94a3b8',      # Medium gray text
            'chat_bg': '#1e293b',         # Dark chat background
            'input_bg': '#334155',        # Input background
            'bubble_ai': '#1e40af',       # AI bubble blue
            'bubble_user': '#475569'      # User bubble gray
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Load robot icon for application
        self.load_robot_icon()
        
        # Set window icon if robot icon is available
        if hasattr(self, 'robot_icon_photo') and self.robot_icon_photo:
            self.root.iconphoto(True, self.robot_icon_photo)
        
        self.agent = None
        self.selected_file = None
        self.current_data = None
        self.chat_history = []
        
        self.setup_ui()
        self.check_ollama_status()
    
    def load_robot_icon(self):
        """Load robot icon and create custom UI elements"""
        try:
            if os.path.exists("titel.png"):
                # Load the robot icon
                self.robot_icon = Image.open("titel.png")
                self.robot_icon = self.robot_icon.resize((32, 32), Image.Resampling.LANCZOS)
                self.robot_icon_photo = ImageTk.PhotoImage(self.robot_icon)
                
                # Create smaller versions for avatars
                self.ai_avatar_icon = self.robot_icon.resize((24, 24), Image.Resampling.LANCZOS)
                self.ai_avatar_photo = ImageTk.PhotoImage(self.ai_avatar_icon)
                
                print("🤖 Robot icon loaded successfully")
            else:
                self.robot_icon_photo = None
                self.ai_avatar_photo = None
                print("⚠️ Robot icon not found, using text avatars")
        except Exception as e:
            self.robot_icon_photo = None
            self.ai_avatar_photo = None
            print(f"⚠️ Could not load robot icon: {e}")
    
    def setup_ui(self):
        # Modern gradient header
        self.header = tk.Canvas(self.root, height=120, bg=self.colors['primary'], highlightthickness=0)
        self.header.pack(fill='x')
        
        # Header content with dynamic centering
        self.header_text1 = self.header.create_text(400, 40, text="AI Vehicle Diagnosis Assistant", 
                          font=('Segoe UI', 28, 'bold'), fill='white', anchor='center')
        self.header_text2 = self.header.create_text(400, 80, text="Powered by Advanced AI  •  Real-time Analysis  •  Expert Insights",
                          font=('Segoe UI', 11), fill='#93c5fd', anchor='center')
        
        # Bind resize event to center header text
        self.header.bind('<Configure>', self.center_header_text)
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill='both', expand=True, padx=25, pady=20)
        
        # Three panel layout
        # Top panel - Controls
        top_panel = tk.Frame(main_container, bg=self.colors['background'], height=150)
        top_panel.pack(fill='x', pady=(0, 15))
        top_panel.pack_propagate(False)
        
        # Bottom container for viz and chat (swapped) - RESPONSIVE
        bottom_container = tk.Frame(main_container, bg=self.colors['background'])
        bottom_container.pack(fill='both', expand=True)
        
        # Configure grid weights for responsiveness
        bottom_container.grid_rowconfigure(0, weight=1)
        bottom_container.grid_columnconfigure(0, weight=1)  # Left panel
        bottom_container.grid_columnconfigure(1, weight=2)  # Right panel (2x wider)
        
        # Left panel - Visualization (SWAPPED) - RESPONSIVE
        left_panel = tk.Frame(bottom_container, bg=self.colors['background'])
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 15))
        
        # Right panel - Chat interface (SWAPPED) - RESPONSIVE
        right_panel = tk.Frame(bottom_container, bg=self.colors['background'])
        right_panel.grid(row=0, column=1, sticky='nsew')
        
        self.setup_top_panel(top_panel)
        self.setup_visualization_panel(left_panel)   # Viz on left
        self.setup_chat_panel(right_panel)           # Chat on right
        
        # Bind window resize event for responsive behavior
        self.root.bind('<Configure>', self.on_window_resize)
    
    def center_header_text(self, event):
        """Center header text when window is resized"""
        if event.widget == self.header:
            width = event.width
            self.header.coords(self.header_text1, width//2, 40)
            self.header.coords(self.header_text2, width//2, 80)
    
    def create_modern_card(self, parent, title=None, color=None):
        """Create dark theme card"""
        if color is None:
            color = self.colors['card']
        
        card = tk.Frame(parent, bg=color, relief='flat', bd=0)
        card.configure(highlightbackground='#334155', highlightthickness=1)
        
        if title:
            title_frame = tk.Frame(card, bg=color)
            title_frame.pack(fill='x', padx=25, pady=(20, 10))
            
            title_label = tk.Label(
                title_frame,
                text=title,
                font=('Segoe UI', 14, 'bold'),
                bg=color,
                fg=self.colors['text']
            )
            title_label.pack(anchor='w')
            
            # Underline
            underline = tk.Frame(card, bg=self.colors['primary_light'], height=2)
            underline.pack(fill='x', padx=25)
        
        return card
    
    def create_gradient_button(self, parent, text, command, icon="", width=25):
        """Create modern gradient button"""
        btn_frame = tk.Frame(parent, bg=self.colors['card'])
        
        btn = tk.Button(
            btn_frame,
            text=f"{icon} {text}",
            command=command,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['primary_light'],
            fg='white',
            activebackground=self.colors['primary'],
            activeforeground='white',
            cursor='hand2',
            relief='flat',
            bd=0,
            width=width,
            height=2
        )
        btn.pack(padx=2, pady=2)
        
        # Hover effects
        def on_enter(e):
            btn.configure(bg=self.colors['primary'])
        
        def on_leave(e):
            btn.configure(bg=self.colors['primary_light'])
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn_frame
    
    def setup_top_panel(self, parent):
        # Control card
        control_card = self.create_modern_card(parent, "Vehicle Selection & Analysis")
        control_card.pack(fill='both', expand=True)
        
        control_inner = tk.Frame(control_card, bg=self.colors['card'])
        control_inner.pack(fill='both', expand=True, padx=25, pady=(15, 20))
        
        # Row 1: File selection
        row1 = tk.Frame(control_inner, bg=self.colors['card'])
        row1.pack(fill='x', pady=(0, 10))
        
        # Status indicator
        status_frame = tk.Frame(row1, bg=self.colors['card'])
        status_frame.pack(side='left', padx=(0, 20))
        
        tk.Label(status_frame, text="Status:", font=('Segoe UI', 10, 'bold'),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0, 8))
        
        self.status_indicator = tk.Label(
            status_frame,
            text="Checking...",
            font=('Segoe UI', 10),
            bg=self.colors['card'],
            fg=self.colors['warning']
        )
        self.status_indicator.pack(side='left')
        
        # File info
        file_frame = tk.Frame(row1, bg=self.colors['card'])
        file_frame.pack(side='left', fill='x', expand=True)
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected - Click browse to get started",
            font=('Segoe UI', 10),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            anchor='w'
        )
        self.file_label.pack(side='left')
        
        # Row 2: Buttons
        row2 = tk.Frame(control_inner, bg=self.colors['card'])
        row2.pack(fill='x')
        
        btn_container = tk.Frame(row2, bg=self.colors['card'])
        btn_container.pack()
        
        browse_btn = self.create_gradient_button(btn_container, "Browse Files", 
                                                 self.browse_file, "[F]", width=20)
        browse_btn.pack(side='left', padx=5)
        
        self.diagnose_btn = self.create_gradient_button(btn_container, "Run AI Diagnosis",
                                                        self.run_diagnosis, "[D]", width=20)
        self.diagnose_btn.pack(side='left', padx=5)
        self.diagnose_btn.pack_forget()  # Hide initially
        
        clear_btn = self.create_gradient_button(btn_container, "Clear Chat",
                                                self.clear_chat, "[C]", width=15)
        clear_btn.pack(side='left', padx=5)
    
    def setup_visualization_panel(self, parent):
        """Setup visualization panel (now on LEFT) with scrollbar"""
        # Visualization card
        viz_card = self.create_modern_card(parent, "Data Visualization")
        viz_card.pack(fill='both', expand=True)
        
        viz_inner = tk.Frame(viz_card, bg=self.colors['card'])
        viz_inner.pack(fill='both', expand=True, padx=20, pady=(15, 20))
        
        # Create scrollable frame for matplotlib
        canvas_frame = tk.Frame(viz_inner, bg=self.colors['card'])
        canvas_frame.pack(fill='both', expand=True)
        
        # Create matplotlib figure with dark theme (responsive)
        self.fig = Figure(figsize=(6, 10), facecolor=self.colors['card'])  # Increased height for more plots
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        
        # Create scrollbar
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.get_tk_widget().yview)
        self.canvas.get_tk_widget().configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Make canvas responsive
        self.canvas.get_tk_widget().configure(height=1, width=1)
        
        # Bind canvas resize
        def on_canvas_resize(event):
            if event.widget == self.canvas.get_tk_widget():
                try:
                    self.canvas.draw()
                except:
                    pass
        
        self.canvas.get_tk_widget().bind('<Configure>', on_canvas_resize)
        
        self.show_placeholder_chart()
    
    def setup_chat_panel(self, parent):
        # Chat card
        chat_card = self.create_modern_card(parent, "AI Chat Assistant", self.colors['card'])
        chat_card.pack(fill='both', expand=True)
        
        chat_inner = tk.Frame(chat_card, bg=self.colors['card'])
        chat_inner.pack(fill='both', expand=True, padx=0, pady=(15, 0))
        
        # Chat messages area
        chat_frame = tk.Frame(chat_inner, bg=self.colors['chat_bg'])
        chat_frame.pack(fill='both', expand=True)
        
        # Scrollable chat
        chat_canvas = tk.Canvas(chat_frame, bg=self.colors['chat_bg'], 
                               highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(chat_frame, orient='vertical', 
                                  command=chat_canvas.yview)
        
        self.chat_container = tk.Frame(chat_canvas, bg=self.colors['chat_bg'])
        
        chat_canvas.create_window((0, 0), window=self.chat_container, anchor='nw')
        chat_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Use grid for better responsiveness
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        chat_canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        def on_frame_configure(event):
            chat_canvas.configure(scrollregion=chat_canvas.bbox('all'))
            chat_canvas.yview_moveto(1.0)  # Auto-scroll to bottom
        
        self.chat_container.bind('<Configure>', on_frame_configure)
        self.chat_canvas = chat_canvas
        
        # Welcome message
        self.add_chat_message("Hello! I'm your AI Vehicle Diagnosis Assistant. Upload a vehicle data file to get started!", is_ai=True)
        
        # Input area
        input_frame = tk.Frame(chat_inner, bg=self.colors['card'], height=80)
        input_frame.pack(fill='x', pady=(10, 10))
        input_frame.pack_propagate(False)
        
        input_container = tk.Frame(input_frame, bg=self.colors['card'])
        input_container.pack(fill='both', expand=True, padx=15, pady=10)
        
        self.chat_input = tk.Text(
            input_container,
            font=('Segoe UI', 10),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            relief='flat',
            bd=0,
            height=2,
            wrap='word',
            padx=15,
            pady=10,
            insertbackground='white'
        )
        self.chat_input.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        send_btn = tk.Button(
            input_container,
            text="Send\n[S]",
            command=self.send_message,
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['primary_light'],
            fg='white',
            cursor='hand2',
            relief='flat',
            bd=0,
            width=8
        )
        send_btn.pack(side='right')
        
        # Bind Enter key
        self.chat_input.bind('<Return>', lambda e: self.send_message() or 'break')
    
    
    def show_placeholder_chart(self):
        """Show empty chart when no data is loaded"""
        self.fig.clear()
        self.fig.patch.set_facecolor(self.colors['card'])
        # Just show empty dark background - no placeholder text
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(self.colors['card'])
        ax.axis('off')
        self.canvas.draw()
    
    def plot_vehicle_data(self, df):
        """Create dark theme visualizations with scrollable plots"""
        self.fig.clear()
        self.fig.patch.set_facecolor(self.colors['card'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if 'Unnamed' not in col]
        
        if len(numeric_cols) == 0:
            self.show_placeholder_chart()
            return
        
        # Show up to 12 plots (instead of 6) - 6 rows x 2 cols
        cols = 2
        rows = min(6, (len(numeric_cols) + 1) // 2)
        
        # Create plots for all available numeric columns (up to 12)
        for idx, col in enumerate(numeric_cols[:12], 1):
            ax = self.fig.add_subplot(rows, cols, idx)
            ax.set_facecolor('#0f172a')
            data = df[col].dropna()
            
            if len(data) > 15:
                # Dark theme line plot
                x = np.arange(len(data))
                y = data.values
                ax.plot(x, y, color='#60a5fa', linewidth=2.5, alpha=0.9)
                ax.fill_between(x, y, alpha=0.3, color='#3b82f6')
            else:
                # Dark theme bar chart
                ax.bar(range(len(data)), data.values, color='#3b82f6', alpha=0.8, edgecolor='#60a5fa')
            
            # Dark theme styling
            ax.set_title(col[:35], fontsize=9, fontweight='bold', pad=6, color='#f1f5f9')
            ax.set_xlabel('Sample', fontsize=7, color='#94a3b8')
            ax.set_ylabel('Value', fontsize=7, color='#94a3b8')
            ax.tick_params(labelsize=6, colors='#94a3b8')
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='#475569')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')
        
        # Adjust layout for more plots
        self.fig.tight_layout(pad=1.5)
        
        # Update scrollable region
        self.canvas.get_tk_widget().configure(scrollregion=self.canvas.get_tk_widget().bbox("all"))
        self.canvas.draw()
    
    def add_chat_message(self, message, is_ai=True, typing_effect=False):
        """Add message to chat with optional typing effect and robot icons"""
        msg_widget = ChatMessage(self.chat_container, message, is_ai=is_ai, bg_color=self.colors['chat_bg'], ai_avatar_photo=self.ai_avatar_photo)
        msg_widget.pack(fill='x', pady=2)
        self.chat_canvas.yview_moveto(1.0)
    
    def check_ollama_status(self):
        """Check Ollama status"""
        def check():
            try:
                ollama_manager = OllamaManager(model_name="qwen3:8b")
                
                if not ollama_manager.is_ollama_running():
                    self.status_indicator.config(text="Offline", fg=self.colors['danger'])
                    self.add_chat_message("Ollama is not running. Please start it with: ollama serve", is_ai=True)
                    return
                
                if not ollama_manager.is_model_available():
                    self.status_indicator.config(text="Model Missing", fg=self.colors['danger'])
                    self.add_chat_message("AI model not found. Run: ollama pull qwen3:8b", is_ai=True)
                    return
                
                self.agent = VehicleDiagnosisAgent(model_name="qwen3:8b")
                self.status_indicator.config(text="Online", fg=self.colors['success'])
                self.add_chat_message("AI system is ready! I can analyze vehicle diagnostic data and answer your questions.", is_ai=True)
                
            except Exception as e:
                self.status_indicator.config(text="Error", fg=self.colors['danger'])
                self.add_chat_message(f"Error: {str(e)}", is_ai=True)
        
        threading.Thread(target=check, daemon=True).start()
    
    def browse_file(self):
        """Browse for file"""
        filetypes = (
            ('Excel files', '*.xls *.xlsx'),
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select Vehicle Data File',
            initialdir=Path.cwd(),
            filetypes=filetypes
        )
        
        if filename:
            self.selected_file = filename
            vehicle_name = Path(filename).stem
            self.file_label.config(text=f"File: {vehicle_name}", fg=self.colors['text'])
            
            # Load and visualize
            self.load_file_info()
            self.diagnose_btn.pack(side='left', padx=5)
            
            self.add_chat_message(f"File loaded: {vehicle_name}\nClick 'Run AI Diagnosis' to analyze this vehicle.", is_ai=True)
    
    def load_file_info(self):
        """Load file and create visualizations"""
        try:
            if self.selected_file.endswith('.csv'):
                df = pd.read_csv(self.selected_file)
            else:
                df = pd.read_excel(self.selected_file)
            
            self.current_data = df
            
            # Add info message
            info_msg = f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns\n"
            info_msg += f"Found {len([c for c in df.columns if 'Unnamed' not in c])} diagnostic parameters"
            self.add_chat_message(info_msg, is_ai=True)
            
            # Plot
            self.plot_vehicle_data(df)
            
        except Exception as e:
            self.add_chat_message(f"Error loading file: {str(e)}", is_ai=True)
    
    def run_diagnosis(self):
        """Run AI diagnosis"""
        if not self.selected_file or not self.agent:
            return
        
        self.add_chat_message("Analyzing vehicle diagnostic data...", is_ai=False)
        self.add_chat_message("Processing data and running AI analysis. This may take 30-60 seconds...", is_ai=True)
        
        def diagnose():
            try:
                vehicle_data = self.agent.read_vehicle_data(self.selected_file)
                vehicle_name = Path(self.selected_file).stem
                prompt = self.agent.prepare_diagnosis_prompt(vehicle_data, vehicle_name)
                diagnosis = self.agent.query_ollama(prompt)
                
                # Format diagnosis for chat
                formatted = self.format_diagnosis_for_chat(diagnosis, vehicle_name)
                self.root.after(0, self.display_diagnosis, formatted)
                
            except Exception as e:
                self.root.after(0, self.add_chat_message, f"Error: {str(e)}", True)
        
        threading.Thread(target=diagnose, daemon=True).start()
    
    def format_diagnosis_for_chat(self, diagnosis, vehicle_name):
        """Format diagnosis as chat-friendly messages"""
        messages = []
        
        messages.append(f"DIAGNOSIS REPORT: {vehicle_name.upper()}\n" + "-"*50)
        
        lines = diagnosis.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'MAIN ISSUE' in line:
                messages.append("\n[MAIN ISSUE]")
                content = line.split('**', 2)[-1].strip() if '**' in line else line
                messages.append(content)
            elif 'WHAT IT MEANS' in line:
                messages.append("\n[WHAT IT MEANS]")
                content = line.split('**', 2)[-1].strip() if '**' in line else line
                messages.append(content)
            elif 'SEVERITY' in line:
                messages.append("\n[SEVERITY]")
                content = line.split('**', 2)[-1].strip() if '**' in line else line
                messages.append(content)
            elif 'RECOMMENDED ACTION' in line:
                messages.append("\n[RECOMMENDED ACTION]")
                content = line.split('**', 2)[-1].strip() if '**' in line else line
                messages.append(content)
            elif 'URGENCY' in line:
                messages.append("\n[ESTIMATED URGENCY]")
                content = line.split('**', 2)[-1].strip() if '**' in line else line
                messages.append(content)
            elif line.startswith('**'):
                clean_line = line.replace('**', '')
                if clean_line.strip():
                    messages.append(clean_line)
        
        return '\n'.join(messages)
    
    def display_diagnosis(self, diagnosis):
        """Display diagnosis in chat"""
        self.add_chat_message(diagnosis, is_ai=True, typing_effect=False)
        self.add_chat_message("Analysis complete. You can now ask me specific questions about your vehicle's condition, recommended actions, or any concerns you may have.", is_ai=True)
    
    def send_message(self):
        """Send user message and get AI response"""
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        self.chat_input.delete("1.0", tk.END)
        self.add_chat_message(message, is_ai=False)
        
        # Simple AI response based on context
        def get_response():
            if not self.agent:
                response = "AI system is not ready. Please check if Ollama is running."
            elif not self.selected_file:
                response = "Please select a vehicle data file first so I can help you with specific diagnostics."
            else:
                # Generate contextual response
                vehicle_name = Path(self.selected_file).stem
                prompt = f"""You are a professional vehicle diagnostic assistant. The user is asking about {vehicle_name}. 
                User question: {message}
                
                Provide a professional, concise answer (2-3 sentences) related to vehicle diagnostics. Use formal language and be specific about technical aspects."""
                
                response = self.agent.query_ollama(prompt)
            
            self.root.after(0, self.add_chat_message, response, True, False)
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def clear_chat(self):
        """Clear chat messages"""
        for widget in self.chat_container.winfo_children():
            widget.destroy()
        self.add_chat_message("Chat cleared. How can I help you?", is_ai=True)
    
    def on_window_resize(self, event):
        """Handle window resize for responsive layout"""
        if event.widget == self.root:
            # Update matplotlib canvas size
            if hasattr(self, 'canvas'):
                try:
                    self.canvas.draw()
                except:
                    pass


def main():
    root = tk.Tk()
    app = UltraModernVehicleDiagnosisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

