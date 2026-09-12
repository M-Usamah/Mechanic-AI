import pandas as pd
import time
from pathlib import Path
import os
from groq import Groq
from dotenv import load_dotenv

class GroqManager:
    """Manage Groq API connection and model availability"""
    
    def __init__(self, model_name="qwen/qwen3-32b"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client with API key from .env file"""
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            
            self.client = Groq(api_key=api_key)
            print("✅ Groq client initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Groq client: {e}")
            return False
    
    def is_available(self):
        """Check if Groq API is available"""
        try:
            if not self.client:
                return False
            
            # Test the connection with a simple request
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0
            )
            return True
            
        except Exception as e:
            print(f"❌ Groq API not available: {e}")
            return False

class VehicleDiagnosisAgent:
    """AI-powered vehicle diagnosis agent using Groq"""
    
    def __init__(self, model_name="qwen/qwen3-32b"):
        self.model_name = model_name
        self.groq_manager = GroqManager(model_name)
    
    def read_vehicle_data(self, file_path):
        """Read vehicle data from Excel or CSV file"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
    
    def prepare_diagnosis_prompt(self, df, vehicle_name):
        """Prepare diagnosis prompt with data summary"""
        if df is None or df.empty:
            return "No data available for diagnosis."
        
        # Create data summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if 'Unnamed' not in col]
        
        data_summary = f"Vehicle: {vehicle_name}\n"
        data_summary += f"Data points: {len(df)} rows, {len(numeric_cols)} numeric parameters\n\n"
        
        # Add key statistics for each numeric column
        for col in numeric_cols[:15]:  # Limit to first 15 columns
            if not df[col].isna().all():
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                data_summary += f"{col}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]\n"
        
        prompt = f"""You are a Master ASE Certified Automotive Diagnostic Technician with 25+ years of experience in vehicle diagnostics, specializing in engine management systems, fuel injection, and electronic control units.

VEHICLE DATA ANALYSIS FOR: {vehicle_name}
{data_summary}

DIAGNOSTIC PROTOCOL:
Analyze the data systematically and identify ALL anomalies:
1. Zero values in critical parameters (sensor failures, injector failures, electrical faults)
2. Extreme or out-of-range values (component malfunctions, circuit issues)
3. Patterns indicating specific system failures
4. Related fault codes and their implications

Provide a COMPLETE, PROFESSIONAL DIAGNOSTIC REPORT following this EXACT structure:

**MAIN ISSUE:**
[Provide a clear, specific technical diagnosis with exact component identification. Include measured values that led to this conclusion. Example: "Cylinder 4 fuel injector failure - injection time consistently 0.00ms across all measurements"]

**SYMPTOMS & IMPACT:**
[List 4-5 specific symptoms the driver experiences, using bullet points]
• [Specific symptom with technical explanation]
• [Performance degradation details]
• [Warning lights/codes that may appear]
• [Fuel consumption or emissions impact]
• [Safety concerns if applicable]

**SEVERITY LEVEL:**
[Choose: CRITICAL / HIGH / MODERATE / LOW]
[Explain in 2-3 sentences WHY this severity level, including potential damage if ignored]

**ROOT CAUSE:**
[Provide detailed technical explanation of WHY this failure occurs. Include:]
• Most likely cause (electrical, mechanical, or software)
• Contributing factors
• How the failure mechanism works
• Related components that may be affected

**DIAGNOSTIC VERIFICATION:**
[List specific tests to confirm the diagnosis:]
• [Test 1 with expected results]
• [Test 2 with expected results]
• [Recommended diagnostic equipment/tools]

**IMMEDIATE ACTION REQUIRED:**
[Provide step-by-step repair procedure:]
1. [First step with details]
2. [Second step with parts/specifications]
3. [Third step with torque specs or calibration needs]
4. [Final verification steps]

**URGENCY TIMEFRAME:**
[Choose: IMMEDIATE / WITHIN 3 DAYS / WITHIN 7 DAYS / WITHIN 30 DAYS]
[Justify with safety and damage prevention rationale]

REQUIREMENTS:
- Write in clear, professional automotive industry language
- Use proper technical terminology
- Include specific measurements and values from the data
- Provide complete, actionable recommendations
- Maximum 500 words but must be COMPLETE
- Do NOT cut off sentences or leave sections incomplete
- Each section must be fully written out
- Do NOT include any cost estimates, pricing, or financial information
- End the report immediately after the URGENCY TIMEFRAME section
- Do NOT add any concluding statements or summaries after the URGENCY TIMEFRAME"""
        
        return prompt
    
    def query_groq(self, prompt):
        """Query Groq with the diagnosis prompt"""
        try:
            if not self.groq_manager.client:
                print("❌ Groq client not initialized")
                return None
            
            print("🤖 Querying Groq AI (this may take 30-60 seconds)...")
            
            response = self.groq_manager.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a Master ASE Certified Automotive Diagnostic Technician. Provide complete, detailed, professional vehicle diagnosis reports. NEVER leave sentences incomplete. ALWAYS finish every section completely."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.2,
                top_p=0.95,
                stream=False
            )
            
            diagnosis = response.choices[0].message.content.strip()
            
            # Clean up the response
            diagnosis = diagnosis.strip()
            
            # Remove any thinking blocks if present
            if '<think>' in diagnosis and '</think>' in diagnosis:
                diagnosis = diagnosis.split('</think>')[-1].strip()
            
            # Remove "done thinking" phrases
            diagnosis = diagnosis.replace('...done thinking', '').strip()
            
            # Remove any cost-related sections (ESTIMATED COST)
            if '**ESTIMATED COST:**' in diagnosis or '**ESTIMATED COST**' in diagnosis or 'ESTIMATED COST:' in diagnosis:
                # Find where URGENCY TIMEFRAME ends
                if '**URGENCY TIMEFRAME:**' in diagnosis or 'URGENCY TIMEFRAME:' in diagnosis:
                    # Split and find the timeframe section
                    lines = diagnosis.split('\n')
                    cleaned_lines = []
                    skip_mode = False
                    urgency_found = False
                    lines_after_urgency = 0
                    
                    for line in lines:
                        if 'URGENCY TIMEFRAME' in line:
                            urgency_found = True
                            cleaned_lines.append(line)
                            continue
                        
                        if urgency_found and not skip_mode:
                            # Keep lines until we hit ESTIMATED COST or empty line after timeframe content
                            if 'ESTIMATED COST' in line:
                                skip_mode = True
                                continue
                            if line.strip():
                                lines_after_urgency += 1
                                cleaned_lines.append(line)
                                # After 3 lines of content, stop (timeframe explanation is complete)
                                if lines_after_urgency >= 3:
                                    skip_mode = True
                            else:
                                if lines_after_urgency > 0:
                                    # Empty line after urgency content, cost section follows
                                    skip_mode = True
                        elif not skip_mode:
                            cleaned_lines.append(line)
                    
                    diagnosis = '\n'.join(cleaned_lines).strip()
            
            # Remove unnecessary concluding statements
            concluding_phrases = [
                'This diagnosis ensures',
                'Immediate repair is critical',
                'This repair is essential',
                'Following this diagnosis',
                'This assessment',
                'Total estimate:',
                '€'
            ]
            
            for phrase in concluding_phrases:
                if phrase in diagnosis:
                    parts = diagnosis.split(phrase)
                    if len(parts) > 1:
                        # Check if this appears after URGENCY TIMEFRAME
                        before_phrase = parts[0]
                        if 'URGENCY TIMEFRAME' in before_phrase:
                            diagnosis = before_phrase.rstrip()
                            break
            
            print("✅ Diagnosis received from Groq AI")
            return diagnosis
            
        except Exception as e:
            print(f"❌ Error querying Groq: {e}")
            return None
    
    def diagnose_vehicle(self, file_path):
        """Complete vehicle diagnosis workflow"""
        try:
            # Check if Groq is available
            if not self.groq_manager.is_available():
                print("❌ Groq API not available. Please check your API key and internet connection.")
                return None
            
            # Read vehicle data
            df = self.read_vehicle_data(file_path)
            if df is None:
                return None
            
            # Prepare prompt
            vehicle_name = Path(file_path).stem
            prompt = self.prepare_diagnosis_prompt(df, vehicle_name)
            
            # Get diagnosis
            diagnosis = self.query_groq(prompt)
            
            return diagnosis
            
        except Exception as e:
            print(f"❌ Error in diagnosis workflow: {e}")
            return None

    def chat_with_ai(self, user_message):
        """Chat with the AI for follow-up questions"""
        try:
            if not self.groq_manager.client:
                return "❌ Groq client not initialized"
            
            print("🤖 Chatting with Groq AI...")
            
            response = self.groq_manager.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert automotive diagnostic technician with access to the vehicle's diagnosis report. Provide clear, direct answers without showing your thinking process. Reference specific details from the diagnosis when answering questions. Be concise but informative."
                    },
                    {
                        "role": "user", 
                        "content": user_message
                    }
                ],
                max_tokens=800,
                temperature=0.7,
                top_p=0.9,
                stream=False
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Remove any thinking blocks if present
            if '<think>' in ai_response and '</think>' in ai_response:
                ai_response = ai_response.split('</think>')[-1].strip()
            
            # Remove standalone <think> tags
            if '<think>' in ai_response:
                ai_response = ai_response.split('<think>')[0].strip()
            
            # Remove "done thinking" phrases
            ai_response = ai_response.replace('...done thinking', '').strip()
            
            print("✅ Chat response received from Groq AI")
            return ai_response
            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return f"❌ Error: {str(e)}"


def main():
    """Test the diagnosis agent"""
    agent = VehicleDiagnosisAgent()
    
    # Test with sample file if available
    sample_files = ['ford 1.5 diesel.xls', 'audi a1.xls', 'hyndai.xls']
    
    for filename in sample_files:
        if os.path.exists(filename):
            print(f"\n🧪 Testing with {filename}...")
            diagnosis = agent.diagnose_vehicle(filename)
            if diagnosis:
                print("\n📋 Diagnosis:")
                print(diagnosis)
            break
    else:
        print("❌ No sample files found for testing")


if __name__ == "__main__":
    main()