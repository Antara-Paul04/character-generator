from flask import Flask, render_template, request, jsonify
import json
import os
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import spacy
import re
from typing import Dict, List, Tuple, Any
import numpy as np

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_properties():
    """Load character properties from CSV file"""
    try:
        possible_paths = [
            "New-Text-Document.csv",
            "data/New-Text-Document.csv", 
            "../New-Text-Document.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✓ Loaded {len(df)} properties from {path}")
                return df[df.columns[0]].tolist()
        
        print("✗ Could not find properties CSV file")
        return []
    except Exception as e:
        print(f"✗ Error loading properties: {e}")
        return []

CHARACTER_PROPERTIES = load_properties()

try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded successfully")
except OSError:
    print("✗ spaCy model not found")
    nlp = None

# =============================================================================
# HAIR ANALYSIS SYSTEM
# =============================================================================

class HairAnalyzer:
    """Analyzes hair descriptions from prompts"""
    
    def __init__(self):
        self.hair_keywords = {
            'length': {
                'very_short': ['buzz cut', 'shaved', 'bald', 'crew cut'],
                'short': ['short hair', 'cropped', 'pixie', 'bob cut', 'short cut'],
                'medium': ['medium hair', 'shoulder-length', 'medium length'],
                'long': ['long hair', 'flowing hair', 'lengthy hair'],
                'very_long': ['very long', 'extremely long', 'waist-length', 'floor-length']
            },
            'type': {
                'straight': ['straight hair', 'silky', 'sleek', 'flat'],
                'wavy': ['wavy hair', 'waves', 'beach waves'],
                'curly': ['curly hair', 'curls', 'ringlets', 'spiral curls'],
                'kinky': ['kinky', 'coily', 'afro textured', 'tight curls']
            },
            'style': {
                'ponytail': ['ponytail', 'pony tail', 'tied back'],
                'bun': ['bun', 'top knot', 'hair bun'],
                'braids': ['braids', 'braided', 'plaits'],
                'afro': ['afro', 'natural afro'],
                'bob': ['bob', 'bob cut'],
                'pixie': ['pixie', 'pixie cut']
            },
            'color': {
                'black': ['black hair', 'jet black', 'ebony hair'],
                'dark_brown': ['dark brown', 'dark hair', 'brunette'],
                'brown': ['brown hair', 'chestnut'],
                'light_brown': ['light brown', 'mousy brown'],
                'blonde': ['blonde', 'golden hair', 'yellow hair'],
                'red': ['red hair', 'ginger', 'redhead'],
                'auburn': ['auburn', 'reddish brown'],
                'white': ['white hair', 'silver hair'],
                'gray': ['gray hair', 'grey hair', 'salt and pepper']
            },
            'density': {
                'thick': ['thick hair', 'full hair', 'voluminous', 'abundant'],
                'thin': ['thin hair', 'fine hair', 'sparse', 'wispy']
            }
        }
    
    def analyze_hair(self, prompt: str) -> Dict:
        """Analyze prompt for hair characteristics"""
        prompt_lower = prompt.lower()
        
        hair_data = {
            'has_hair_description': False,
            'length': None,
            'type': None,
            'style': None,
            'color': None,
            'density': 'normal'
        }
        
        # Check if hair is mentioned
        hair_indicators = ['hair', 'hairstyle', 'haircut', 'locks', 'mane', 'tresses']
        if any(indicator in prompt_lower for indicator in hair_indicators):
            hair_data['has_hair_description'] = True
        else:
            return hair_data
        
        # Detect length
        for length_type, keywords in self.hair_keywords['length'].items():
            if any(keyword in prompt_lower for keyword in keywords):
                hair_data['length'] = length_type
                break
        
        # Detect type
        for hair_type, keywords in self.hair_keywords['type'].items():
            if any(keyword in prompt_lower for keyword in keywords):
                hair_data['type'] = hair_type
                break
        
        # Detect style
        for style_name, keywords in self.hair_keywords['style'].items():
            if any(keyword in prompt_lower for keyword in keywords):
                hair_data['style'] = style_name
                break
        
        # Detect color
        for color_name, keywords in self.hair_keywords['color'].items():
            if any(keyword in prompt_lower for keyword in keywords):
                hair_data['color'] = color_name
                break
        
        # Detect density
        for density_type, keywords in self.hair_keywords['density'].items():
            if any(keyword in prompt_lower for keyword in keywords):
                hair_data['density'] = density_type
                break
        
        return hair_data

# =============================================================================
# CHARACTER PROPERTY ANALYZER (with hair integration)
# =============================================================================

class CharacterPropertyAnalyzer:
    def __init__(self, properties_list):
        self.properties = properties_list
        self.nlp = nlp
        self.hair_analyzer = HairAnalyzer()
        
        # Feature keywords (keeping your existing ones)
        self.feature_keywords = {
            'female': ['female', 'woman', 'girl', 'lady', 'feminine', 'she', 'her'],
            'male': ['male', 'man', 'boy', 'guy', 'masculine', 'he', 'him'],
            'big_eyes': ['big eyes', 'large eyes', 'wide eyes'],
            'small_eyes': ['small eyes', 'narrow eyes'],
            'sharp_nose': ['sharp nose', 'pointed nose'],
            'full_lips': ['full lips', 'plump lips', 'thick lips'],
            'muscular_body': ['muscular', 'athletic', 'toned', 'fit'],
            'slim_body': ['slim', 'thin', 'lean', 'slender'],
            # ... (keep all your existing feature keywords)
        }
        
        # Property mapping (keeping your existing ones)
        self.property_mapping = {
            'big_eyes': ['L2__Eyes_Size_max', 'L2__Eyes_IrisSize_max'],
            'small_eyes': ['L2__Eyes_Size_min', 'L2__Eyes_IrisSize_min'],
            # ... (keep all your existing mappings)
        }
        
        self.ethnicity_base_map = {
            'caucasian': 'L1_Caucasian',
            'asian': 'L1_Asian',
            'african': 'L1_African',
            'latino': 'L1_Latin',
            'elf': 'L1_Elf',
            'dwarf': 'L1_Dwarf',
            'anime': 'L1_Anime'
        }
    
    def detect_gender(self, prompt):
        """Detect gender from prompt"""
        prompt_lower = prompt.lower()
        
        female_score = 0
        male_score = 0
        
        for keyword in self.feature_keywords['female']:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, prompt_lower)
            female_score += len(matches)
        
        if female_score == 0:
            for keyword in self.feature_keywords['male']:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, prompt_lower)
                male_score += len(matches)
        
        if female_score > 0:
            return 'female', female_score
        elif male_score > 0:
            return 'male', male_score
        else:
            return 'male', 1
    
    def detect_ethnicity(self, prompt):
        """Detect ethnicity with fallback"""
        prompt_lower = prompt.lower()
        
        ethnicity_keywords = {
            'caucasian': ['caucasian', 'white', 'european'],
            'asian': ['asian', 'chinese', 'japanese', 'korean'],
            'african': ['african', 'black'],
            'latino': ['latino', 'hispanic']
        }
        
        for ethnicity, keywords in ethnicity_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                return ethnicity, 1, 'detected'
        
        return 'caucasian', 0, 'default'
    
    def map_to_properties(self, prompt):
        """Map prompt to properties including hair analysis"""
        gender, gender_conf = self.detect_gender(prompt)
        ethnicity, eth_conf, eth_source = self.detect_ethnicity(prompt)
        
        # Analyze hair
        hair_data = self.hair_analyzer.analyze_hair(prompt)
        
        properties = {}
        features = {'gender': gender, 'ethnicity': ethnicity}
        
        # Add ethnicity base
        if ethnicity in self.ethnicity_base_map:
            base_prop = self.ethnicity_base_map[ethnicity]
            if base_prop in self.properties:
                properties[base_prop] = 0.85
        
        # Simple feature detection for demo
        prompt_lower = prompt.lower()
        if 'big eyes' in prompt_lower or 'large eyes' in prompt_lower:
            properties['L2__Eyes_Size_max'] = 0.8
            features['big_eyes'] = 0.8
        
        if 'muscular' in prompt_lower or 'athletic' in prompt_lower:
            properties['L2__Arms_UpperarmMass-UpperarmTone_max-max'] = 0.7
            features['muscular_body'] = 0.7
        
        # Ensure minimum properties
        if len(properties) < 30:
            defaults = [
                'L2__Head_Size_max', 'L2__Body_Size_max', 'L2__Eyes_Size_max',
                'L2__Torso_Length_max', 'L2__Legs_UpperlegLength_max'
            ]
            for prop in defaults:
                if prop in self.properties and prop not in properties:
                    properties[prop] = 0.5
                    if len(properties) >= 30:
                        break
        
        return properties, features, gender, ethnicity, hair_data

# Initialize analyzers
property_analyzer = CharacterPropertyAnalyzer(CHARACTER_PROPERTIES)

# =============================================================================
# LLM SETUP (simplified)
# =============================================================================

MODEL_DIR = "microsoft/DialoGPT-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model load failed: {e}")
    model = None
    tokenizer = None

# =============================================================================
# BLENDER CONFIGURATION
# =============================================================================

COMMUNICATION_DIR = r"C:\temp\blender_bridge"
REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")

BLENDER_EXECUTABLE = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
MODEL_BLEND_FILE = os.path.join(os.getcwd(), "base.blend")
BRIDGE_SCRIPT_PATH = os.path.join(os.getcwd(), "blender_bridge.py")

blender_started_once = False
last_successful_generation = None
os.makedirs(COMMUNICATION_DIR, exist_ok=True)

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index_hair.html')

@app.route('/start-blender', methods=['POST'])
def start_blender():
    global blender_started_once
    
    try:
        if not os.path.exists(MODEL_BLEND_FILE):
            return jsonify({"success": False, "error": f"Model file not found"}), 500
        
        startup_script = f'''
import bpy
import sys
import os

sys.path.append(r"{os.getcwd()}")
exec(open(r"{BRIDGE_SCRIPT_PATH}").read())
start_bridge_monitoring()
print("=== BLENDER BRIDGE WITH HAIR AUTO-STARTED ===")
'''
        
        startup_script_path = os.path.join(COMMUNICATION_DIR, "startup_script.py")
        with open(startup_script_path, 'w') as f:
            f.write(startup_script)
        
        cmd = [BLENDER_EXECUTABLE, MODEL_BLEND_FILE, "--python", startup_script_path]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        
        time.sleep(5)
        blender_started_once = True
        
        return jsonify({
            "success": True, 
            "message": "Blender started with hair generation support"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_character():
    global last_successful_generation
    
    try:
        user_prompt = request.json.get('prompt', '')
        
        if not user_prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        if not blender_started_once:
            return jsonify({
                "error": "Please start Blender first"
            }), 400

        print(f"\n{'='*70}")
        print(f"🎭 PROCESSING CHARACTER WITH HAIR")
        print(f"{'='*70}")
        print(f"Prompt: {user_prompt}")
        
        # Analyze character and hair
        properties, features, gender, ethnicity, hair_data = property_analyzer.map_to_properties(user_prompt)
        
        print(f"\n✓ Gender: {gender.upper()}")
        print(f"✓ Ethnicity: {ethnicity.upper()}")
        print(f"✓ Properties: {len(properties)}")
        
        if hair_data['has_hair_description']:
            print(f"\n💇 HAIR DETECTED:")
            print(f"  Length: {hair_data.get('length', 'not specified')}")
            print(f"  Type: {hair_data.get('type', 'not specified')}")
            print(f"  Style: {hair_data.get('style', 'not specified')}")
            print(f"  Color: {hair_data.get('color', 'not specified')}")
            print(f"  Density: {hair_data.get('density', 'normal')}")
        
        # Prepare structured data
        structured_data = {
            "properties": properties,
            "analysis": {
                "analysis": f"Character with {len(properties)} properties",
                "hair_info": hair_data
            },
            "prompt": user_prompt,
            "timestamp": datetime.now().isoformat(),
            "gender": gender,
            "ethnicity": ethnicity,
            "hair_data": hair_data,
            "property_count": len(properties)
        }
        
        # Send to Blender
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": user_prompt,
            "structured_data": structured_data,
            "status": "pending"
        }
        
        with open(REQUEST_FILE, 'w') as f:
            json.dump(request_data, f, indent=2)
        
        # Wait for response
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(RESPONSE_FILE):
                with open(RESPONSE_FILE, 'r') as f:
                    response_data = json.load(f)
                
                os.remove(RESPONSE_FILE)
                last_successful_generation = datetime.now()
                
                print(f"\n✅ SUCCESS! Character generated")
                print(f"{'='*70}\n")
                
                return jsonify({
                    "success": True,
                    "message": f"✓ Character with hair generated!",
                    "details": response_data,
                    "property_count": len(properties),
                    "hair_generated": response_data.get("hair_generated", False),
                    "hair_data": hair_data,
                    "gender": gender,
                    "ethnicity": ethnicity
                })
            
            time.sleep(0.5)
        
        return jsonify({
            "error": "Timeout waiting for Blender"
        }), 408
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "blender_running": blender_started_once,
        "properties_loaded": len(CHARACTER_PROPERTIES) > 0,
        "llm_available": model is not None,
        "nlp_available": nlp is not None,
        "hair_system_available": True,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🎭 Character Generator with Hair System Starting...")
    print(f"📊 Properties loaded: {len(CHARACTER_PROPERTIES)}")
    print(f"💇 Hair generation: Enabled")
    print("🌐 Open http://127.0.0.1:5000")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=False)