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
# CONFIGURATION - PROPERTY MAPPING SYSTEM
# =============================================================================

# Load properties from CSV
def load_properties():
    """Load character properties from CSV file"""
    try:
        # Try multiple possible file locations
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

# Load properties at startup
CHARACTER_PROPERTIES = load_properties()

# Initialize spaCy for NLP
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded successfully")
except OSError:
    print("✗ spaCy model not found. Please install: python -m spacy download en_core_web_sm")
    nlp = None

# Enhanced feature mapping system
class CharacterPropertyAnalyzer:
    def __init__(self, properties_list):
        self.properties = properties_list
        self.nlp = nlp
        
        # Enhanced feature mapping with cultural and lifestyle context
        self.feature_keywords = {
            # Direct physical features
            'big_eyes': ['big eyes', 'large eyes', 'wide eyes', 'expressive eyes', 'doe eyes'],
            'small_eyes': ['small eyes', 'narrow eyes', 'squinty eyes', 'beady eyes'],
            'sharp_nose': ['sharp nose', 'pointed nose', 'angular nose', 'refined nose'],
            'wide_nose': ['wide nose', 'broad nose', 'flat nose', 'button nose'],
            'long_nose': ['long nose', 'prominent nose', 'aquiline nose'],
            'short_nose': ['short nose', 'small nose', 'snub nose'],
            'full_lips': ['full lips', 'plump lips', 'big lips', 'luscious lips', 'pouty lips'],
            'thin_lips': ['thin lips', 'small lips', 'narrow lips'],
            'strong_jaw': ['strong jaw', 'defined jaw', 'angular jaw', 'square jaw', 'chiseled jaw'],
            'soft_jaw': ['soft jaw', 'round jaw', 'gentle jaw', 'delicate jaw'],
            'wide_jaw': ['wide jaw', 'broad jaw'],
            'narrow_jaw': ['narrow jaw', 'thin jaw'],
            'prominent_chin': ['strong chin', 'prominent chin', 'defined chin', 'cleft chin'],
            'weak_chin': ['weak chin', 'receding chin', 'small chin'],
            'high_cheekbones': ['high cheekbones', 'prominent cheeks', 'defined cheeks', 'sharp cheekbones'],
            'round_face': ['round face', 'full face', 'chubby face', 'moon face'],
            'angular_face': ['angular face', 'sharp face', 'defined face'],
            
            # Body type features
            'muscular_body': ['muscular', 'athletic', 'toned', 'fit', 'built', 'ripped'],
            'slim_body': ['slim', 'thin', 'lean', 'slender', 'willowy'],
            'large_body': ['large', 'big', 'heavy', 'stocky', 'burly', 'husky'],
            'tall': ['tall', 'height'],
            'short': ['short', 'small stature', 'petite'],
            
            # Age features
            'young': ['young', 'youthful', 'boyish', 'girlish', 'teenage'],
            'old': ['old', 'aged', 'elderly', 'wrinkled', 'senior'],
            
            # Cultural/ethnic indicators
            'caucasian': ['caucasian', 'white', 'european', 'western'],
            'asian': ['asian', 'oriental', 'east asian', 'chinese', 'japanese', 'korean'],
            'african': ['african', 'black', 'afro', 'ebony'],
            'indian': ['indian', 'south asian', 'desi'],
            'middle_eastern': ['middle eastern', 'arab', 'persian'],
            'latino': ['latino', 'hispanic', 'mexican', 'brazilian'],
            
            # Lifestyle indicators
            'food_lover': ['foody', 'loves food', 'eats a lot', 'big eater', 'enjoys food', 'food lover'],
            'athletic': ['athletic', 'sports', 'works out', 'gym goer', 'fit', 'active lifestyle'],
            'sedentary': ['sedentary', 'desk job', 'office worker', 'sits all day', 'inactive'],
            'outdoors': ['outdoors', 'nature lover', 'hiker', 'camper', 'outdoor activities'],
            'intellectual': ['intellectual', 'bookish', 'academic', 'scholar', 'thinker', 'philosopher'],
            'manual_worker': ['manual labor', 'construction', 'farming', 'physical work', 'blue collar'],
            'wealthy': ['wealthy', 'rich', 'affluent', 'upper class', 'well-off'],
            'rural': ['rural', 'countryside', 'village', 'farm', 'agricultural'],
            'urban': ['urban', 'city', 'metropolitan', 'downtown']
        }
        
        # Property mapping with intensity
        self.property_mapping = {
            # Eyes
            'big_eyes': ['L2__Eyes_Size_max', 'L2__Eyes_IrisSize_max'],
            'small_eyes': ['L2__Eyes_Size_min', 'L2__Eyes_IrisSize_min'],
            
            # Nose
            'sharp_nose': ['L2_Caucasian_Nose_TipSize_min', 'L2_Caucasian_Nose_BridgeSizeX_min'],
            'wide_nose': ['L2_Caucasian_Nose_BaseSizeX_max', 'L2_Caucasian_Nose_BridgeSizeX_max'],
            'long_nose': ['L2_Caucasian_Nose_SizeY_max'],
            'short_nose': ['L2_Caucasian_Nose_SizeY_min'],
            
            # Lips
            'full_lips': ['L2_Caucasian_Mouth_UpperlipVolume_max', 'L2_Caucasian_Mouth_LowerlipVolume_max'],
            'thin_lips': ['L2_Caucasian_Mouth_UpperlipVolume_min', 'L2_Caucasian_Mouth_LowerlipVolume_min'],
            
            # Jaw
            'strong_jaw': ['L2_Caucasian_Jaw_Angle_min', 'L2_Caucasian_Jaw_Prominence_max'],
            'soft_jaw': ['L2_Caucasian_Jaw_Angle_max', 'L2_Caucasian_Jaw_Prominence_min'],
            'wide_jaw': ['L2_Caucasian_Jaw_ScaleX_max'],
            'narrow_jaw': ['L2_Caucasian_Jaw_ScaleX_min'],
            
            # Chin
            'prominent_chin': ['L2_Caucasian_Chin_Prominence_max', 'L2_Caucasian_Chin_SizeZ_max'],
            'weak_chin': ['L2_Caucasian_Chin_Prominence_min', 'L2_Caucasian_Chin_SizeZ_min'],
            
            # Cheeks
            'high_cheekbones': ['L2_Caucasian_Cheeks_Zygom_max', 'L2_Caucasian_Cheeks_ZygomPosZ_max'],
            
            # Face shape
            'round_face': ['L2_Caucasian_Face_Ellipsoid_max'],
            'angular_face': ['L2_Caucasian_Face_Triangle_max'],
            
            # Body type
            'muscular_body': ['L2__Body_Size_max', 'L2__Arms_UpperarmMass-UpperarmTone_max-max'],
            'slim_body': ['L2__Body_Size_min', 'L2__Arms_UpperarmMass-UpperarmTone_min-min'],
            'large_body': ['L2__Body_Size_max', 'L2__Chest_Girth_max'],
            'tall': ['L2__Body_Size_max'],
            'short': ['L2__Body_Size_min'],
            
            # Lifestyle traits
            'food_lover': ['L2__Stomach_LocalFat_max', 'L2__Abdomen_Mass-Tone_max-max', 'L2__Body_Size_max'],
            'sedentary': ['L2__Stomach_LocalFat_max', 'L2__Body_Size_max', 'L2__Abdomen_Mass-Tone_max-max'],
            'manual_worker': ['L2__Arms_UpperarmMass-UpperarmTone_max-max', 'L2__Hands_Mass-Tone_max-max', 'L2__Shoulders_Mass-Tone_max-max'],
            'wealthy': ['L2__Body_Size_min', 'L2_Caucasian_Skin_Complexion_max'],
            'rural': ['L2_Caucasian_Skin_Freckles_max', 'L2__Body_Size_max', 'L2__Hands_Mass-Tone_max-max'],
            'urban': ['L2__Body_Size_min', 'L2_Caucasian_Skin_Complexion_min']
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'slightly': 0.3, 'somewhat': 0.4, 'moderately': 0.6,
            'very': 0.8, 'extremely': 0.9, 'incredibly': 1.0,
            'quite': 0.7, 'rather': 0.6, 'fairly': 0.5
        }
        
        self.default_intensity = 0.7

    def analyze_prompt_with_nlp(self, prompt):
        """Use spaCy for detailed NLP analysis of the prompt"""
        if self.nlp is None:
            return self._simple_analysis(prompt)
        
        doc = self.nlp(prompt.lower())
        features = {}
        
        # Extract features using NLP
        for token in doc:
            if token.pos_ in ['ADJ', 'NOUN']:
                for feature, keywords in self.feature_keywords.items():
                    for keyword in keywords:
                        if token.text in keyword.split():
                            intensity = self.default_intensity
                            # Check for intensity modifiers
                            for child in token.children:
                                if child.pos_ == 'ADV' and child.text in self.intensity_modifiers:
                                    intensity = self.intensity_modifiers[child.text]
                            features[feature] = intensity
        
        # Check for multi-word phrases
        for feature, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in prompt.lower() and feature not in features:
                    features[feature] = self.default_intensity
        
        return features

    def _simple_analysis(self, prompt):
        """Fallback analysis without spaCy"""
        prompt_lower = prompt.lower()
        features = {}
        
        for feature, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    intensity = self.default_intensity
                    for modifier, mod_intensity in self.intensity_modifiers.items():
                        if f"{modifier} {keyword}" in prompt_lower:
                            intensity = mod_intensity
                            break
                    features[feature] = intensity
        
        return features

    def map_to_properties(self, prompt):
        """Map analyzed features to specific character properties"""
        features = self.analyze_prompt_with_nlp(prompt)
        property_values = {}
        
        print(f"✓ NLP Analysis - Detected {len(features)} features: {list(features.keys())}")
        
        for feature, intensity in features.items():
            if feature in self.property_mapping:
                properties = self.property_mapping[feature]
                for prop in properties:
                    # Check if property exists in our dataset
                    if prop in self.properties:
                        property_values[prop] = intensity
                    else:
                        # Try to find similar properties
                        similar_props = [p for p in self.properties if prop in p]
                        for similar_prop in similar_props:
                            property_values[similar_prop] = intensity
        
        return property_values

# Initialize the property analyzer
property_analyzer = CharacterPropertyAnalyzer(CHARACTER_PROPERTIES)

# =============================================================================
# CONFIGURATION - LLM LOCAL INFERENCE
# =============================================================================

# =============================================================================
# CONFIGURATION - LLM LOCAL INFERENCE
# =============================================================================

MODEL_DIR = "microsoft/DialoGPT-large"  # Changed to smaller model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

try:
    print(f"Loading model on device: {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else "cpu",
        low_cpu_mem_usage=True,
        offload_folder="./offload"
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("⚠️  Continuing with NLP-only analysis")

# System prompt for LLM to enhance property mapping
SYSTEM_PROMPT = """You are a character analysis expert. Analyze the character description and identify physical traits, cultural background, lifestyle, and personality indicators.

Focus on extracting:
1. Physical features (face shape, eyes, nose, lips, jaw, body type)
2. Cultural/ethnic indicators
3. Lifestyle traits (athletic, sedentary, manual worker, etc.)
4. Age indicators
5. Personality traits that might affect appearance

Return a JSON with:
- "analysis": brief explanation of key traits identified
- "enhanced_features": list of specific physical traits found
- "cultural_context": any cultural/regional indicators
- "lifestyle_traits": lifestyle indicators found"""

# =============================================================================
# CONFIGURATION - BLENDER INTEGRATION
# =============================================================================

COMMUNICATION_DIR = r"C:\temp\blender_bridge"
REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")
BLENDER_STATUS_FILE = os.path.join(COMMUNICATION_DIR, "blender_status.json")

BLENDER_EXECUTABLE = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
MODEL_BLEND_FILE = os.path.join(os.getcwd(), "base.blend")
BRIDGE_SCRIPT_PATH = os.path.join(os.getcwd(), "blender_bridge.py")

blender_started_once = False
last_successful_generation = None
os.makedirs(COMMUNICATION_DIR, exist_ok=True)

# =============================================================================
# BLENDER MANAGEMENT FUNCTIONS
# =============================================================================

def is_blender_responsive():
    global last_successful_generation
    
    if not blender_started_once:
        return False
    
    if last_successful_generation:
        time_since_last = (datetime.now() - last_successful_generation).total_seconds()
        if time_since_last < 60:
            return True
    
    try:
        test_request = {
            "timestamp": datetime.now().isoformat(),
            "prompt": "_STATUS_CHECK_",
            "status": "pending"
        }
        
        with open(REQUEST_FILE, 'w') as f:
            json.dump(test_request, f)
        
        start_time = time.time()
        while time.time() - start_time < 3:
            if os.path.exists(RESPONSE_FILE):
                os.remove(RESPONSE_FILE)
                return True
            time.sleep(0.1)
        
        if os.path.exists(REQUEST_FILE):
            os.remove(REQUEST_FILE)
            
    except:
        pass
    
    try:
        if os.path.exists(BLENDER_STATUS_FILE):
            with open(BLENDER_STATUS_FILE, 'r') as f:
                status = json.load(f)
            last_update = datetime.fromisoformat(status.get('timestamp', '2000-01-01T00:00:00'))
            time_diff = (datetime.now() - last_update).total_seconds()
            return time_diff < 30
    except:
        pass
    
    return False

def start_blender_with_model():
    global blender_started_once
    
    try:
        if not os.path.exists(MODEL_BLEND_FILE):
            return {"success": False, "error": f"Model file not found: {MODEL_BLEND_FILE}"}
        
        if not os.path.exists(BLENDER_EXECUTABLE):
            return {"success": False, "error": f"Blender executable not found: {BLENDER_EXECUTABLE}"}
        
        startup_script = f'''
import bpy
import sys
import os

sys.path.append(r"{os.getcwd()}")
exec(open(r"{BRIDGE_SCRIPT_PATH}").read())
start_bridge_monitoring()
print("=== BLENDER BRIDGE AUTO-STARTED ===")
'''
        
        startup_script_path = os.path.join(COMMUNICATION_DIR, "startup_script.py")
        with open(startup_script_path, 'w') as f:
            f.write(startup_script)
        
        cmd = [
            BLENDER_EXECUTABLE,
            MODEL_BLEND_FILE,
            "--python", startup_script_path
        ]
        
        print(f"Starting Blender with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        
        time.sleep(5)
        blender_started_once = True
        
        return {
            "success": True, 
            "message": f"Blender started with {os.path.basename(MODEL_BLEND_FILE)}",
            "process_id": process.pid
        }
        
    except Exception as e:
        return {"success": False, "error": f"Failed to start Blender: {str(e)}"}

# =============================================================================
# ENHANCED PROPERTY MAPPING WITH LLM
# =============================================================================

def enhance_analysis_with_llm(prompt, nlp_properties):
    """Use Mistral to enhance the NLP analysis"""
    if model is None or tokenizer is None:
        print("✗ LLM not available, using NLP analysis only")
        return nlp_properties, {"analysis": "LLM not available", "enhanced_features": []}
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Character description: {prompt}"}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.3,
                eos_token_id=tokenizer.eos_token_id
            )
        
        raw_output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Parse LLM output to extract additional insights
        llm_analysis = parse_llm_output(raw_output)
        
        # Enhance properties based on LLM insights
        enhanced_properties = enhance_properties_with_llm_insights(nlp_properties, llm_analysis)
        
        return enhanced_properties, llm_analysis
        
    except Exception as e:
        print(f"✗ LLM analysis failed: {e}")
        return nlp_properties, {"analysis": f"LLM analysis failed: {str(e)}", "enhanced_features": []}

def parse_llm_output(raw_output):
    """Parse LLM output to extract structured insights"""
    analysis = {
        "analysis": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
        "enhanced_features": [],
        "cultural_context": "",
        "lifestyle_traits": ""
    }
    
    # Simple keyword extraction from LLM output
    llm_lower = raw_output.lower()
    
    # Extract cultural context
    cultural_indicators = []
    for culture in ['chinese', 'indian', 'japanese', 'korean', 'african', 'european', 'american', 'middle eastern', 'latin']:
        if culture in llm_lower:
            cultural_indicators.append(culture)
    
    if cultural_indicators:
        analysis["cultural_context"] = f"Detected cultural indicators: {', '.join(cultural_indicators)}"
    
    # Extract lifestyle traits
    lifestyle_indicators = []
    for trait in ['athletic', 'sedentary', 'manual', 'intellectual', 'wealthy', 'rural', 'urban', 'outdoor']:
        if trait in llm_lower:
            lifestyle_indicators.append(trait)
    
    if lifestyle_indicators:
        analysis["lifestyle_traits"] = f"Detected lifestyle: {', '.join(lifestyle_indicators)}"
    
    return analysis

def enhance_properties_with_llm_insights(properties, llm_analysis):
    """Enhance property mapping based on LLM insights"""
    enhanced_properties = properties.copy()
    
    # Add cultural-specific properties based on LLM analysis
    cultural_context = llm_analysis.get("cultural_context", "").lower()
    
    if "chinese" in cultural_context or "asian" in cultural_context:
        # Add Asian-specific features
        enhanced_properties["L1_Asian"] = 0.8
        enhanced_properties["L2_Asian_Eyes_TypeAlmond_max"] = 0.7
        
    elif "indian" in cultural_context or "south asian" in cultural_context:
        # Add South Asian features
        enhanced_properties["L1_Asian"] = 0.8
        enhanced_properties["L2_Asian_Eyes_Size_max"] = 0.6
        
    elif "african" in cultural_context:
        # Add African features
        enhanced_properties["L1_African"] = 0.8
        enhanced_properties["L2_African_Nose_BaseSizeX_max"] = 0.7
        
    # Add lifestyle-based properties
    lifestyle_traits = llm_analysis.get("lifestyle_traits", "").lower()
    
    if "athletic" in lifestyle_traits:
        enhanced_properties["L2__Arms_UpperarmMass-UpperarmTone_max-max"] = 0.8
        enhanced_properties["L2__Shoulders_Mass-Tone_max-max"] = 0.8
        
    elif "sedentary" in lifestyle_traits:
        enhanced_properties["L2__Stomach_LocalFat_max"] = 0.7
        enhanced_properties["L2__Body_Size_max"] = 0.6
        
    return enhanced_properties

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-blender', methods=['POST'])
def start_blender():
    global blender_started_once
    
    if is_blender_responsive():
        return jsonify({"success": True, "message": "Blender is already running and responsive!"})
    
    result = start_blender_with_model()
    return jsonify(result) if result["success"] else (jsonify(result), 500)

@app.route('/generate', methods=['POST'])
def generate_character():
    global last_successful_generation
    
    try:
        user_prompt = request.json.get('prompt', '')
        
        if not user_prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        if not blender_started_once:
            return jsonify({
                "error": "Please start Blender first using the 'Start Blender' button."
            }), 400

        print(f"🎭 Processing character description: {user_prompt}")
        
        # STEP 1: NLP-based property mapping
        print("🔍 Step 1: Analyzing prompt with NLP...")
        nlp_properties = property_analyzer.map_to_properties(user_prompt)
        print(f"✓ NLP mapped {len(nlp_properties)} properties")
        
        # STEP 2: Enhance with LLM analysis
        print("🧠 Step 2: Enhancing analysis with Mistral LLM...")
        final_properties, llm_analysis = enhance_analysis_with_llm(user_prompt, nlp_properties)
        print(f"✓ Final mapping: {len(final_properties)} properties")
        
        # STEP 3: Prepare data for Blender
        structured_data = {
            "properties": final_properties,
            "analysis": llm_analysis,
            "prompt": user_prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create request for Blender
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": user_prompt,
            "structured_data": structured_data,
            "status": "pending"
        }
        
        # Write request to file
        with open(REQUEST_FILE, 'w') as f:
            json.dump(request_data, f)
        
        # Wait for response
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(RESPONSE_FILE):
                with open(RESPONSE_FILE, 'r') as f:
                    response_data = json.load(f)
                
                os.remove(RESPONSE_FILE)
                last_successful_generation = datetime.now()
                
                return jsonify({
                    "success": True,
                    "message": "Character generated successfully!",
                    "details": response_data,
                    "property_count": len(final_properties),
                    "llm_analysis": llm_analysis.get("analysis", "No LLM analysis available")
                })
            
            time.sleep(0.5)
        
        return jsonify({
            "error": "Timeout waiting for Blender response."
        }), 408
    
    except Exception as e:
        return jsonify({"error": f"Generation Error: {str(e)}"}), 500

@app.route('/status')
def status():
    global blender_started_once, last_successful_generation
    
    blender_probably_running = blender_started_once
    last_gen_time = last_successful_generation.isoformat() if last_successful_generation else None
    
    return jsonify({
        "blender_running": blender_probably_running,
        "blender_started_once": blender_started_once,
        "last_successful_generation": last_gen_time,
        "model_file": MODEL_BLEND_FILE,
        "model_exists": os.path.exists(MODEL_BLEND_FILE),
        "blender_executable": BLENDER_EXECUTABLE,
        "blender_exists": os.path.exists(BLENDER_EXECUTABLE),
        "properties_loaded": len(CHARACTER_PROPERTIES) > 0,
        "llm_available": model is not None,
        "nlp_available": nlp is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/reset-status', methods=['POST'])
def reset_status():
    global blender_started_once, last_successful_generation
    blender_started_once = False
    last_successful_generation = None
    return jsonify({"success": True, "message": "Status reset. You can now start Blender again."})

@app.route('/config')
def config():
    return jsonify({
        "model_file": MODEL_BLEND_FILE,
        "blender_executable": BLENDER_EXECUTABLE,
        "communication_dir": COMMUNICATION_DIR,
        "properties_count": len(CHARACTER_PROPERTIES)
    })

if __name__ == '__main__':
    print("🎭 ENHANCED Character Generator Frontend Starting...")
    print(f"📊 Properties loaded: {len(CHARACTER_PROPERTIES)}")
    print(f"🧠 LLM Device: {DEVICE}")
    print(f"🔍 NLP Available: {nlp is not None}")
    print(f"📁 Communication directory: {COMMUNICATION_DIR}")
    print()
    
    if not os.path.exists(MODEL_BLEND_FILE):
        print(f"⚠  WARNING: Model file not found: {MODEL_BLEND_FILE}")
    
    if not os.path.exists(BLENDER_EXECUTABLE):
        print(f"⚠  WARNING: Blender executable not found: {BLENDER_EXECUTABLE}")
    
    print("🌐 Open http://127.0.0.1:5000 in your browser")
    print("🚀 Click 'Start Blender' once, then generate characters!")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=False)