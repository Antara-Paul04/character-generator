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
        
        # Enhanced feature mapping with cultural and gender context
        self.feature_keywords = {
            # Gender detection
            'female': ['female', 'woman', 'girl', 'lady', 'feminine', 'she', 'her'],
            'male': ['male', 'man', 'boy', 'guy', 'masculine', 'he', 'him'],
            
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
            'large_body': ['large', 'big', 'heavy', 'stocky', 'burly', 'husky', 'wide belly', 'big belly'],
            'tall': ['tall', 'height'],
            'short': ['short', 'small stature', 'petite'],
            
            # Age features
            'young': ['young', 'youthful', 'boyish', 'girlish', 'teenage'],
            'old': ['old', 'aged', 'elderly', 'wrinkled', 'senior', 'aged skin'],
            
            # Cultural/ethnic indicators
            'caucasian': ['caucasian', 'white', 'european', 'western'],
            'asian': ['asian', 'oriental', 'east asian', 'chinese', 'japanese', 'korean'],
            'african': ['african', 'black', 'afro', 'ebony'],
            'indian': ['indian', 'south asian', 'desi'],
            'middle_eastern': ['middle eastern', 'arab', 'persian'],
            'latino': ['latino', 'hispanic', 'mexican', 'brazilian'],
            'latin': ['latin', 'hispanic'],
            
            # Fantasy races
            'elf': ['elf', 'elven', 'pointed ears'],
            'dwarf': ['dwarf', 'dwarven', 'short stature'],
            'anime': ['anime', 'cartoon', 'animated'],
            
            # Lifestyle indicators
            'food_lover': ['foody', 'loves food', 'eats a lot', 'big eater', 'enjoys food', 'food lover'],
            'athletic': ['athletic', 'sports', 'works out', 'gym goer', 'fit', 'active lifestyle'],
            'sedentary': ['sedentary', 'desk job', 'office worker', 'sits all day', 'inactive'],
        }
        
        # Enhanced property mapping with cultural variants
        self.property_mapping = {
            # Gender base
            'female': ['L1_Female'],
            'male': ['L1_Male'],
            
            # Cultural bases
            'asian': ['L1_Asian'],
            'african': ['L1_African'], 
            'caucasian': ['L1_Caucasian'],
            'latin': ['L1_Latin'],
            'elf': ['L1_Elf'],
            'dwarf': ['L1_Dwarf'],
            'anime': ['L1_Anime'],
            
            # Eyes
            'big_eyes': ['L2__Eyes_Size_max', 'L2__Eyes_IrisSize_max'],
            'small_eyes': ['L2__Eyes_Size_min', 'L2__Eyes_IrisSize_min'],
            
            # Nose - with cultural variants
            'sharp_nose': [
                'L2_Caucasian_Nose_TipSize_min', 'L2_Caucasian_Nose_BridgeSizeX_min',
                'L2_Asian_Nose_TipSize_min', 'L2_Asian_Nose_BridgeSizeX_min',
                'L2_African_Nose_TipSize_min', 'L2_African_Nose_BridgeSizeX_min'
            ],
            'wide_nose': [
                'L2_Caucasian_Nose_BaseSizeX_max', 'L2_Caucasian_Nose_BridgeSizeX_max',
                'L2_Asian_Nose_BaseSizeX_max', 'L2_Asian_Nose_BridgeSizeX_max', 
                'L2_African_Nose_BaseSizeX_max', 'L2_African_Nose_BridgeSizeX_max'
            ],
            'long_nose': [
                'L2_Caucasian_Nose_SizeY_max',
                'L2_Asian_Nose_SizeY_max',
                'L2_African_Nose_SizeY_max'
            ],
            'short_nose': [
                'L2_Caucasian_Nose_SizeY_min',
                'L2_Asian_Nose_SizeY_min', 
                'L2_African_Nose_SizeY_min'
            ],
            
            # Lips
            'full_lips': [
                'L2_Caucasian_Mouth_UpperlipVolume_max', 'L2_Caucasian_Mouth_LowerlipVolume_max',
                'L2_Asian_Mouth_UpperlipVolume_max', 'L2_Asian_Mouth_LowerlipVolume_max',
                'L2_African_Mouth_UpperlipVolume_max', 'L2_African_Mouth_LowerlipVolume_max'
            ],
            'thin_lips': [
                'L2_Caucasian_Mouth_UpperlipVolume_min', 'L2_Caucasian_Mouth_LowerlipVolume_min',
                'L2_Asian_Mouth_UpperlipVolume_min', 'L2_Asian_Mouth_LowerlipVolume_min',
                'L2_African_Mouth_UpperlipVolume_min', 'L2_African_Mouth_LowerlipVolume_min'
            ],
            
            # Body type
            'muscular_body': ['L2__Body_Size_max', 'L2__Arms_UpperarmMass-UpperarmTone_max-max'],
            'slim_body': ['L2__Body_Size_min', 'L2__Arms_UpperarmMass-UpperarmTone_min-min'],
            'large_body': ['L2__Body_Size_max', 'L2__Stomach_LocalFat_max', 'L2__Abdomen_Mass-Tone_max-max'],
            'tall': ['L2__Body_Size_max'],
            'short': ['L2__Body_Size_min'],
            
            # Age
            'old': ['L2_Caucasian_Skin_Wrinkles_max', 'L2_Asian_Skin_Wrinkles_max', 'L2_African_Skin_Wrinkles_max'],
            'young': ['L2_Caucasian_Skin_Wrinkles_min', 'L2_Asian_Skin_Wrinkles_min', 'L2_African_Skin_Wrinkles_min'],
            
            # Lifestyle
            'food_lover': ['L2__Stomach_LocalFat_max', 'L2__Abdomen_Mass-Tone_max-max', 'L2__Body_Size_max'],
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
        
        # Determine cultural context for property selection
        cultural_context = None
        for culture in ['asian', 'african', 'caucasian', 'latin', 'elf', 'dwarf', 'anime']:
            if culture in features:
                cultural_context = culture
                break
        
        for feature, intensity in features.items():
            if feature in self.property_mapping:
                properties = self.property_mapping[feature]
                
                # Filter properties based on cultural context if available
                filtered_properties = []
                for prop in properties:
                    if cultural_context and cultural_context in prop.lower():
                        # Prefer culturally specific properties
                        filtered_properties.append(prop)
                    elif not cultural_context or cultural_context not in prop.lower():
                        # Use generic properties if no cultural context or property is generic
                        filtered_properties.append(prop)
                
                # If we have culturally filtered properties, use them; otherwise use all
                final_properties = filtered_properties if filtered_properties else properties
                
                for prop in final_properties:
                    # Check if property exists in our dataset
                    if prop in self.properties:
                        property_values[prop] = intensity
                    else:
                        # Try to find similar properties
                        similar_props = [p for p in self.properties if prop in p]
                        for similar_prop in similar_props:
                            property_values[similar_prop] = intensity
        
        return property_values, features

# Initialize the property analyzer
property_analyzer = CharacterPropertyAnalyzer(CHARACTER_PROPERTIES)

# =============================================================================
# CONFIGURATION - LLM LOCAL INFERENCE  
# =============================================================================

# Use a smaller, more reliable model for chat
MODEL_DIR = "microsoft/DialoGPT-small"  # Changed to small for better compatibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

try:
    print(f"Loading model on device: {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Set padding token if it doesn't exist
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
    print(f"✗ Failed to load model: {e}")
    print("⚠️  Continuing with NLP-only analysis")

# System prompt for LLM to enhance property mapping
SYSTEM_PROMPT = """Analyze this character description and identify key traits. Focus on:
- Physical features (eyes, nose, lips, body type)
- Cultural/ethnic background
- Age indicators
- Lifestyle traits

Return a brief analysis highlighting the main characteristics."""

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

def enhance_analysis_with_llm(prompt, nlp_properties, nlp_features):
    """Use LLM to enhance the NLP analysis"""
    if model is None or tokenizer is None:
        print("✗ LLM not available, using NLP analysis only")
        return nlp_properties, {
            "analysis": "LLM not available", 
            "enhanced_features": list(nlp_features.keys()),
            "cultural_context": "",
            "lifestyle_traits": "",
            "llm_used": False
        }
    
    try:
        # Create a simple prompt for DialoGPT (it doesn't support chat templates well)
        input_text = f"{SYSTEM_PROMPT}\n\nCharacter description: {prompt}\n\nAnalysis:"
        
        # Tokenize with proper attention mask
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        ).to(DEVICE)
        
        print("🧠 Generating LLM analysis...")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Add attention mask
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs.input_ids.shape[1]
        raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        print(f"✓ LLM Raw Output: {raw_output}")
        
        # Parse LLM output to extract additional insights
        llm_analysis = parse_llm_output(raw_output, prompt)
        
        # Enhance properties based on LLM insights
        enhanced_properties = enhance_properties_with_llm_insights(nlp_properties, llm_analysis, nlp_features)
        
        print(f"✓ LLM enhanced {len(enhanced_properties) - len(nlp_properties)} additional properties")
        
        return enhanced_properties, llm_analysis
        
    except Exception as e:
        print(f"✗ LLM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return nlp_properties, {
            "analysis": f"LLM analysis failed: {str(e)}", 
            "enhanced_features": list(nlp_features.keys()),
            "cultural_context": "",
            "lifestyle_traits": "",
            "llm_used": False
        }

def parse_llm_output(raw_output, original_prompt):
    """Parse LLM output to extract structured insights"""
    analysis = {
        "analysis": raw_output.strip(),
        "enhanced_features": [],
        "cultural_context": "",
        "lifestyle_traits": "",
        "llm_used": True,
        "raw_output": raw_output.strip()
    }
    
    # Simple keyword extraction from LLM output
    llm_lower = raw_output.lower()
    prompt_lower = original_prompt.lower()
    
    # Extract cultural context from both LLM output and original prompt
    cultural_indicators = []
    for culture in ['chinese', 'indian', 'japanese', 'korean', 'african', 'european', 'american', 'middle eastern', 'latin', 'asian']:
        if culture in llm_lower or culture in prompt_lower:
            cultural_indicators.append(culture)
    
    if cultural_indicators:
        analysis["cultural_context"] = f"Detected cultural indicators: {', '.join(set(cultural_indicators))}"
    
    # Extract lifestyle traits
    lifestyle_indicators = []
    for trait in ['athletic', 'sedentary', 'manual', 'intellectual', 'wealthy', 'rural', 'urban', 'outdoor', 'active', 'fit']:
        if trait in llm_lower:
            lifestyle_indicators.append(trait)
    
    if lifestyle_indicators:
        analysis["lifestyle_traits"] = f"Detected lifestyle: {', '.join(lifestyle_indicators)}"
    
    # Extract enhanced features from LLM analysis
    enhanced_features = []
    for feature in ['big eyes', 'small eyes', 'sharp nose', 'wide nose', 'full lips', 'thin lips', 'aged', 'young', 'muscular', 'slim']:
        if feature in llm_lower:
            enhanced_features.append(feature)
    
    analysis["enhanced_features"] = enhanced_features
    
    return analysis

def enhance_properties_with_llm_insights(properties, llm_analysis, nlp_features):
    """Enhance property mapping based on LLM insights"""
    enhanced_properties = properties.copy()
    
    # Add cultural-specific properties based on LLM analysis
    cultural_context = llm_analysis.get("cultural_context", "").lower()
    
    if "asian" in cultural_context or any(x in cultural_context for x in ['chinese', 'japanese', 'korean']):
        # Add Asian-specific features
        if 'L1_Asian' not in enhanced_properties:
            enhanced_properties["L1_Asian"] = 0.8
        if 'big_eyes' in nlp_features and 'L2_Asian_Eyes_Size_max' not in enhanced_properties:
            enhanced_properties["L2_Asian_Eyes_Size_max"] = nlp_features['big_eyes']
        
    elif "indian" in cultural_context or "south asian" in cultural_context:
        # Add South Asian features
        if 'L1_Asian' not in enhanced_properties:
            enhanced_properties["L1_Asian"] = 0.8
        if 'big_eyes' in nlp_features and 'L2_Asian_Eyes_Size_max' not in enhanced_properties:
            enhanced_properties["L2_Asian_Eyes_Size_max"] = nlp_features['big_eyes']
        
    elif "african" in cultural_context:
        # Add African features
        if 'L1_African' not in enhanced_properties:
            enhanced_properties["L1_African"] = 0.8
        if 'wide_nose' in nlp_features and 'L2_African_Nose_BaseSizeX_max' not in enhanced_properties:
            enhanced_properties["L2_African_Nose_BaseSizeX_max"] = nlp_features['wide_nose']
    
    # Add properties based on LLM enhanced features
    enhanced_features = llm_analysis.get("enhanced_features", [])
    for feature in enhanced_features:
        if 'aged' in feature and 'old' not in nlp_features:
            # Add aging properties if LLM detected age but NLP didn't
            enhanced_properties["L2_Caucasian_Skin_Wrinkles_max"] = 0.7
        elif 'muscular' in feature and 'muscular_body' not in nlp_features:
            enhanced_properties["L2__Arms_UpperarmMass-UpperarmTone_max-max"] = 0.6
        
    # Add lifestyle-based properties
    lifestyle_traits = llm_analysis.get("lifestyle_traits", "").lower()
    
    if "athletic" in lifestyle_traits or "active" in lifestyle_traits or "fit" in lifestyle_traits:
        if 'L2__Arms_UpperarmMass-UpperarmTone_max-max' not in enhanced_properties:
            enhanced_properties["L2__Arms_UpperarmMass-UpperarmTone_max-max"] = 0.7
        if 'L2__Shoulders_Mass-Tone_max-max' not in enhanced_properties:
            enhanced_properties["L2__Shoulders_Mass-Tone_max-max"] = 0.7
        
    elif "sedentary" in lifestyle_traits:
        if 'L2__Stomach_LocalFat_max' not in enhanced_properties:
            enhanced_properties["L2__Stomach_LocalFat_max"] = 0.6
        if 'L2__Body_Size_max' not in enhanced_properties:
            enhanced_properties["L2__Body_Size_max"] = 0.5
        
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
        nlp_properties, nlp_features = property_analyzer.map_to_properties(user_prompt)
        print(f"✓ NLP mapped {len(nlp_properties)} properties")
        print("📊 NLP Property-Value Map:")
        for prop, value in nlp_properties.items():
            print(f"   {prop}: {value:.2f}")
        
        # STEP 2: Enhance with LLM analysis
        print("🧠 Step 2: Enhancing analysis with LLM...")
        final_properties, llm_analysis = enhance_analysis_with_llm(user_prompt, nlp_properties, nlp_features)
        
        # Show what LLM added
        llm_added = set(final_properties.keys()) - set(nlp_properties.keys())
        if llm_added:
            print(f"✓ LLM added {len(llm_added)} properties: {list(llm_added)}")
        else:
            print("✓ LLM analysis completed (no additional properties added)")
        
        print(f"✓ Final mapping: {len(final_properties)} properties")
        print("📊 Final Property-Value Map:")
        for prop, value in final_properties.items():
            source = "LLM" if prop in llm_added else "NLP"
            print(f"   {prop}: {value:.2f} [{source}]")
        
        # STEP 3: Prepare data for Blender
        structured_data = {
            "properties": final_properties,
            "analysis": llm_analysis,
            "prompt": user_prompt,
            "timestamp": datetime.now().isoformat(),
            "property_map": final_properties,
            "llm_used": llm_analysis.get("llm_used", False)
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
                    "property_map": final_properties,
                    "llm_analysis": llm_analysis,
                    "features_detected": list(nlp_features.keys()),
                    "llm_used": llm_analysis.get("llm_used", False)
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