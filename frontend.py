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
            'female': ['female', 'woman', 'girl', 'lady', 'feminine', 'she', 'her', 'wife', 'mother', 'daughter', 'sister'],
            'male': ['male', 'man', 'boy', 'guy', 'masculine', 'he', 'him', 'husband', 'father', 'son', 'brother'],
            
            # Direct physical features
            'big_eyes': ['big eyes', 'large eyes', 'wide eyes', 'expressive eyes', 'doe eyes'],
            'small_eyes': ['small eyes', 'narrow eyes', 'squinty eyes', 'beady eyes'],
            'sharp_nose': ['sharp nose', 'pointed nose', 'angular nose', 'refined nose'],
            'wide_nose': ['wide nose', 'broad nose', 'flat nose', 'button nose'],
            'long_nose': ['long nose', 'prominent nose', 'aquiline nose'],
            'short_nose': ['short nose', 'small nose', 'snub nose'],
            'full_lips': ['full lips', 'plump lips', 'big lips', 'luscious lips', 'pouty lips', 'thick lips'],
            'thin_lips': ['thin lips', 'small lips', 'narrow lips'],
            'strong_jaw': ['strong jaw', 'defined jaw', 'angular jaw', 'square jaw', 'chiseled jaw'],
            'soft_jaw': ['soft jaw', 'round jaw', 'gentle jaw', 'delicate jaw'],
            'wide_jaw': ['wide jaw', 'broad jaw'],
            'narrow_jaw': ['narrow jaw', 'thin jaw'],
            'prominent_chin': ['strong chin', 'prominent chin', 'defined chin', 'cleft chin'],
            'weak_chin': ['weak chin', 'receding chin', 'small chin'],
            'high_cheekbones': ['high cheekbones', 'prominent cheeks', 'defined cheeks', 'sharp cheekbones'],
            'round_face': ['round face', 'full face', 'chubby face', 'moon face', 'baby face'],
            'angular_face': ['angular face', 'sharp face', 'defined face', 'chiseled face'],
            'oval_face': ['oval face', 'elongated face'],
            'heart_face': ['heart shaped face', 'heart face'],
            
            # Body type features
            'muscular_body': ['muscular', 'athletic', 'toned', 'fit', 'built', 'ripped', 'strong'],
            'slim_body': ['slim', 'thin', 'lean', 'slender', 'willowy', 'skinny'],
            'large_body': ['large', 'big', 'heavy', 'stocky', 'burly', 'husky', 'wide belly', 'big belly', 'overweight'],
            'curvy_body': ['curvy', 'hourglass', 'voluptuous', 'full figured'],
            'tall': ['tall', 'height', 'towering'],
            'short': ['short', 'small stature', 'petite', 'compact'],
            'average_height': ['average height', 'medium height', 'normal height'],
            
            # Explicit Tone and Mass Keywords
            'very_toned': ['very toned', 'defined muscle', 'high tone', 'sculpted', 'low fat', 'chiseled body', 'six pack', 'abs'],
            'low_tone': ['flabby', 'soft body', 'low muscle tone', 'out of shape', 'untrained body', 'saggy'],
            'high_mass': ['chubby', 'heavy build', 'high fat', 'plump', 'full figured', 'rotund', 'portly'],
            'low_mass': ['skinny', 'underweight', 'gaunt', 'bony', 'emaciated'],
            
            # Age features
            'young': ['young', 'youthful', 'boyish', 'girlish', 'teenage', 'teen', 'adolescent', 'child', 'kid'],
            'middle_aged': ['middle aged', 'mature', 'mid life', 'forties', 'fifties'],
            'old': ['old', 'aged', 'elderly', 'wrinkled', 'senior', 'aged skin', 'grandparent'],
            
            # Cultural/ethnic indicators - ENHANCED
            'caucasian': ['caucasian', 'white', 'european', 'western', 'pale skin', 'fair skin'],
            'asian': ['asian', 'east asian', 'chinese', 'japanese', 'korean', 'vietnamese', 'thai'],
            'south_asian': ['indian', 'south asian', 'desi', 'pakistani', 'bangladeshi', 'sri lankan'],
            'african': ['african', 'black', 'afro', 'ebony', 'dark skin', 'nigerian', 'ethiopian', 'kenyan'],
            'middle_eastern': ['middle eastern', 'arab', 'persian', 'iranian', 'turkish', 'lebanese'],
            'latino': ['latino', 'hispanic', 'mexican', 'brazilian', 'colombian', 'argentinian'],
            'latin': ['latin', 'hispanic', 'spanish'],
            
            # Fantasy races
            'elf': ['elf', 'elven', 'pointed ears'],
            'dwarf': ['dwarf', 'dwarven', 'short stature'],
            'anime': ['anime', 'cartoon', 'animated'],
            
            # Lifestyle indicators
            'food_lover': ['foody', 'loves food', 'eats a lot', 'big eater', 'enjoys food', 'food lover'],
            'athletic': ['athletic', 'sports', 'works out', 'gym goer', 'fit', 'active lifestyle', 'runner', 'swimmer'],
            'sedentary': ['sedentary', 'desk job', 'office worker', 'sits all day', 'inactive'],
            
            # Additional detailed features
            'wide_shoulders': ['wide shoulders', 'broad shoulders', 'strong shoulders'],
            'narrow_shoulders': ['narrow shoulders', 'sloping shoulders', 'small shoulders'],
            'long_neck': ['long neck', 'graceful neck', 'swan neck'],
            'short_neck': ['short neck', 'thick neck'],
            'large_hands': ['large hands', 'big hands', 'strong hands'],
            'small_hands': ['small hands', 'delicate hands', 'petite hands'],
            'large_feet': ['large feet', 'big feet'],
            'small_feet': ['small feet', 'petite feet'],
        }
        
        # ENHANCED property mapping with comprehensive coverage
        self.property_mapping = {
            # Eyes - comprehensive
            'big_eyes': ['L2__Eyes_Size_max', 'L2__Eyes_IrisSize_max'],
            'small_eyes': ['L2__Eyes_Size_min', 'L2__Eyes_IrisSize_min'],
            
            # Nose - all cultural variants
            'sharp_nose': [
                'L2_Caucasian_Nose_TipSize_min', 'L2_Caucasian_Nose_BridgeSizeX_min',
                'L2_Asian_Nose_TipSize_min', 'L2_Asian_Nose_BridgeSizeX_min',
                'L2_African_Nose_TipSize_min', 'L2_African_Nose_BridgeSizeX_min',
                'L2_Anime_Nose_TipSize_min', 'L2_Elf_Nose_TipSize_min', 'L2_Dwarf_Nose_TipSize_min'
            ],
            'wide_nose': [
                'L2_Caucasian_Nose_BaseSizeX_max', 'L2_Caucasian_Nose_BridgeSizeX_max',
                'L2_Asian_Nose_BaseSizeX_max', 'L2_Asian_Nose_BridgeSizeX_max', 
                'L2_African_Nose_BaseSizeX_max', 'L2_African_Nose_BridgeSizeX_max',
                'L2_Anime_Nose_BaseSizeX_max', 'L2_Elf_Nose_BaseSizeX_max', 'L2_Dwarf_Nose_BaseSizeX_max'
            ],
            'long_nose': [
                'L2_Caucasian_Nose_SizeY_max', 'L2_Asian_Nose_SizeY_max', 'L2_African_Nose_SizeY_max',
                'L2_Anime_Nose_SizeY_max', 'L2_Elf_Nose_SizeY_max', 'L2_Dwarf_Nose_SizeY_max'
            ],
            'short_nose': [
                'L2_Caucasian_Nose_SizeY_min', 'L2_Asian_Nose_SizeY_min', 'L2_African_Nose_SizeY_min',
                'L2_Anime_Nose_SizeY_min', 'L2_Elf_Nose_SizeY_min', 'L2_Dwarf_Nose_SizeY_min'
            ],
            
            # Lips - all cultural variants
            'full_lips': [
                'L2_Caucasian_Mouth_UpperlipVolume_max', 'L2_Caucasian_Mouth_LowerlipVolume_max',
                'L2_Asian_Mouth_UpperlipVolume_max', 'L2_Asian_Mouth_LowerlipVolume_max',
                'L2_African_Mouth_UpperlipVolume_max', 'L2_African_Mouth_LowerlipVolume_max',
                'L2_Anime_Mouth_UpperlipVolume_max', 'L2_Anime_Mouth_LowerlipVolume_max',
                'L2_Elf_Mouth_UpperlipVolume_max', 'L2_Elf_Mouth_LowerlipVolume_max'
            ],
            'thin_lips': [
                'L2_Caucasian_Mouth_UpperlipVolume_min', 'L2_Caucasian_Mouth_LowerlipVolume_min',
                'L2_Asian_Mouth_UpperlipVolume_min', 'L2_Asian_Mouth_LowerlipVolume_min',
                'L2_African_Mouth_UpperlipVolume_min', 'L2_African_Mouth_LowerlipVolume_min',
                'L2_Anime_Mouth_UpperlipVolume_min', 'L2_Anime_Mouth_LowerlipVolume_min'
            ],
            
            # Jaw and chin - comprehensive
            'strong_jaw': [
                'L2_Caucasian_Jaw_Prominence_max', 'L2_Asian_Jaw_Prominence_max', 'L2_African_Jaw_Prominence_max',
                'L2_Anime_Jaw_Prominence_max', 'L2_Elf_Jaw_Prominence_max', 'L2_Dwarf_Jaw_Prominence_max'
            ],
            'soft_jaw': [
                'L2_Caucasian_Jaw_Prominence_min', 'L2_Asian_Jaw_Prominence_min', 'L2_African_Jaw_Prominence_min'
            ],
            'prominent_chin': [
                'L2_Caucasian_Chin_Prominence_max', 'L2_Asian_Chin_Prominence_max', 'L2_African_Chin_Prominence_max'
            ],
            'high_cheekbones': [
                'L2_Caucasian_Cheeks_Zygom_max', 'L2_Asian_Cheeks_Zygom_max', 'L2_African_Cheeks_Zygom_max',
                'L2_Anime_Cheeks_Zygom_max', 'L2_Elf_Cheeks_Zygom_max', 'L2_Dwarf_Cheeks_Zygom_max'
            ],
            
            # Body type - detailed
            'muscular_body': [
                'L2__Body_Size_max', 'L2__Arms_UpperarmMass-UpperarmTone_max-max',
                'L2__Legs_UpperlegsMass-UpperlegsTone_max-max', 'L2__Chest_Girth_max',
                'L2__Shoulders_Mass-Tone_max-max', 'L2__Torso_Mass-Tone_max-max'
            ],
            'slim_body': [
                'L2__Body_Size_min', 'L2__Arms_UpperarmMass-UpperarmTone_min-min',
                'L2__Torso_Mass-Tone_min-min', 'L2__Waist_Size_min'
            ],
            'large_body': [
                'L2__Body_Size_max', 'L2__Stomach_LocalFat_max', 'L2__Abdomen_Mass-Tone_max-max',
                'L2__Torso_Mass-Tone_max-min', 'L2__Waist_Size_max', 'L2__Chest_Girth_max'
            ],
            'curvy_body': [
                'L2__Pelvis_Girth_max', 'L2__Chest_SizeZ_max', 'L2__Waist_Size_min',
                'L2__Pelvis_GluteusMass-GluteusTone_max-max'
            ],
            'tall': ['L2__Body_Size_max', 'L2__Legs_UpperlegLength_max', 'L2__Torso_Length_max'],
            'short': ['L2__Body_Size_min', 'L2__Legs_UpperlegLength_min', 'L2__Torso_Length_min'],
            
            # Explicit Mass/Tone Mappings
            'very_toned': [
                'L2__Arms_UpperarmMass-UpperarmTone_max-max',
                'L2__Legs_UpperlegsMass-UpperlegsTone_max-max',
                'L2__Shoulders_Mass-Tone_max-max', 
                'L2__Pelvis_GluteusMass-GluteusTone_max-max',
                'L2__Abdomen_Mass-Tone_max-max',
                'L2__Torso_Mass-Tone_max-max'
            ],
            'low_tone': [
                'L2__Arms_UpperarmMass-UpperarmTone_min-min', 
                'L2__Legs_UpperlegsMass-UpperlegsTone_min-min',
                'L2__Shoulders_Mass-Tone_min-min', 
                'L2__Pelvis_GluteusMass-GluteusTone_min-min',
                'L2__Torso_Mass-Tone_min-min'
            ],
            'high_mass': [
                'L2__Body_Size_max', 'L2__Stomach_LocalFat_max', 
                'L2__Torso_Mass-Tone_max-min', 'L2__Abdomen_Mass-Tone_max-max',
                'L2__Pelvis_GluteusMass-GluteusTone_max-min'
            ],
            'low_mass': [
                'L2__Body_Size_min', 'L2__Hands_Mass-Tone_min-min',
                'L2__Torso_Mass-Tone_min-min', 'L2__Arms_UpperarmMass-UpperarmTone_min-min'
            ],
            
            # Shoulders and upper body
            'wide_shoulders': [
                'L2__Shoulders_Length_max', 'L2__Shoulders_Size_max', 'L2__Shoulders_Mass-Tone_max-max'
            ],
            'narrow_shoulders': [
                'L2__Shoulders_Length_min', 'L2__Shoulders_Size_min'
            ],
            
            # Neck
            'long_neck': ['L2__Neck_Length_max'],
            'short_neck': ['L2__Neck_Length_min'],
            
            # Hands and feet
            'large_hands': ['L2__Hands_Size_max', 'L2__Hands_Length_max'],
            'small_hands': ['L2__Hands_Size_min', 'L2__Hands_Length_min'],
            'large_feet': ['L2__Feet_Size_max', 'L2__Feet_SizeX_max', 'L2__Feet_SizeY_max'],
            'small_feet': ['L2__Feet_Size_min', 'L2__Feet_SizeX_min', 'L2__Feet_SizeY_min'],
            
            # Lifestyle
            'food_lover': ['L2__Stomach_LocalFat_max', 'L2__Abdomen_Mass-Tone_max-max', 'L2__Body_Size_max'],
            'athletic': ['L2__Arms_UpperarmMass-UpperarmTone_max-max', 'L2__Legs_UpperlegsMass-UpperlegsTone_max-max'],
        }
        
        # Ethnicity to base property mapping
        self.ethnicity_base_map = {
            'caucasian': 'L1_Caucasian',
            'asian': 'L1_Asian',
            'south_asian': 'L1_Asian',  # Map to Asian as best fit
            'african': 'L1_African',
            'middle_eastern': 'L1_Caucasian',  # Map to Caucasian as best fit
            'latino': 'L1_Latin',
            'latin': 'L1_Latin',
            'elf': 'L1_Elf',
            'dwarf': 'L1_Dwarf',
            'anime': 'L1_Anime'
        }
        
        # Default properties to ensure minimum 30 properties
        self.default_properties = [
            'L2__Head_Size_max', 'L2__Head_Size_min',
            'L2__Body_Size_max', 'L2__Body_Size_min',
            'L2__Eyes_Size_max', 'L2__Eyes_Size_min',
            'L2__Torso_Length_max', 'L2__Torso_Length_min',
            'L2__Legs_UpperlegLength_max', 'L2__Legs_UpperlegLength_min',
            'L2__Arms_UpperarmLength_max', 'L2__Arms_UpperarmLength_min',
            'L2__Hands_Size_max', 'L2__Hands_Size_min',
            'L2__Feet_Size_max', 'L2__Feet_Size_min'
        ]
        
        self.intensity_modifiers = {
            'slightly': 0.3, 'somewhat': 0.4, 'moderately': 0.6,
            'very': 0.8, 'extremely': 0.9, 'incredibly': 1.0,
            'quite': 0.7, 'rather': 0.6, 'fairly': 0.5
        }
        
        self.default_intensity = 0.65

    def detect_gender(self, prompt):
        """Detect gender from prompt with confidence scoring"""
        prompt_lower = prompt.lower()
        
        female_score = 0
        male_score = 0
        
        for keyword in self.feature_keywords['female']:
            if keyword in prompt_lower:
                female_score += 1
        
        for keyword in self.feature_keywords['male']:
            if keyword in prompt_lower:
                male_score += 1
        
        # Default to male if no clear indicators
        if female_score > male_score:
            return 'female', female_score
        else:
            return 'male', male_score if male_score > 0 else 1

    def detect_ethnicity(self, prompt):
        """Detect ethnicity with fallback to best fit"""
        prompt_lower = prompt.lower()
        
        ethnicity_scores = {}
        
        for ethnicity in ['caucasian', 'asian', 'south_asian', 'african', 'middle_eastern', 'latino', 'latin', 'elf', 'dwarf', 'anime']:
            score = 0
            if ethnicity in self.feature_keywords:
                for keyword in self.feature_keywords[ethnicity]:
                    if keyword in prompt_lower:
                        score += 1
            ethnicity_scores[ethnicity] = score
        
        # Get best match
        best_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get)
        best_score = ethnicity_scores[best_ethnicity]
        
        # If no ethnicity detected, default to caucasian
        if best_score == 0:
            return 'caucasian', 0, 'default'
        
        return best_ethnicity, best_score, 'detected'

    def get_cultural_property_prefix(self, ethnicity):
        """Get the appropriate property prefix for detected ethnicity"""
        ethnicity_prefix_map = {
            'caucasian': 'Caucasian',
            'asian': 'Asian',
            'south_asian': 'Asian',
            'african': 'African',
            'middle_eastern': 'Caucasian',
            'latino': 'Latin',
            'latin': 'Latin',
            'elf': 'Elf',
            'dwarf': 'Dwarf',
            'anime': 'Anime'
        }
        return ethnicity_prefix_map.get(ethnicity, 'Caucasian')

    def filter_properties_by_culture(self, properties, cultural_context):
        """Filter properties to match detected culture"""
        if not cultural_context:
            return properties
        
        cultural_prefix = self.get_cultural_property_prefix(cultural_context)
        filtered = []
        
        for prop in properties:
            # Check if property is culturally specific
            if any(culture in prop for culture in ['Caucasian', 'Asian', 'African', 'Anime', 'Elf', 'Dwarf']):
                # Only include if it matches our detected culture
                if cultural_prefix in prop:
                    filtered.append(prop)
            else:
                # Include generic properties
                filtered.append(prop)
        
        return filtered if filtered else properties

    def ensure_minimum_properties(self, property_values, detected_features, ethnicity):
        """Ensure at least 30 properties are selected"""
        cultural_prefix = self.get_cultural_property_prefix(ethnicity)
        
        if len(property_values) >= 30:
            return property_values
        
        print(f"⚠️  Only {len(property_values)} properties detected. Adding defaults to reach 30...")
        
        # Add default properties with moderate values
        for default_prop in self.default_properties:
            if default_prop not in property_values:
                property_values[default_prop] = 0.5
                if len(property_values) >= 30:
                    break
        
        # Add culturally appropriate facial features if still not enough
        if len(property_values) < 30:
            cultural_features = [
                f'L2_{cultural_prefix}_Eyes_Size_max',
                f'L2_{cultural_prefix}_Nose_SizeY_max',
                f'L2_{cultural_prefix}_Mouth_SizeX_max',
                f'L2_{cultural_prefix}_Cheeks_Zygom_max',
                f'L2_{cultural_prefix}_Jaw_Angle_max',
                f'L2_{cultural_prefix}_Chin_SizeZ_max',
                f'L2_{cultural_prefix}_Forehead_SizeX_max',
                f'L2_{cultural_prefix}_Ears_SizeY_max'
            ]
            
            for feature in cultural_features:
                if feature in self.properties and feature not in property_values:
                    property_values[feature] = 0.55
                    if len(property_values) >= 30:
                        break
        
        print(f"✓ Enhanced to {len(property_values)} properties")
        return property_values

    def analyze_prompt_with_nlp(self, prompt):
        """Enhanced NLP analysis with better phrase matching"""
        if self.nlp is None:
            return self._simple_analysis(prompt)
        
        features = {}
        prompt_lower = prompt.lower()
        
        print("🔍 Starting enhanced NLP analysis...")
        
        for feature, keywords in self.feature_keywords.items():
            best_intensity = 0.0
            
            for keyword in sorted(keywords, key=len, reverse=True):
                if keyword in prompt_lower:
                    intensity = self.default_intensity
                    match_index = prompt_lower.find(keyword)
                    
                    for modifier, mod_intensity in self.intensity_modifiers.items():
                        modifier_phrase = f"{modifier} {keyword}"
                        if prompt_lower.find(modifier_phrase) == match_index - (len(modifier) + 1):
                            intensity = mod_intensity
                            break
                    
                    best_intensity = max(best_intensity, intensity)
            
            if best_intensity > 0.0:
                features[feature] = best_intensity
        
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
        """Enhanced property mapping with gender and ethnicity detection"""
        # Detect gender and ethnicity
        gender, gender_confidence = self.detect_gender(prompt)
        ethnicity, ethnicity_confidence, ethnicity_source = self.detect_ethnicity(prompt)
        
        print(f"🎭 Gender detected: {gender} (confidence: {gender_confidence})")
        print(f"🌍 Ethnicity detected: {ethnicity} (confidence: {ethnicity_confidence}, source: {ethnicity_source})")
        
        # Analyze features
        features = self.analyze_prompt_with_nlp(prompt)
        
        # Add gender and ethnicity to features
        features['gender'] = gender
        features['ethnicity'] = ethnicity
        
        print(f"✓ NLP Analysis - Detected {len(features)} features: {list(features.keys())}")
        
        property_values = {}
        
        # Add ethnicity base property if available
        if ethnicity in self.ethnicity_base_map:
            base_property = self.ethnicity_base_map[ethnicity]
            if base_property in self.properties:
                property_values[base_property] = 0.85
                print(f"✓ Added ethnicity base: {base_property}")
        
        # Map features to properties
        for feature, intensity in features.items():
            if feature in ['gender', 'ethnicity']:
                continue
            
            if feature in self.property_mapping:
                properties = self.property_mapping[feature]
                
                # Filter properties based on detected ethnicity
                filtered_properties = self.filter_properties_by_culture(properties, ethnicity)
                
                for prop in filtered_properties:
                    if prop in self.properties:
                        property_values[prop] = intensity
        
        # Ensure minimum 30 properties
        property_values = self.ensure_minimum_properties(property_values, features, ethnicity)
        
        return property_values, features, gender, ethnicity

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

# Add this to replace the existing generate_character route in frontend.py

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

        print(f"\n{'='*70}")
        print(f"🎭 PROCESSING NEW CHARACTER REQUEST")
        print(f"{'='*70}")
        print(f"Prompt: {user_prompt}")
        print(f"{'='*70}\n")
        
        # STEP 1: Detect gender and ethnicity first
        print("🔍 Step 1: Detecting gender and ethnicity...")
        gender, gender_confidence = property_analyzer.detect_gender(user_prompt)
        ethnicity, ethnicity_confidence, ethnicity_source = property_analyzer.detect_ethnicity(user_prompt)
        
        print(f"✓ Gender: {gender.upper()} (confidence: {gender_confidence})")
        print(f"✓ Ethnicity: {ethnicity.upper()} (confidence: {ethnicity_confidence}, source: {ethnicity_source})")
        
        # STEP 2: NLP-based property mapping with detected context
        print("\n🔍 Step 2: Analyzing prompt with NLP...")
        nlp_properties, nlp_features, detected_gender, detected_ethnicity = property_analyzer.map_to_properties(user_prompt)
        
        print(f"✓ NLP mapped {len(nlp_properties)} properties")
        print(f"✓ Features detected: {list(nlp_features.keys())}")
        
        # Ensure minimum 30 properties
        if len(nlp_properties) < 30:
            print(f"\n⚠️  Only {len(nlp_properties)} properties detected. Enhancing...")
            nlp_properties = property_analyzer.ensure_minimum_properties(
                nlp_properties, nlp_features, detected_ethnicity
            )
        
        print(f"\n📊 Property Breakdown:")
        print(f"  - Total properties: {len(nlp_properties)}")
        print(f"  - Gender: {detected_gender}")
        print(f"  - Ethnicity: {detected_ethnicity}")
        
        # Show sample of properties
        sample_props = list(nlp_properties.items())[:10]
        print(f"\n📋 Sample Properties (showing 10/{len(nlp_properties)}):")
        for prop, value in sample_props:
            print(f"   {prop}: {value:.2f}")
        
        # STEP 3: Enhance with LLM analysis
        print("\n🧠 Step 3: Enhancing analysis with LLM...")
        final_properties, llm_analysis = enhance_analysis_with_llm(
            user_prompt, nlp_properties, nlp_features
        )
        
        # Show what LLM added
        llm_added = set(final_properties.keys()) - set(nlp_properties.keys())
        if llm_added:
            print(f"✓ LLM added {len(llm_added)} properties")
            if len(llm_added) <= 10:
                print(f"  Added properties: {list(llm_added)}")
        else:
            print("✓ LLM analysis completed (no additional properties)")
        
        # Final count
        print(f"\n✅ FINAL MAPPING: {len(final_properties)} properties")
        print(f"   Gender: {detected_gender.upper()}")
        print(f"   Ethnicity: {detected_ethnicity.upper()}")
        
        # STEP 4: Prepare data for Blender
        structured_data = {
            "properties": final_properties,
            "analysis": llm_analysis,
            "prompt": user_prompt,
            "timestamp": datetime.now().isoformat(),
            "property_map": final_properties,
            "llm_used": llm_analysis.get("llm_used", False),
            "gender": detected_gender,
            "ethnicity": detected_ethnicity,
            "property_count": len(final_properties),
            "features_detected": list(nlp_features.keys())
        }
        
        # Create request for Blender
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": user_prompt,
            "structured_data": structured_data,
            "status": "pending"
        }
        
        # Write request to file
        print(f"\n📤 Sending request to Blender...")
        with open(REQUEST_FILE, 'w') as f:
            json.dump(request_data, f, indent=2)
        
        # Wait for response
        timeout = 30
        start_time = time.time()
        
        print(f"⏳ Waiting for Blender response (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if os.path.exists(RESPONSE_FILE):
                with open(RESPONSE_FILE, 'r') as f:
                    response_data = json.load(f)
                
                os.remove(RESPONSE_FILE)
                last_successful_generation = datetime.now()
                
                print(f"\n✅ SUCCESS! Character generated in Blender")
                print(f"{'='*70}\n")
                
                return jsonify({
                    "success": True,
                    "message": f"✓ {detected_gender.capitalize()} {detected_ethnicity} character generated!",
                    "details": response_data,
                    "property_count": len(final_properties),
                    "property_map": final_properties,
                    "llm_analysis": llm_analysis,
                    "features_detected": list(nlp_features.keys()),
                    "llm_used": llm_analysis.get("llm_used", False),
                    "gender": detected_gender,
                    "ethnicity": detected_ethnicity,
                    "character_object": response_data.get("character_object", "unknown")
                })
            
            time.sleep(0.5)
        
        print(f"\n❌ TIMEOUT: No response from Blender after {timeout}s")
        print(f"{'='*70}\n")
        
        return jsonify({
            "error": "Timeout waiting for Blender response. Is Blender still running?"
        }), 408
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        
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