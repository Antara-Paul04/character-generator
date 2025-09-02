import bpy
import re
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional

# =============================================================================
# --- NLP AND SEMANTIC MAPPING ---
# =============================================================================
# Enhanced semantic mappings for personality traits and descriptive terms
PERSONALITY_TO_FEATURES = {
    "intelligent": {
        "forehead": {"high": 0.7},
        "eyes": {"focused": 0.6, "sharp": 0.7},
        "nose": {"refined": 0.5},
        "jaw": {"defined": 0.6}
    },
    "wise": {
        "forehead": {"high": 0.8},
        "eyes": {"deep": 0.7},
        "cheeks": {"hollow": 0.4},
        "face": {"aged": 0.5}
    },
    "strong": {
        "jaw": {"strong": 0.8, "wide": 0.6},
        "chin": {"prominent": 0.7},
        "cheeks": {"defined": 0.6},
        "neck": {"thick": 0.5}
    },
    "gentle": {
        "eyes": {"soft": 0.6, "large": 0.5},
        "lips": {"full": 0.5},
        "face": {"round": 0.4},
        "jaw": {"soft": 0.3}
    },
    "fierce": {
        "eyes": {"narrow": 0.7, "intense": 0.8},
        "eyebrows": {"thick": 0.6, "low": 0.5},
        "jaw": {"angular": 0.7},
        "nose": {"sharp": 0.6}
    },
    "kind": {
        "eyes": {"warm": 0.6, "crinkled": 0.4},
        "mouth": {"upturned": 0.5},
        "face": {"soft": 0.5}
    }
}
# Age-related feature mappings
AGE_MAPPINGS = {
    "young": {
        "face": {"smooth": 0.8, "full": 0.6},
        "eyes": {"bright": 0.7},
        "skin": {"tight": 0.8}
    },
    "old": {
        "face": {"wrinkled": 0.7, "hollow": 0.5},
        "eyes": {"droopy": 0.6},
        "skin": {"loose": 0.6}
    },
    "middle-aged": {
        "face": {"mature": 0.5},
        "eyes": {"experienced": 0.4}
    }
}
# Enhanced synonym dictionary for better NLP understanding
SYNONYM_MAP = {
    # Intelligence synonyms
    "smart": "intelligent",
    "clever": "intelligent",
    "bright": "intelligent",
    "sharp": "intelligent",
   
    # Strength synonyms
    "powerful": "strong",
    "muscular": "strong",
    "robust": "strong",
   
    # Beauty/attractiveness
    "beautiful": "attractive",
    "handsome": "attractive",
    "pretty": "attractive",
   
    # Age synonyms
    "elderly": "old",
    "senior": "old",
    "youthful": "young",
    "teenage": "young"
}
# Contextual feature interpretation
CONTEXTUAL_FEATURES = {
    "businessman": {
        "overall": "professional",
        "jaw": {"clean": 0.6},
        "hair": {"neat": 0.8},
        "face": {"confident": 0.7}
    },
    "athlete": {
        "overall": "fit",
        "body": {"muscular": 0.8},
        "jaw": {"strong": 0.7},
        "neck": {"thick": 0.6}
    },
    "artist": {
        "overall": "creative",
        "eyes": {"expressive": 0.7},
        "face": {"unique": 0.5},
        "hair": {"unconventional": 0.6}
    }
}
# Gender keywords for detection
GENDER_KEYWORDS = {
    "male": ["man", "male", "boy", "gentleman", "guy", "dude", "he", "him", "his"],
    "female": ["woman", "female", "girl", "lady", "gal", "she", "her", "hers"]
}
# =============================================================================
# --- ENHANCED CONFIGURATION ---
# =============================================================================
DEFAULT_ETHNICITY = "Caucasian"
DEFAULT_GENDER = "male"
CONCEPT_MAP = {
    "caucasian": "Caucasian", "white": "Caucasian", "european": "Caucasian",
    "asian": "Asian", "oriental": "Asian", "east-asian": "Asian",
    "african": "African", "black": "African", "afro": "African",
    "elf": "Elf", "elven": "Elf", "elvish": "Elf",
    "dwarf": "Dwarf", "dwarven": "Dwarf"
}
# Enhanced feature mapping with more descriptive options
FEATURE_MAP = {
    "chin": {
        "long": "L2_{ethnicity}_Chin_SizeZ_max",
        "short": "L2_{ethnicity}_Chin_SizeZ_min",
        "wide": "L2_{ethnicity}_Chin_SizeX_max",
        "narrow": "L2_{ethnicity}_Chin_SizeX_min",
        "prominent": "L2_{ethnicity}_Chin_Prominence_max",
        "defined": "L2_{ethnicity}_Chin_Prominence_max",
        "strong": "L2_{ethnicity}_Chin_Prominence_max",
        "cleft": "L2_{ethnicity}_Chin_Cleft_max"
    },
    "jaw": {
        "strong": "L2_{ethnicity}_Jaw_Angle_min",
        "defined": "L2_{ethnicity}_Jaw_Angle_min",
        "angular": "L2_{ethnicity}_Jaw_Angle_min",
        "wide": "L2_{ethnicity}_Jaw_ScaleX_max",
        "narrow": "L2_{ethnicity}_Jaw_ScaleX_min",
        "soft": "L2_{ethnicity}_Jaw_Angle_max"
    },
    "eyes": {
        "big": "L2__Eyes_Size_max",
        "large": "L2__Eyes_Size_max",
        "small": "L2__Eyes_Size_min",
        "narrow": "L2__Eyes_Size_min",
        "wide-set": "L2_{ethnicity}_Eyes_PosX_max",
        "close-set": "L2_{ethnicity}_Eyes_PosX_min",
        "focused": "L2_{ethnicity}_Eyes_PosX_min",
        "sharp": "L2__Eyes_Size_min",
        "intense": "L2__Eyes_Size_min"
    },
    "nose": {
        "long": "L2_{ethnicity}_Nose_SizeY_max",
        "short": "L2_{ethnicity}_Nose_SizeY_min",
        "wide": "L2_{ethnicity}_Nose_BaseSizeX_max",
        "thin": "L2_{ethnicity}_Nose_BridgeSizeX_min",
        "refined": "L2_{ethnicity}_Nose_BridgeSizeX_min",
        "sharp": "L2_{ethnicity}_Nose_TipSize_min",
        "pointy": "L2_{ethnicity}_Nose_TipSize_min"
    },
    "lips": {
        "full": ["L2_{ethnicity}_Mouth_UpperlipVolume_max", "L2_{ethnicity}_Mouth_LowerlipVolume_max"],
        "thin": ["L2_{ethnicity}_Mouth_UpperlipVolume_min", "L2_{ethnicity}_Mouth_LowerlipVolume_min"]
    },
    "forehead": {
        "high": "L2_{ethnicity}_Forehead_SizeY_max",
        "low": "L2_{ethnicity}_Forehead_SizeY_min",
        "wide": "L2_{ethnicity}_Forehead_SizeX_max"
    },
    "ears": {
        "big": ["L2_{ethnicity}_Ears_SizeX_max", "L2_{ethnicity}_Ears_SizeY_max"],
        "small": ["L2_{ethnicity}_Ears_SizeX_min", "L2_{ethnicity}_Ears_SizeY_min"],
        "pointed": "L2__Fantasy_EarsPointed_max"
    }
}
INTENSITY_MAP = {
    "slightly": 0.3, "somewhat": 0.5, "moderately": 0.6,
    "very": 0.8, "extremely": 0.9, "incredibly": 1.0
}
DEFAULT_VALUE = 0.7
# =============================================================================
# --- NLP PROCESSING FUNCTIONS ---
# =============================================================================
def extract_keywords(prompt: str) -> List[str]:
    """Extract meaningful keywords from the prompt."""
    # Remove common stop words but keep descriptive ones
    stop_words = {'a', 'an', 'the', 'of', 'with', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'from'}
    words = re.findall(r'\b\w+(?:-\w+)?\b', prompt.lower())
    return [word for word in words if word not in stop_words]
def apply_synonyms(words: List[str]) -> List[str]:
    """Replace words with their canonical forms using synonym mapping."""
    return [SYNONYM_MAP.get(word, word) for word in words]
def detect_personality_traits(words: List[str]) -> Dict[str, float]:
    """Detect personality traits and map them to facial features."""
    detected_traits = {}
   
    for word in words:
        if word in PERSONALITY_TO_FEATURES:
            detected_traits[word] = 1.0
        # Also check for contextual professions/roles
        elif word in CONTEXTUAL_FEATURES:
            detected_traits[word] = 1.0
           
    return detected_traits
def map_traits_to_features(traits: Dict[str, float], detected_ethnicity: str) -> Dict[str, float]:
    """Convert personality traits to specific shape key modifications."""
    changes = {}
   
    for trait, intensity in traits.items():
        if trait in PERSONALITY_TO_FEATURES:
            trait_features = PERSONALITY_TO_FEATURES[trait]
        elif trait in CONTEXTUAL_FEATURES:
            trait_features = CONTEXTUAL_FEATURES[trait]
        else:
            continue
           
        for feature_part, modifiers in trait_features.items():
            if feature_part == "overall":
                continue # Skip overall descriptors for now
               
            if feature_part in FEATURE_MAP:
                for modifier, mod_intensity in modifiers.items():
                    if modifier in FEATURE_MAP[feature_part]:
                        shape_key_template = FEATURE_MAP[feature_part][modifier]
                        if isinstance(shape_key_template, list):
                            for template in shape_key_template:
                                final_key = template.format(ethnicity=detected_ethnicity)
                                changes[final_key] = mod_intensity * intensity
                        else:
                            final_key = shape_key_template.format(ethnicity=detected_ethnicity)
                            changes[final_key] = mod_intensity * intensity
   
    return changes
def smart_prompt_analysis(prompt: str) -> Dict:
    """Main NLP analysis function."""
    keywords = extract_keywords(prompt)
    keywords = apply_synonyms(keywords)
   
    # Detect basic demographics
    detected_ethnicity = DEFAULT_ETHNICITY
    for keyword in keywords:
        if keyword in CONCEPT_MAP:
            detected_ethnicity = CONCEPT_MAP[keyword]
            break
   
    # Detect gender
    detected_gender = DEFAULT_GENDER
    for word in keywords:
        if word in GENDER_KEYWORDS["female"]:
            detected_gender = "female"
            break
        elif word in GENDER_KEYWORDS["male"]:
            detected_gender = "male"
            break
   
    # Detect personality traits
    personality_traits = detect_personality_traits(keywords)
   
    # Detect age-related descriptors
    age_features = {}
    for keyword in keywords:
        if keyword in AGE_MAPPINGS:
            age_features.update(AGE_MAPPINGS[keyword])
   
    analysis_result = {
        "ethnicity": detected_ethnicity,
        "gender": detected_gender,
        "personality_traits": personality_traits,
        "age_features": age_features,
        "all_keywords": keywords
    }
   
    return analysis_result
# =============================================================================
# --- BLENDER HELPER FUNCTIONS ---
# =============================================================================
def get_object(name="mb_male"):
    """Safely gets the character object from the scene."""
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            return obj
    return None
def reset_character_shape_keys(obj):
    """Resets all shape key values to 0.0 for a clean start."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return
    print("--- Resetting all shape keys ---")
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
    print("Reset complete.")
def apply_morph(obj, shape_key_name, value):
    """Applies a single morph value, checking if the key exists."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return
    if shape_key_name in obj.data.shape_keys.key_blocks:
        print(f"Applying morph: '{shape_key_name}' with value {value:.2f}")
        obj.data.shape_keys.key_blocks[shape_key_name].value = min(1.0, value)
    else:
        print(f"Warning: Shape key '{shape_key_name}' not found.")
# =============================================================================
# --- MAIN ENHANCED PROCESSING FUNCTION ---
# =============================================================================
def process_and_apply_smart_prompt(prompt: str, character_obj):
    """Enhanced prompt processing with NLP capabilities."""
    print(f"Processing prompt: '{prompt}'")
   
    reset_character_shape_keys(character_obj)
   
    # Step 1: Smart analysis
    analysis = smart_prompt_analysis(prompt)
    print(f"Analysis result: {analysis}")
   
    changes_to_apply = {}
   
    # Step 2: Apply ethnicity
    ethnicity = analysis["ethnicity"]
    if ethnicity != DEFAULT_ETHNICITY:
        ethnicity_key = f"L1_{ethnicity}"
        changes_to_apply[ethnicity_key] = 1.0
   
    # Step 3: Apply personality-based features
    personality_changes = map_traits_to_features(analysis["personality_traits"], ethnicity)
    changes_to_apply.update(personality_changes)
   
    # Step 4: Process remaining keywords using original logic
    words = analysis["all_keywords"]
    for i, word in enumerate(words):
        if word in FEATURE_MAP:
            if i > 0:
                modifier = words[i-1]
                if modifier in FEATURE_MAP[word]:
                    value = DEFAULT_VALUE
                   
                    # Check for intensity
                    if i > 1 and words[i-2] in INTENSITY_MAP:
                        value = INTENSITY_MAP[words[i-2]]
                   
                    shape_key_templates = FEATURE_MAP[word][modifier]
                    if not isinstance(shape_key_templates, list):
                        shape_key_templates = [shape_key_templates]
                   
                    for template in shape_key_templates:
                        final_key = template.format(ethnicity=ethnicity)
                        changes_to_apply[final_key] = value
   
    # Step 5: Apply all changes
    print("\n--- Applying detected changes ---")
    if not changes_to_apply:
        print("No features detected. Applying default character.")
        return
   
    for shape_key, value in changes_to_apply.items():
        apply_morph(character_obj, shape_key, value)
   
    bpy.context.view_layer.update()
    print("--- Smart character generation complete! ---")
# =============================================================================
# BLENDER BRIDGE CONFIGURATION
# =============================================================================
# Communication directory (CHANGE THIS PATH TO MATCH YOUR FRONTEND!)
COMMUNICATION_DIR = r"C:\temp\blender_bridge" # Windows
# COMMUNICATION_DIR = "/tmp/blender_bridge" # macOS/Linux
REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")
# Global variable to control the monitoring loop
is_monitoring = False
def start_bridge_monitoring():
    """Start monitoring for character generation requests."""
    global is_monitoring
   
    if is_monitoring:
        print("Bridge monitoring is already active.")
        return
   
    # Ensure communication directory exists
    os.makedirs(COMMUNICATION_DIR, exist_ok=True)
   
    print(f"Starting Blender Bridge monitoring...")
    print(f"Watching directory: {COMMUNICATION_DIR}")
    print("Waiting for character generation requests...")
   
    is_monitoring = True
   
    # Register timer to check for requests every 0.5 seconds
    bpy.app.timers.register(check_for_requests, first_interval=0.5)
def stop_bridge_monitoring():
    """Stop monitoring for requests."""
    global is_monitoring
    is_monitoring = False
   
    # Unregister the timer
    if bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.unregister(check_for_requests)
   
    print("Bridge monitoring stopped.")
def check_for_requests():
    """Timer function that checks for new character requests."""
    global is_monitoring
   
    if not is_monitoring:
        return None # Stop the timer
   
    try:
        if os.path.exists(REQUEST_FILE):
            # Read the request
            with open(REQUEST_FILE, 'r') as f:
                request_data = json.load(f)
           
            print(f"Received request: {request_data['prompt']}")
           
            # Perform analysis to detect gender
            analysis = smart_prompt_analysis(request_data['prompt'])
            gender = analysis.get("gender", DEFAULT_GENDER)
           
            # Select the appropriate character based on gender
            if gender == "female":
                char_name = "mb_female"
            else:
                char_name = "mb_male"
           
            character = get_object(char_name)
           
            if character:
                # Use the enhanced NLP function
                process_and_apply_smart_prompt(request_data['prompt'], character)
               
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": request_data['prompt'],
                    "status": "completed",
                    "message": "Character generated successfully in Blender!"
                }
            else:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": request_data['prompt'],
                    "status": "error",
                    "message": f"Could not find character object for {gender} in Blender scene."
                }
           
            # Send response
            with open(RESPONSE_FILE, 'w') as f:
                json.dump(response_data, f)
           
            # Remove request file
            os.remove(REQUEST_FILE)
           
    except Exception as e:
        print(f"Error processing request: {e}")
        # Send error response
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        with open(RESPONSE_FILE, 'w') as f:
            json.dump(error_response, f)
       
        # Try to remove request file
        if os.path.exists(REQUEST_FILE):
            os.remove(REQUEST_FILE)
   
    return 0.5 # Continue checking every 0.5 seconds
# =============================================================================
# BLENDER UI PANEL (Optional - adds buttons to Blender UI)
# =============================================================================
class MESH_PT_character_bridge(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Character Generator Bridge"
    bl_idname = "MESH_PT_character_bridge"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("mesh.start_bridge")
        row = layout.row()
        row.operator("mesh.stop_bridge")
        row = layout.row()
        row.operator("mesh.test_generation")
class MESH_OT_start_bridge(bpy.types.Operator):
    """Start the VS Code bridge"""
    bl_idname = "mesh.start_bridge"
    bl_label = "Start Bridge"
    def execute(self, context):
        start_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring started")
        return {'FINISHED'}
class MESH_OT_stop_bridge(bpy.types.Operator):
    """Stop the VS Code bridge"""
    bl_idname = "mesh.stop_bridge"
    bl_label = "Stop Bridge"
    def execute(self, context):
        stop_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring stopped")
        return {'FINISHED'}
class MESH_OT_test_generation(bpy.types.Operator):
    """Test character generation"""
    bl_idname = "mesh.test_generation"
    bl_label = "Test Generation"
    def execute(self, context):
        test_prompt = "Generate an intelligent looking man with sharp features"  # Change to test female: "Generate an intelligent looking woman"
        analysis = smart_prompt_analysis(test_prompt)
        gender = analysis.get("gender", DEFAULT_GENDER)
        if gender == "female":
            char_name = "mb_female"
        else:
            char_name = "mb_male"
        character = get_object(char_name)
        if character:
            process_and_apply_smart_prompt(test_prompt, character)
            self.report({'INFO'}, f"Test generation completed: {test_prompt} (Gender: {gender})")
        else:
            self.report({'ERROR'}, f"Character object for {gender} not found")
        return {'FINISHED'}
def register():
    bpy.utils.register_class(MESH_PT_character_bridge)
    bpy.utils.register_class(MESH_OT_start_bridge)
    bpy.utils.register_class(MESH_OT_stop_bridge)
    bpy.utils.register_class(MESH_OT_test_generation)
def unregister():
    bpy.utils.unregister_class(MESH_PT_character_bridge)
    bpy.utils.unregister_class(MESH_OT_start_bridge)
    bpy.utils.unregister_class(MESH_OT_stop_bridge)
    bpy.utils.unregister_class(MESH_OT_test_generation)
# Register classes
register()
# =============================================================================
# AUTO-START (Optional - automatically start bridge when script runs)
# =============================================================================
# Uncomment the line below to automatically start monitoring when script runs
# start_bridge_monitoring()
print("=== CHARACTER GENERATOR BRIDGE LOADED ===")
print("Instructions:")
print("1. Run the frontend.py script in VS Code")
print("2. Click 'Start Bridge' in Blender Object Properties panel")
print("3. Open http://127.0.0.1:5000 in your browser")
print("4. Enter character descriptions and generate!")
# =============================================================================
# MAIN EXECUTION FOR TESTING
# =============================================================================
if __name__ == "__main__":
    # Test with various intelligent prompts
    test_prompts = [
        "generate an image of an intelligent looking man",
        "create a wise old wizard",
        "make a strong athletic woman",
        "generate a gentle kind teacher",
        "create a fierce warrior"
    ]
   
    # Use the first prompt or modify as needed
    user_prompt = test_prompts[0]
   
    analysis = smart_prompt_analysis(user_prompt)
    gender = analysis.get("gender", DEFAULT_GENDER)
    if gender == "female":
        char_name = "mb_female"
    else:
        char_name = "mb_male"
    character = get_object(char_name)
    if character:
        process_and_apply_smart_prompt(user_prompt, character)
    else:
        print(f"Error: Could not find character object for {gender}.")