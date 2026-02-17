"""
Hair generation module using GBH Tool
Maps text prompts to GBH Tool parameters for procedural hair generation
"""

import re
from typing import Dict, List, Tuple, Any, Optional

class HairPropertyAnalyzer:
    """Analyzes text prompts for hair-related features and maps to GBH parameters"""
    
    def __init__(self):
        # Hair style keywords mapping to GBH presets/types
        self.hair_style_keywords = {
            'straight': ['straight', 'sleek', 'flat', 'pin straight', 'stick straight'],
            'wavy': ['wavy', 'beachy waves', 'loose waves', 'soft waves', 'undulating'],
            'curly': ['curly', 'curls', 'ringlets', 'coily', 'corkscrew', 'tight curls'],
            'kinky': ['kinky', 'coily', 'afro', 'tight coils', 'natural hair'],
            'braided': ['braided', 'braids', 'cornrows', 'dutch braid', 'french braid'],
            'twisted': ['twisted', 'dreads', 'locs', 'twists', 'twist out'],
            'bob': ['bob cut', 'bob hairstyle', 'chin-length', 'shoulder-length bob'],
            'pixie': ['pixie cut', 'short crop', 'short hair', 'cropped'],
            'bun': ['bun', 'top knot', 'man bun', 'hair bun'],
            'ponytail': ['ponytail', 'high ponytail', 'low ponytail', 'side ponytail'],
            'mohawk': ['mohawk', 'faux hawk', 'fohawk'],
            'bald': ['bald', 'shaved head', 'no hair', 'bald head'],
            'long': ['long hair', 'flowing hair', 'waist-length', 'very long'],
            'short': ['short hair', 'cropped', 'close cut', 'buzz cut', 'military cut'],
            'beard': ['beard', 'facial hair', 'goatee', 'mustache', 'stubble'],
            'afro': ['afro', 'fro', 'natural afro', 'big hair'],
            'cornrows': ['cornrows', 'rows', 'canerows'],
            'dreadlocks': ['dreadlocks', 'dreads', 'locs', 'sisterlocks'],
            'mullet': ['mullet', 'business in front party in back'],
            'undercut': ['undercut', 'shaved sides', 'fade'],
            'quiff': ['quiff', 'pompadour'],
            'spiky': ['spiky', 'spikes', 'spiked hair'],
        }
        
        # Hair length mapping (0-1 scale for GBH parameters)
        self.length_keywords = {
            'very_short': ['buzz cut', 'shaved', 'very short', 'military cut'],
            'short': ['short hair', 'cropped', 'ear-length'],
            'medium': ['medium length', 'shoulder-length', 'collarbone-length'],
            'long': ['long hair', 'chest-length', 'waist-length'],
            'very_long': ['very long', 'hip-length', 'floor-length', 'extremely long'],
        }
        
        # Hair volume/thickness mapping
        self.volume_keywords = {
            'thin': ['thin hair', 'fine hair', 'sparse', 'thin', 'low density'],
            'normal': ['normal hair', 'average thickness', 'medium density'],
            'thick': ['thick hair', 'full hair', 'voluminous', 'dense', 'lots of hair'],
            'very_thick': ['very thick', 'extremely thick', 'super dense', 'very full'],
        }
        
        # Hair color mapping
        self.color_keywords = {
            'black': ['black hair', 'jet black', 'dark black', 'raven hair'],
            'brown': ['brown hair', 'brunette', 'chestnut', 'chocolate brown'],
            'blonde': ['blonde', 'blond', 'golden hair', 'platinum', 'ash blonde'],
            'red': ['red hair', 'ginger', 'auburn', 'copper', 'strawberry blonde'],
            'gray': ['gray hair', 'grey hair', 'silver', 'white hair', 'salt and pepper'],
            'blue': ['blue hair', 'dyed blue', 'cyan hair'],
            'pink': ['pink hair', 'rose gold', 'magenta hair'],
            'purple': ['purple hair', 'lavender hair', 'violet hair'],
            'green': ['green hair', 'emerald hair', 'teal hair'],
            'rainbow': ['rainbow hair', 'multicolor', 'colorful hair'],
        }
        
        # GBH conversion types
        self.gbh_conversion_types = {
            'strands': 'CURVES',      # For realistic hair strands
            'mesh': 'MESH',            # For stylized mesh hair
            'cards': 'MESH',           # For game-ready hair cards
            'particles': 'PARTICLES',  # For particle system
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'slightly': 0.3, 'somewhat': 0.4, 'moderately': 0.6,
            'very': 0.8, 'extremely': 0.9, 'incredibly': 1.0,
            'quite': 0.7, 'rather': 0.6, 'fairly': 0.5
        }
        
        self.default_intensity = 0.65
    
    def analyze_hair_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Main method to analyze hair-related text and return GBH parameters
        """
        prompt_lower = prompt.lower()
        
        hair_analysis = {
            'has_hair': True,
            'style': None,
            'length': None,
            'volume': None,
            'color': None,
            'conversion_type': 'CURVES',  # Default to strands
            'special_features': [],
            'parameters': {},
            'confidence': 0.0
        }
        
        # Check if bald/no hair mentioned
        for kw in self.hair_style_keywords['bald']:
            if kw in prompt_lower:
                hair_analysis['has_hair'] = False
                hair_analysis['style'] = 'bald'
                return hair_analysis
        
        # Detect hair style
        detected_style = self._detect_feature(prompt_lower, self.hair_style_keywords)
        if detected_style:
            hair_analysis['style'] = detected_style
        
        # Detect hair length
        detected_length = self._detect_feature(prompt_lower, self.length_keywords)
        if detected_length:
            hair_analysis['length'] = detected_length
        
        # Detect volume/thickness
        detected_volume = self._detect_feature(prompt_lower, self.volume_keywords)
        if detected_volume:
            hair_analysis['volume'] = detected_volume
        
        # Detect color
        detected_color = self._detect_feature(prompt_lower, self.color_keywords)
        if detected_color:
            hair_analysis['color'] = detected_color
        
        # Determine conversion type based on style and keywords
        if any(x in prompt_lower for x in ['mesh hair', 'stylized', 'cartoon', 'anime']):
            hair_analysis['conversion_type'] = 'MESH'
        elif any(x in prompt_lower for x in ['hair cards', 'game', 'low poly']):
            hair_analysis['conversion_type'] = 'MESH'  # Cards are also mesh
        elif any(x in prompt_lower for x in ['particles', 'particle system']):
            hair_analysis['conversion_type'] = 'PARTICLES'
        
        # Map to GBH parameters
        hair_analysis['parameters'] = self._map_to_gbh_parameters(hair_analysis, prompt_lower)
        
        # Calculate confidence
        detected_count = sum(1 for v in [detected_style, detected_length, detected_volume, detected_color] if v)
        hair_analysis['confidence'] = detected_count / 4.0 if detected_count > 0 else 0.0
        
        return hair_analysis
    
    def _detect_feature(self, prompt_lower: str, keyword_dict: Dict) -> Optional[str]:
        """Helper to detect which feature is mentioned"""
        best_match = None
        best_score = 0
        
        for feature, keywords in keyword_dict.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    # Check for modifiers
                    for modifier, mod_score in self.intensity_modifiers.items():
                        if f"{modifier} {keyword}" in prompt_lower:
                            score += mod_score * 2
                            break
                    else:
                        score += self.default_intensity
                    
                    # Also check if keyword appears with word boundaries
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, prompt_lower):
                        score += 0.3
            
            if score > best_score:
                best_score = score
                best_match = feature
        
        return best_match if best_score > 0 else None
    
    def _map_to_gbh_parameters(self, analysis: Dict, prompt_lower: str) -> Dict[str, float]:
        """
        Map detected hair features to GBH Tool parameters
        """
        params = {}
        
        # Style-specific parameter mapping
        style_params = {
            'straight': {'curl': 0.0, 'noise': 0.2, 'density': 0.7},
            'wavy': {'curl': 0.4, 'noise': 0.3, 'density': 0.7},
            'curly': {'curl': 0.8, 'noise': 0.4, 'density': 0.8},
            'kinky': {'curl': 1.0, 'noise': 0.5, 'density': 0.9},
            'afro': {'curl': 1.0, 'noise': 0.6, 'density': 1.0, 'volume': 1.0},
            'braided': {'braid_intensity': 0.8, 'twist': 1.0, 'density': 0.8},
            'twisted': {'twist': 0.8, 'curl': 0.3, 'density': 0.7},
            'dreadlocks': {'twist': 1.0, 'curl': 0.2, 'density': 0.6, 'thickness': 0.8},
            'cornrows': {'braid_intensity': 1.0, 'pattern': 0.5, 'density': 0.7},
            'bob': {'length': 0.3, 'volume': 0.5, 'style_type': 0.5},
            'pixie': {'length': 0.1, 'volume': 0.3, 'style_type': 0.3},
            'bun': {'bun_size': 0.7, 'style_type': 2.0, 'length': 0.2},
            'ponytail': {'ponytail_height': 0.6, 'style_type': 1.0, 'length': 0.6},
            'mohawk': {'mohawk_width': 0.7, 'style_type': 3.0, 'length': 0.4},
            'mullet': {'length_front': 0.2, 'length_back': 0.8, 'style_type': 4.0},
            'undercut': {'sides': 0.0, 'top': 0.7, 'style_type': 5.0},
            'quiff': {'front_volume': 0.8, 'length': 0.5, 'style_type': 6.0},
            'spiky': {'spike_height': 0.7, 'spike_density': 0.6, 'style_type': 7.0},
        }
        
        # Length mapping (0-1)
        length_to_value = {
            'very_short': 0.1,
            'short': 0.3,
            'medium': 0.5,
            'long': 0.7,
            'very_long': 0.9,
        }
        
        # Volume mapping (0-1)
        volume_to_value = {
            'thin': 0.3,
            'normal': 0.5,
            'thick': 0.7,
            'very_thick': 0.9,
        }
        
        # Apply style parameters
        style = analysis.get('style')
        if style and style in style_params:
            params.update(style_params[style])
        
        # Apply length
        if analysis['length'] and analysis['length'] in length_to_value:
            params['length'] = length_to_value[analysis['length']]
        
        # Apply volume/density
        if analysis['volume'] and analysis['volume'] in volume_to_value:
            params['density'] = volume_to_value[analysis['volume']]
            params['volume'] = volume_to_value[analysis['volume']]
        
        # Check for parting
        if 'middle part' in prompt_lower or 'center part' in prompt_lower:
            params['parting'] = 0.5
        elif 'side part' in prompt_lower or 'left part' in prompt_lower:
            params['parting'] = 0.3
        elif 'right part' in prompt_lower:
            params['parting'] = 0.7
        
        # Check for bangs/fringe
        if any(x in prompt_lower for x in ['bangs', 'fringe', 'front hair']):
            params['bangs'] = 0.7
            params['bangs_length'] = params.get('length', 0.5) * 0.7
        
        # Check for facial hair
        if any(x in prompt_lower for x in ['beard', 'mustache', 'goatee', 'stubble']):
            params['facial_hair'] = 1.0
            if 'stubble' in prompt_lower:
                params['beard_length'] = 0.1
            elif 'goatee' in prompt_lower:
                params['beard_style'] = 0.5
            elif 'mustache' in prompt_lower:
                params['mustache'] = 1.0
        
        return params
    
    def get_gbh_operator_sequence(self, target_object_name: str, hair_analysis: Dict) -> List[Dict]:
        """
        Generate sequence of GBH operators to execute in Blender
        Using the actual GBH Tool operators from the source code
        """
        operators = []
        
        # Only generate hair if hair is requested
        if not hair_analysis.get('has_hair', True):
            return []
        
        # Select the target object
        operators.append({
            'type': 'SELECT',
            'object': target_object_name
        })
        
        # Determine conversion type
        conversion_type = hair_analysis.get('conversion_type', 'CURVES')
        
        # Main hair generation/conversion
        # Note: GBH Tool seems to work by converting between types
        # We'll create a base curves object and then convert as needed
        
        if conversion_type == 'PARTICLES':
            operators.append({
                'type': 'GBH_OT_strands_to_particle',
                'properties': {}
            })
        else:
            # First convert to curves (if not already)
            operators.append({
                'type': 'GBH_OT_convert_hair',
                'properties': {
                    'convert_to': 'CURVES'
                }
            })
            
            # Then convert to desired type if needed
            if conversion_type == 'MESH':
                operators.append({
                    'type': 'GBH_OT_convert_hair',
                    'properties': {
                        'convert_to': 'MESH'
                    }
                })
        
        # Attach to surface if needed
        if hair_analysis.get('attach_to_surface', False):
            operators.append({
                'type': 'GBH_OT_attach_curves_to_surface',
                'properties': {}
            })
        
        return operators

# Global instance
hair_analyzer = HairPropertyAnalyzer()