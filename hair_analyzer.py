import re
from typing import Dict, Tuple, Optional
import spacy

class HairDescriptionAnalyzer:
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model
        
        # Hair feature keywords
        self.hair_keywords = {
            # Length
            'very_long': ['very long hair', 'extremely long hair', 'waist length', 'hip length'],
            'long': ['long hair', 'lengthy hair', 'flowing hair'],
            'medium': ['medium hair', 'shoulder length', 'mid length'],
            'short': ['short hair', 'cropped hair', 'pixie cut'],
            'very_short': ['very short', 'buzz cut', 'crew cut', 'shaved'],
            
            # Style
            'straight': ['straight hair', 'sleek hair', 'flat hair'],
            'wavy': ['wavy hair', 'beach waves', 'loose waves'],
            'curly': ['curly hair', 'curls', 'ringlets', 'spiral curls'],
            'kinky': ['kinky hair', 'coily hair', 'afro', 'tight curls'],
            'braided': ['braided', 'braids', 'cornrows', 'box braids'],
            'dreadlocks': ['dreadlocks', 'dreads', 'locs'],
            
            # Volume/Density
            'thick': ['thick hair', 'full hair', 'voluminous hair', 'abundant hair'],
            'thin': ['thin hair', 'fine hair', 'sparse hair'],
            'normal_volume': ['normal hair', 'medium volume'],
            
            # Texture
            'smooth': ['smooth hair', 'silky hair', 'sleek'],
            'rough': ['rough hair', 'coarse hair', 'frizzy'],
            'soft': ['soft hair', 'fine texture'],
            
            # Color (for reference)
            'black': ['black hair', 'dark hair', 'ebony'],
            'brown': ['brown hair', 'brunette', 'chestnut'],
            'blonde': ['blonde hair', 'fair hair', 'golden'],
            'red': ['red hair', 'ginger', 'auburn'],
            'white': ['white hair', 'gray hair', 'silver hair'],
            
            # Special styles
            'ponytail': ['ponytail', 'tied back', 'pulled back'],
            'bun': ['bun', 'top knot', 'chignon'],
            'bangs': ['bangs', 'fringe', 'front bangs'],
            'mohawk': ['mohawk', 'fohawk'],
            'undercut': ['undercut', 'shaved sides'],
        }
        
        # Map features to HairNet-like parameters
        self.length_mapping = {
            'very_short': 0.2,
            'short': 0.4,
            'medium': 0.6,
            'long': 0.8,
            'very_long': 1.0
        }
        
        self.volume_mapping = {
            'thin': 0.3,
            'normal_volume': 0.6,
            'thick': 0.9
        }
        
        self.curl_mapping = {
            'straight': 0.0,
            'wavy': 0.3,
            'curly': 0.7,
            'kinky': 1.0
        }
    
    def extract_hair_description(self, prompt: str) -> Dict:
        """Extract hair-related features from prompt"""
        prompt_lower = prompt.lower()
        
        features = {
            'has_hair_description': False,
            'length': None,
            'style': None,
            'volume': None,
            'texture': None,
            'color': None,
            'special_style': None,
            'raw_description': ""
        }
        
        # Check if hair is mentioned at all
        hair_mentions = ['hair', 'hairstyle', 'haircut', 'locks', 'mane']
        if not any(mention in prompt_lower for mention in hair_mentions):
            return features
        
        features['has_hair_description'] = True
        
        # Extract each feature type
        for feature_type in ['length', 'style', 'volume', 'texture', 'color', 'special_style']:
            for feature_key, keywords in self.hair_keywords.items():
                if any(self._is_feature_category(feature_key, feature_type)):
                    for keyword in keywords:
                        if keyword in prompt_lower:
                            if features[feature_type] is None:
                                features[feature_type] = feature_key
                            break
        
        # Extract raw description (sentences containing "hair")
        if self.nlp:
            doc = self.nlp(prompt)
            hair_sentences = []
            for sent in doc.sents:
                if any(mention in sent.text.lower() for mention in hair_mentions):
                    hair_sentences.append(sent.text.strip())
            features['raw_description'] = ' '.join(hair_sentences)
        else:
            # Fallback: simple extraction
            sentences = prompt.split('.')
            for sent in sentences:
                if any(mention in sent.lower() for mention in hair_mentions):
                    features['raw_description'] += sent.strip() + '. '
        
        return features
    
    def _is_feature_category(self, feature_key: str, category: str) -> bool:
        """Check if feature belongs to category"""
        category_map = {
            'length': ['very_long', 'long', 'medium', 'short', 'very_short'],
            'style': ['straight', 'wavy', 'curly', 'kinky', 'braided', 'dreadlocks'],
            'volume': ['thick', 'thin', 'normal_volume'],
            'texture': ['smooth', 'rough', 'soft'],
            'color': ['black', 'brown', 'blonde', 'red', 'white'],
            'special_style': ['ponytail', 'bun', 'bangs', 'mohawk', 'undercut']
        }
        return feature_key in category_map.get(category, [])
    
    def convert_to_hairnet_params(self, features: Dict) -> Dict:
        """Convert analyzed features to HairNet-compatible parameters"""
        if not features['has_hair_description']:
            return None
        
        params = {
            'particle_length': 0.5,  # p_l
            'particle_width': 0.02,  # p_wh
            'particle_count': 5000,  # count
            'curl_intensity': 0.0,   # derived from style
            'randomness': 0.3,       # scale_rand
            'density': 0.6           # affects count
        }
        
        # Map length
        if features['length']:
            params['particle_length'] = self.length_mapping.get(features['length'], 0.5)
        
        # Map volume to density and count
        if features['volume']:
            density = self.volume_mapping.get(features['volume'], 0.6)
            params['density'] = density
            params['particle_count'] = int(3000 + density * 7000)  # 3000-10000 range
        
        # Map style to curl intensity
        style_curl_map = {
            'straight': 0.0,
            'wavy': 0.3,
            'curly': 0.7,
            'kinky': 1.0
        }
        if features['style']:
            params['curl_intensity'] = style_curl_map.get(features['style'], 0.0)
        
        # Adjust parameters for special styles
        if features['special_style']:
            if features['special_style'] == 'ponytail':
                params['particle_length'] *= 1.2
                params['particle_count'] = int(params['particle_count'] * 0.7)
            elif features['special_style'] == 'bun':
                params['particle_length'] *= 0.6
                params['particle_count'] = int(params['particle_count'] * 0.8)
            elif features['special_style'] in ['mohawk', 'undercut']:
                params['particle_count'] = int(params['particle_count'] * 0.4)
        
        return params
    
    def generate_hair_description_for_image(self, features: Dict) -> str:
        """Generate a text description for image generation"""
        if not features['has_hair_description']:
            return "generic human hair"
        
        parts = []
        
        if features['length']:
            parts.append(features['length'].replace('_', ' '))
        if features['style']:
            parts.append(features['style'])
        if features['volume']:
            parts.append(features['volume'])
        if features['color']:
            parts.append(features['color'])
        
        parts.append('hair')
        
        if features['special_style']:
            parts.append('in a ' + features['special_style'])
        
        return ' '.join(parts)