import os
import json
import subprocess
import sys
from typing import Dict, Any, Optional

class HairNetIntegrator:
    """Handles HairNet AI integration and hair generation"""
    
    def __init__(self, hairnet_path: str):
        self.hairnet_path = hairnet_path
        self.hairnet_script = os.path.join(hairnet_path, "generate_hair.py")
    
    def generate_hair_model(self, hair_params: Dict[str, Any], character_name: str) -> Dict[str, Any]:
        """
        Generate hair model using HairNet AI
        
        Args:
            hair_params: Dictionary of hair parameters
            character_name: Name for the character
            
        Returns:
            Dictionary with generation results
        """
        try:
            if not os.path.exists(self.hairnet_script):
                return {
                    'success': False,
                    'error': f'HairNet script not found: {self.hairnet_script}',
                    'mock': True
                }
            
            # Prepare output path
            output_dir = os.path.join(os.getcwd(), "generated_hair")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"{character_name}_hair.obj")
            
            # Save parameters to temporary file
            params_file = os.path.join(output_dir, f"{character_name}_hair_params.json")
            with open(params_file, 'w') as f:
                json.dump(hair_params, f)
            
            # Run HairNet generation
            cmd = [
                sys.executable, self.hairnet_script,
                "--params", params_file,
                "--output", output_path,
                "--character", character_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.hairnet_path)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return {
                    'success': True,
                    'output_path': output_path,
                    'params_path': params_file,
                    'message': 'Hair model generated successfully'
                }
            else:
                return {
                    'success': False,
                    'error': f'HairNet generation failed: {result.stderr}',
                    'mock': True
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Hair generation error: {str(e)}',
                'mock': True
            }

class MockHairNetIntegrator:
    """Mock hair generator for when HairNet is not available"""
    
    def generate_hair_model(self, hair_params: Dict[str, Any], character_name: str) -> Dict[str, Any]:
        """
        Generate mock hair data for Blender to use
        
        Args:
            hair_params: Dictionary of hair parameters
            character_name: Name for the character
            
        Returns:
            Dictionary with mock generation results
        """
        print(f"🎨 Generating mock hair for {character_name}")
        print(f"📋 Hair parameters: {hair_params}")
        
        # Create hair data structure for Blender
        hair_data = {
            'success': True,
            'mock': True,
            'hair_params': hair_params,
            'character_name': character_name,
            'method': 'particle_system',  # Tell Blender to use particle system
            'message': 'Using Blender particle system for hair'
        }
        
        # Add particle system parameters based on hair description
        if hair_params.get('length') == 'long':
            hair_data['particle_count'] = 8000
            hair_data['particle_length'] = 0.8
        elif hair_params.get('length') == 'short':
            hair_data['particle_count'] = 4000
            hair_data['particle_length'] = 0.3
        else:  # medium
            hair_data['particle_count'] = 6000
            hair_data['particle_length'] = 0.5
        
        # Style adjustments
        if hair_params.get('style') == 'curly':
            hair_data['curl_intensity'] = 0.7
            hair_data['randomness'] = 0.4
        elif hair_params.get('style') == 'wavy':
            hair_data['curl_intensity'] = 0.3
            hair_data['randomness'] = 0.2
        else:  # straight
            hair_data['curl_intensity'] = 0.0
            hair_data['randomness'] = 0.1
        
        # Volume adjustments
        if hair_params.get('volume') == 'thick':
            hair_data['particle_count'] *= 1.5
            hair_data['child_radius'] = 0.4
        elif hair_params.get('volume') == 'thin':
            hair_data['particle_count'] *= 0.7
            hair_data['child_radius'] = 0.1
        
        print(f"✅ Mock hair data generated: {hair_data['method']}")
        return hair_data