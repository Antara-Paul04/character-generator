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
            
            # REDUCE PARAMETERS FOR LESS HAIR COVERAGE
            reduced_hair_params = hair_params.copy()
            if 'particle_count' in reduced_hair_params:
                reduced_hair_params['particle_count'] = int(reduced_hair_params['particle_count'] * 0.7)  # Reduce by 30%
            if 'particle_length' in reduced_hair_params:
                reduced_hair_params['particle_length'] = reduced_hair_params['particle_length'] * 0.8  # Reduce length
            
            with open(params_file, 'w') as f:
                json.dump(reduced_hair_params, f)
            
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
        
        # REDUCED PARTICLE COUNTS AND LENGTHS
        if hair_params.get('length') == 'long':
            hair_data['particle_count'] = 6000  # Reduced from 8000
            hair_data['particle_length'] = 0.7  # Reduced from 0.8
        elif hair_params.get('length') == 'short':
            hair_data['particle_count'] = 3000  # Reduced from 4000
            hair_data['particle_length'] = 0.2  # Reduced from 0.3
        else:  # medium
            hair_data['particle_count'] = 4500  # Reduced from 6000
            hair_data['particle_length'] = 0.4  # Reduced from 0.5
        
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
        
        # REDUCED VOLUME ADJUSTMENTS
        if hair_params.get('volume') == 'thick':
            hair_data['particle_count'] *= 1.2  # Reduced from 1.5
            hair_data['child_radius'] = 0.2     # Reduced from 0.4
        elif hair_params.get('volume') == 'thin':
            hair_data['particle_count'] *= 0.8  # Increased from 0.7
            hair_data['child_radius'] = 0.05    # Reduced from 0.1
        else:  # normal volume
            hair_data['child_radius'] = 0.1     # Added default
        
        # ADDITIONAL SETTINGS FOR BETTER HEAD CONTAINMENT
        hair_data['child_nbr'] = max(50, int(hair_data['particle_count'] * 0.05))  # Only 5% children
        hair_data['child_length'] = 0.8  # Reduced child length
        
        print(f"✅ Mock hair data generated: {hair_data['method']}")
        print(f"   Particles: {hair_data['particle_count']}, Length: {hair_data['particle_length']}")
        return hair_data