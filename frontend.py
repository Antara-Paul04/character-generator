from flask import Flask, render_template, request, jsonify
import json
import os
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
# =============================================================================

# Communication directory
COMMUNICATION_DIR = r"C:\temp\blender_bridge"  # Windows
# COMMUNICATION_DIR = "/tmp/blender_bridge"    # macOS/Linux

REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")
BLENDER_STATUS_FILE = os.path.join(COMMUNICATION_DIR, "blender_status.json")

# Blender configuration
BLENDER_EXECUTABLE = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"  # Update this path!
# BLENDER_EXECUTABLE = "/Applications/Blender.app/Contents/MacOS/Blender"  # macOS
# BLENDER_EXECUTABLE = "blender"  # If blender is in PATH

# Your model file - UPDATE THIS PATH!
MODEL_BLEND_FILE = os.path.join(os.getcwd(), "base.blend")
# MODEL_BLEND_FILE = r"C:\path\to\your\base.blend"  # Or use absolute path

# Bridge script path
BRIDGE_SCRIPT_PATH = os.path.join(os.getcwd(), "blender_bridge.py")

# Global state tracking
blender_started_once = False
last_successful_generation = None

# Ensure communication directory exists
os.makedirs(COMMUNICATION_DIR, exist_ok=True)

# =============================================================================
# BLENDER MANAGEMENT FUNCTIONS
# =============================================================================

def is_blender_responsive():
    """Check if Blender can respond to requests (more lenient check)."""
    global last_successful_generation
    
    # If we've never started Blender, definitely not responsive
    if not blender_started_once:
        return False
    
    # Check for recent successful generation (within last 60 seconds means it's working)
    if last_successful_generation:
        time_since_last = (datetime.now() - last_successful_generation).total_seconds()
        if time_since_last < 60:
            return True
    
    # Try to send a quick test request to see if Blender responds
    try:
        test_request = {
            "timestamp": datetime.now().isoformat(),
            "prompt": "__STATUS_CHECK__",
            "status": "pending"
        }
        
        with open(REQUEST_FILE, 'w') as f:
            json.dump(test_request, f)
        
        # Wait briefly for response
        start_time = time.time()
        while time.time() - start_time < 3:  # 3 second timeout for status check
            if os.path.exists(RESPONSE_FILE):
                os.remove(RESPONSE_FILE)  # Clean up
                return True
            time.sleep(0.1)
        
        # Clean up test request
        if os.path.exists(REQUEST_FILE):
            os.remove(REQUEST_FILE)
            
    except:
        pass
    
    # Fallback: check status file if it exists and is recent
    try:
        if os.path.exists(BLENDER_STATUS_FILE):
            with open(BLENDER_STATUS_FILE, 'r') as f:
                status = json.load(f)
            last_update = datetime.fromisoformat(status.get('timestamp', '2000-01-01T00:00:00'))
            time_diff = (datetime.now() - last_update).total_seconds()
            return time_diff < 30  # More lenient - 30 seconds
    except:
        pass
    
    return False

def start_blender_with_model():
    """Start Blender with the model file and bridge script."""
    global blender_started_once
    
    try:
        if not os.path.exists(MODEL_BLEND_FILE):
            return {"success": False, "error": f"Model file not found: {MODEL_BLEND_FILE}"}
        
        if not os.path.exists(BLENDER_EXECUTABLE):
            return {"success": False, "error": f"Blender executable not found: {BLENDER_EXECUTABLE}"}
        
        # Create a startup script that loads the bridge
        startup_script = f'''
import bpy
import sys
import os

# Add current directory to Python path
sys.path.append(r"{os.getcwd()}")

# Load and execute the bridge script
exec(open(r"{BRIDGE_SCRIPT_PATH}").read())

# Auto-start the bridge
start_bridge_monitoring()

print("=== BLENDER BRIDGE AUTO-STARTED ===")
'''
        
        # Save startup script
        startup_script_path = os.path.join(COMMUNICATION_DIR, "startup_script.py")
        with open(startup_script_path, 'w') as f:
            f.write(startup_script)
        
        # Launch Blender with the model file and startup script
        cmd = [
            BLENDER_EXECUTABLE,
            MODEL_BLEND_FILE,
            "--python", startup_script_path
        ]
        
        print(f"Starting Blender with command: {' '.join(cmd)}")
        
        # Start Blender in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        
        # Give Blender time to start
        time.sleep(5)  # Increased wait time
        
        blender_started_once = True
        
        return {
            "success": True, 
            "message": f"Blender started with {os.path.basename(MODEL_BLEND_FILE)}",
            "process_id": process.pid
        }
        
    except Exception as e:
        return {"success": False, "error": f"Failed to start Blender: {str(e)}"}

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-blender', methods=['POST'])
def start_blender():
    """Start Blender with the model file."""
    global blender_started_once
    
    if is_blender_responsive():
        return jsonify({"success": True, "message": "Blender is already running and responsive!"})
    
    result = start_blender_with_model()
    return jsonify(result) if result["success"] else (jsonify(result), 500)

@app.route('/generate', methods=['POST'])
def generate_character():
    global last_successful_generation
    
    try:
        prompt = request.json.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # More lenient check - if Blender was started once, try to generate anyway
        if not blender_started_once:
            return jsonify({
                "error": "Please start Blender first using the 'Start Blender' button."
            }), 400
        
        # Create request for Blender
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "status": "pending"
        }
        
        # Write request to file
        with open(REQUEST_FILE, 'w') as f:
            json.dump(request_data, f)
        
        # Wait for response (timeout after 30 seconds)
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(RESPONSE_FILE):
                with open(RESPONSE_FILE, 'r') as f:
                    response_data = json.load(f)
                
                # Clean up response file
                os.remove(RESPONSE_FILE)
                
                # Update last successful generation time
                last_successful_generation = datetime.now()
                
                return jsonify({
                    "success": True,
                    "message": "Character generated successfully!",
                    "details": response_data
                })
            
            time.sleep(0.5)
        
        return jsonify({
            "error": "Timeout waiting for Blender response. Blender might be busy or closed."
        }), 408
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """Return status - now more permissive."""
    global blender_started_once, last_successful_generation
    
    # If Blender was started once, assume it's still working unless proven otherwise
    blender_probably_running = blender_started_once
    
    # Additional info for debugging
    last_gen_time = last_successful_generation.isoformat() if last_successful_generation else None
    
    return jsonify({
        "blender_running": blender_probably_running,
        "blender_started_once": blender_started_once,
        "last_successful_generation": last_gen_time,
        "model_file": MODEL_BLEND_FILE,
        "model_exists": os.path.exists(MODEL_BLEND_FILE),
        "blender_executable": BLENDER_EXECUTABLE,
        "blender_exists": os.path.exists(BLENDER_EXECUTABLE),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/reset-status', methods=['POST'])
def reset_status():
    """Reset the Blender status if user is having issues."""
    global blender_started_once, last_successful_generation
    blender_started_once = False
    last_successful_generation = None
    return jsonify({"success": True, "message": "Status reset. You can now start Blender again."})

@app.route('/config')
def config():
    return jsonify({
        "model_file": MODEL_BLEND_FILE,
        "blender_executable": BLENDER_EXECUTABLE,
        "communication_dir": COMMUNICATION_DIR
    })

if __name__ == '__main__':
    print("🎭 Character Generator Frontend Starting...")
    print(f"📁 Communication directory: {COMMUNICATION_DIR}")
    print(f"🎨 Model file: {MODEL_BLEND_FILE}")
    print(f"🔧 Blender executable: {BLENDER_EXECUTABLE}")
    print()
    
    # Check configuration
    if not os.path.exists(MODEL_BLEND_FILE):
        print(f"⚠️  WARNING: Model file not found: {MODEL_BLEND_FILE}")
        print("   Update MODEL_BLEND_FILE path in the script")
    
    if not os.path.exists(BLENDER_EXECUTABLE):
        print(f"⚠️  WARNING: Blender executable not found: {BLENDER_EXECUTABLE}")
        print("   Update BLENDER_EXECUTABLE path in the script")
    
    print("🌐 Open http://127.0.0.1:5000 in your browser")
    print("🚀 Click 'Start Blender' once, then generate as many characters as you want!")
    print("💡 The Generate button should stay enabled after the first Blender start")
    
    app.run(debug=True, host='127.0.0.1', port=5000)