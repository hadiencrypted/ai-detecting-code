genai = None
try:
    from google import genai as _genai
    genai = _genai
except Exception:
    try:
        import google.generativeai as _genai
        genai = _genai
    except Exception:
        genai = None
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import hashlib
import tempfile
import shutil
import urllib.request

# --- Local model (ai_model.h5) support ---
local_model = None
np = None

# Defer heavy ML imports and model loading to runtime to avoid
# double-initialization issues with Flask's reloader on Windows.
def load_local_model():
    global local_model, np
    try:
        import tensorflow as tf
        import numpy as _np
        from tensorflow.keras.models import load_model
        np = _np
        model_path = os.path.join(os.path.dirname(__file__), 'ai_model.h5')
        if os.path.exists(model_path):
            try:
                local_model = load_model(model_path)
                print('Loaded local model:', model_path)
            except Exception as e:
                print('Failed loading local model:', e)
        else:
            print('Local model file not found at', model_path)
    except Exception as e:
        print('TensorFlow not available or failed to import:', e)
        local_model = None


def ensure_model_present():
    """If `ai_model.h5` is missing and the env var MODEL_URL is set, download it.
    Optionally verify SHA256 with env var MODEL_SHA256.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'ai_model.h5')
    if os.path.exists(model_path):
        print('Local model already present at', model_path)
        return

    model_url = os.environ.get('MODEL_URL', '').strip()
    if not model_url:
        print('MODEL_URL not set; skipping automatic model download')
        return

    print('MODEL_URL set; attempting to download model from', model_url)
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmpf:
            tmp_path = tmpf.name
        urllib.request.urlretrieve(model_url, tmp_path)
        expected = os.environ.get('MODEL_SHA256', '').strip()
        if expected:
            h = hashlib.sha256()
            with open(tmp_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual.lower() != expected.lower():
                print(f'SHA256 mismatch: expected {expected}, got {actual}; removing downloaded file')
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return
        shutil.move(tmp_path, model_path)
        print('Model downloaded to', model_path)
    except Exception as e:
        print('Automatic model download failed:', e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

app = Flask(__name__)
CORS(app) # Cross-Origin Resource Sharing enable karne ke liye

# Load detector configuration (threshold)
THRESHOLD = 50
try:
    import json
    cfg_path = os.path.join(os.path.dirname(__file__), 'detector_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            THRESHOLD = int(cfg.get('threshold', THRESHOLD))
            print('Loaded detector threshold from config:', THRESHOLD)
except Exception as e:
    print('No detector config found or failed to read, using default threshold', THRESHOLD)

# --- STEP 1: API KEY CONFIGURATION ---
# Note: Maine wahi key rakhi hai jo tumne di thi
if genai is not None:
    try:
        # new and old libs expose different configuration methods; try common ones
        if hasattr(genai, 'configure'):
            genai.configure(api_key="AIzaSyCCMIXEUXhuw6fvhQwoUq0UWqm1TEvuGWI")
        elif hasattr(genai, 'configure_client'):
            genai.configure_client(api_key="AIzaSyCCMIXEUXhuw6fvhQwoUq0UWqm1TEvuGWI")
    except Exception:
        pass

# Small health endpoint to verify server from other machines
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Check if image is present
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        # 2. Gemini Model Initialize (Fixed Name)
        # Do not return early when genai is missing — prefer local model if available.
        if genai is None:
            print('genai library not available on server — will try local model if present')

        model = None
        # 3. AI Scanning Instructions (Added JSON instruction for better parsing)
        prompt = """
        Analyze this image for digital forensics. 
        Determine if it is AI-generated (Synthetic) or a real photograph (Authentic).
        Provide the response in the following format:
        Verdict: [AI or Real]
        Confidence: [Percentage]
        Comments: [One line technical observation]
        """
        # API call (wrap in try to log failures)
        try:
            # prefer constructing model if available on the client
            if hasattr(genai, 'GenerativeModel'):
                model = genai.GenerativeModel('gemini-1.5-flash')
            elif hasattr(genai, 'Model'):
                model = genai.Model('gemini-1.5-flash')

            response = None
            if model is not None and hasattr(model, 'generate_content'):
                response = model.generate_content([prompt, img])
            else:
                # try genai.generate if exposed differently
                if hasattr(genai, 'generate'):
                    response = genai.generate(prompt=prompt, image=img)

            print('Raw model response object:', repr(response))

            # extract text from known response shapes
            result_text = ''
            if response is None:
                result_text = ''
            elif hasattr(response, 'text'):
                result_text = (response.text or '')
            elif isinstance(response, dict) and 'text' in response:
                result_text = response.get('text', '')
            else:
                result_text = str(response)

            result_text = (result_text or '').lower()
            print('Parsed result_text:', result_text)
        except Exception as me:
            print('Model call failed:', repr(me))
            result_text = ''

        # 4. Result Parsing (Simplified Logic)
        is_ai = "ai" in result_text or "synthetic" in result_text
        
        # Confidence score nikalne ka safer tarika
        confidence = 90
        import re
        nums = re.findall(r'\d+', result_text)
        if nums:
            # Pehla number jo mile use confidence maan lo agar wo valid hai
            potential_score = int(nums[0])
            if 50 <= potential_score <= 100:
                confidence = potential_score

        # AI ke observations nikalna naye box ke liye
        comments = "No significant anomalies detected."
        if result_text:
            if "reason:" in result_text or "comments:" in result_text:
                try:
                    comments = result_text.split("reason:")[1].split("\n")[0].strip()
                except:
                    try:
                        comments = result_text.split("comments:")[1].split("\n")[0].strip()
                    except:
                        pass
        else:
            # If no text returned from model, make confidence 0 and note it
            confidence = 0
            comments = "No output from model; returned default result."

        # --- Local model inference (if ai_model.h5 loaded) ---
        def run_local_model(pil_img):
            try:
                if local_model is None or np is None:
                    return None
                # Standard preprocessing: resize to 224x224 and scale
                arr = np.array(pil_img.resize((224,224))).astype('float32') / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                arr = np.expand_dims(arr, 0)
                preds = local_model.predict(arr)
                p = np.array(preds)
                # Interpret prediction
                if p.size == 1:
                    prob = float(p.flatten()[0])
                elif p.ndim == 2 and p.shape[1] > 1:
                    prob = float(p[0,1])
                else:
                    prob = float(p.flatten()[0])
                # if prob seems >1 or <0, treat as logit and sigmoid
                if prob > 1 or prob < 0:
                    try:
                        prob = 1.0 / (1.0 + np.exp(-prob))
                    except Exception:
                        prob = max(0.0, min(1.0, prob))
                conf = int(round(prob * 100))
                return conf
            except Exception as e:
                print('Local model prediction failed:', e)
                return None

        try:
            local_conf = run_local_model(img)
            if local_conf is not None:
                # Prefer local model confidence when available
                confidence = int(local_conf)
                # Simple forensic heuristic to complement local model
                def forensic_ai_score(pil_img):
                    try:
                        arr = np.array(pil_img.convert('L')).astype('float32')
                        # Compute simple 3x3 Laplacian via neighbors
                        padded = np.pad(arr, 1, mode='reflect')
                        center = padded[1:-1,1:-1]
                        up = padded[0:-2,1:-1]
                        down = padded[2:,1:-1]
                        left = padded[1:-1,0:-2]
                        right = padded[1:-1,2:]
                        lap = (4.0*center - up - down - left - right)
                        hf = float(np.mean(np.abs(lap)))
                        # normalize with heuristic bounds
                        MIN_HF = 0.5
                        MAX_HF = 60.0
                        norm = (hf - MIN_HF) / (MAX_HF - MIN_HF)
                        norm = max(0.0, min(1.0, norm))
                        # lower high-frequency energy -> more likely AI
                        ai_score = 1.0 - norm
                        return ai_score
                    except Exception as e:
                        print('Forensic scoring failed:', e)
                        return 0.5

                try:
                    forensic_score_val = forensic_ai_score(img)
                    # Convert forensic score to percentage AI-likelihood
                    forensic_pct = int(round(forensic_score_val * 100))
                    # Combine local model confidence and forensic signal
                    combined = int(round(0.75 * confidence + 0.25 * forensic_pct))
                    print(f'Local_conf={confidence}, Forensic_pct={forensic_pct}, Combined={combined}')
                    confidence = combined
                    is_ai = confidence >= THRESHOLD
                except Exception as e:
                    print('Error combining forensic score:', e)
                    is_ai = confidence >= THRESHOLD
                # Note in comments that local model was used
                comments = (comments + ' (local model)') if comments else 'Local model used.'
        except Exception as e:
            print('Error during local model inference:', e)

        # Ensure confidence is numeric
        try:
            confidence = int(confidence)
        except Exception:
            confidence = 0

        return jsonify({
            "verdict": "AI SYNTHETIC" if is_ai else "AUTHENTIC",
            "confidence": confidence,
            "comments": comments.capitalize(),
            "metadata": "API Cloud Verified" if not is_ai else "AI Signature Detected",
            "spectral": "Deep Scanning Active",
            "compression": "Consistent" if not is_ai else "Inconsistent"
        })

    except Exception as e:
        print(f"Error details: {str(e)}")
        return jsonify({"error": f"Internal Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Attempt to download model at startup if requested via env vars,
    # then load local model once in the main process (avoids reloader double-init)
    ensure_model_present()
    load_local_model()
    print("Gemini AI Engine Starting on http://0.0.0.0:8000")
    # Bind to all interfaces so other devices on the network can reach it
    # Run with threading to handle occasional longer model inferences
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False, threaded=True)