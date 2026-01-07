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
import traceback

# --- Local model (ai_model.h5) support ---
local_model = None
np = None

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
                print('✓ Loaded local model:', model_path)
            except Exception as e:
                print('✗ Failed loading local model:', e)
        else:
            print('✗ Local model file not found at', model_path)
    except Exception as e:
        print('✗ TensorFlow not available or failed to import:', e)
        local_model = None


def ensure_model_present():
    """If `ai_model.h5` is missing and the env var MODEL_URL is set, download it."""
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
                print(f'SHA256 mismatch: expected {expected}, got {actual}')
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
CORS(app)

# Load detector configuration
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
    print('No detector config found, using default threshold', THRESHOLD)

# --- API KEY CONFIGURATION ---
print('\n=== GEMINI API SETUP ===')
if genai is not None:
    print('✓ genai module imported')
    try:
        API_KEY = os.environ.get('GENAI_API_KEY', "AIzaSyCCMIXEUXhuw6fvhQwoUq0UWqm1TEvuGWI")
        if hasattr(genai, 'configure'):
            genai.configure(api_key=API_KEY)
            print('✓ Gemini API key configured via configure()')
        elif hasattr(genai, 'configure_client'):
            genai.configure_client(api_key=API_KEY)
            print('✓ Gemini API key configured via configure_client()')
    except Exception as e:
        print(f'✗ Failed to configure Gemini API key: {e}')
else:
    print('✗ genai module not imported - Gemini API will not be available')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'genai_available': genai is not None}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        print(f'\nAnalyzing image: {file.filename}')

        # DEBUG: Metadata Found? [Yes/No]
        def has_exif(pil_img):
            try:
                exif = pil_img._getexif()
                return bool(exif)
            except Exception:
                return False

        meta_found = has_exif(img)
        print(f'DEBUG: Metadata Found? [{"Yes" if meta_found else "No"}]')

        # FORCE Metadata Check: If metadata exists, immediately return Real
        if meta_found:
            return jsonify({
                'verdict': 'Real hai photo',
                'confidence': 100,
                'comments': 'Metadata found - trusted as real camera output',
                'source': 'Metadata (Highest Priority)'
            })

        # FIX Scaling/Normalization: use img_to_array and divide by 255.0
        normalized = None
        try:
            # try Keras util first
            try:
                from tensorflow.keras.preprocessing.image import img_to_array
                arr = img_to_array(img)
            except Exception:
                from keras.utils import img_to_array
                arr = img_to_array(img)
            import numpy as _np
            normalized = arr.astype('float32') / 255.0
        except Exception:
            # fallback to numpy
            import numpy as _np
            arr = _np.array(img).astype('float32')
            normalized = arr / 255.0

        vmin = float(normalized.min())
        vmax = float(normalized.max())
        print(f'DEBUG: Normalized Value: [{vmin:.6f}/{vmax:.6f}]')

        # Stage 2: Heuristic watermark check (used as a decisive negative signal)
        def watermark_check(pil_img):
            try:
                w,h = pil_img.size
                import numpy as _np
                corners = [ (0,0,w//6,h//6), (w - w//6,0,w,h//6), (0,h - h//6,w//6,h), (w - w//6,h - h//6,w,h) ]
                for (x1,y1,x2,y2) in corners:
                    crop = pil_img.crop((x1,y1,x2,y2)).convert('L')
                    arrc = _np.array(crop).astype('float32')
                    gx = _np.abs(_np.diff(arrc, axis=1)).mean()
                    gy = _np.abs(_np.diff(arrc, axis=0)).mean()
                    edge_density = (gx+gy)/2.0
                    dark_ratio = (arrc < arrc.mean()*0.6).sum() / arrc.size
                    if edge_density > 8.0 or dark_ratio > 0.25:
                        return True, {'edge_density': float(edge_density), 'dark_ratio': float(dark_ratio)}
                return False, None
            except Exception as e:
                print('Watermark check failed:', e)
                return False, None

        # Step 2: Gemini API
        gemini_text = None
        gemini_result = None
        try:
            if genai is None:
                raise RuntimeError('genai not available')
            # Try creating model with API key parameter if supported
            API_KEY = os.environ.get('GENAI_API_KEY', None)
            try:
                if API_KEY is not None:
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash', api_key=API_KEY)
                    except TypeError:
                        # some genai versions don't accept api_key on constructor
                        model = genai.GenerativeModel('gemini-1.5-flash')
                else:
                    model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                raise RuntimeError(f'Gemini model create failed: {e}')
            prompt = 'Analyze this image for AI artifacts (fingers, textures, background distortion). Is it AI or Real? Give a final verdict.'
            try:
                resp = model.generate_content([prompt, img])
            except Exception as e:
                # Log the exact error and abort Gemini processing
                print('Gemini generate_content failed:', type(e).__name__, e)
                print(traceback.format_exc())
                raise
            if resp and hasattr(resp, 'text'):
                gemini_text = resp.text.strip()
            else:
                gemini_text = str(resp)
            print(f'DEBUG: Gemini Response: [{gemini_text}]')
            # parse simple verdict/conf
            import re
            verdict = None
            confidence = None
            if gemini_text:
                for line in gemini_text.split('\n'):
                    l = line.lower().strip()
                    if l.startswith('verdict:'):
                        verdict = 'ai' if 'ai' in l else 'real'
                    if l.startswith('confidence:'):
                        nums = re.findall(r'\d+', l)
                        if nums:
                            confidence = int(nums[0])
                if verdict is None:
                    if 'synthetic' in gemini_text.lower() or 'ai' in gemini_text.lower():
                        verdict = 'ai'
                    elif 'real' in gemini_text.lower() or 'authentic' in gemini_text.lower():
                        verdict = 'real'
                if confidence is None:
                    nums = re.findall(r'\d+', gemini_text)
                    if nums:
                        confidence = int(nums[0])
                if confidence is None:
                    confidence = 50
                confidence = max(0, min(100, int(confidence)))
                gemini_result = {'verdict': verdict, 'confidence': confidence, 'raw': gemini_text}
        except Exception as e:
            gemini_text = f'Gemini error: {type(e).__name__}: {e}'
            gemini_result = None
            print(f'DEBUG: Gemini Response: [{gemini_text}]')

        # If Gemini succeeded with a decisive confidence, accept it
        if gemini_result and gemini_result.get('verdict') and gemini_result.get('confidence') is not None:
            gc = gemini_result['confidence']
            gv = gemini_result['verdict']
            # decisive thresholds
            if gc >= 60:
                return jsonify({'verdict': 'AI hai photo', 'confidence': gc, 'comments': 'Gemini decisive AI', 'source': 'Gemini'})
            if gc <= 40:
                return jsonify({'verdict': 'Real hai photo', 'confidence': gc, 'comments': 'Gemini decisive Real', 'source': 'Gemini'})

        # If Gemini failed or inconclusive, check watermark before defaulting Real
        wm_flag, wm_info = watermark_check(img)
        if wm_flag:
            print('Watermark-like pattern detected:', wm_info)
            return jsonify({'verdict': 'AI hai photo', 'confidence': 95, 'comments': 'Watermark-like pattern detected', 'source': 'Watermark Heuristic'})

        # If Gemini inconclusive but available, use local model as backup (ensure correct scaling)
        local_conf = None
        if gemini_result is not None and (40 < gemini_result.get('confidence',50) < 60):
            try:
                # Use img_to_array normalization already computed in `normalized`
                import numpy as _np
                inp = _np.array(normalized)
                if inp.ndim == 2:
                    inp = _np.stack([inp, inp, inp], axis=-1)
                if inp.shape[-1] == 4:
                    inp = inp[..., :3]
                inp = _np.expand_dims(inp, 0)
                if local_model is not None:
                    preds = local_model.predict(inp, verbose=0)
                    p = _np.array(preds)
                    if p.size == 1:
                        prob = float(p.flatten()[0])
                    elif p.ndim == 2 and p.shape[1] > 1:
                        prob = float(p[0,1])
                    else:
                        prob = float(p.flatten()[0])
                    if prob > 1 or prob < 0:
                        prob = 1.0 / (1.0 + _np.exp(-prob))
                    local_conf = int(round(max(0.0, min(1.0, prob)) * 100))
                    print(f'DEBUG: Local model confidence: [{local_conf}]')
            except Exception as e:
                print('Local model fallback failed:', e)

        # If local model provided a result, combine 80% Gemini + 20% local
        if gemini_result and local_conf is not None:
            combined = int(round(0.8 * gemini_result['confidence'] + 0.2 * local_conf))
            verdict = 'AI hai photo' if combined >= THRESHOLD else 'Real hai photo'
            return jsonify({'verdict': verdict, 'confidence': combined, 'comments': 'Combined Gemini+Local', 'source': 'Combined'})

        # If Gemini failed entirely or was unavailable, do NOT return a fake score.
        if gemini_result is None:
            # If watermark was found earlier, we already returned AI. Otherwise, surface an explicit error.
            err_details = gemini_text or 'Gemini produced no output'
            print('ERROR: Gemini unavailable or failed:', err_details)
            return jsonify({'error': 'Gemini unavailable or failed', 'details': err_details}), 503

        # Last resort: return Gemini's (non-decisive) result
        if gemini_result:
            gv = gemini_result.get('verdict')
            gc = gemini_result.get('confidence', 50)
            return jsonify({'verdict': 'AI hai photo' if gv == 'ai' else 'Real hai photo', 'confidence': gc, 'comments': 'Gemini inconclusive', 'source': 'Gemini'})

        return jsonify({'error': 'Unhandled path', 'source': 'None'}), 500
    except Exception as e:
        print('Analyze endpoint error:', e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    ensure_model_present()
    load_local_model()
    print("\n" + "="*60)
    print("🚀 AI DETECTOR SERVICE STARTING")
    print("="*60)
    print("Primary: Gemini 1.5 Flash API")
    print("Fallback: Local TensorFlow Model (ai_model.h5)")
    print("Server: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False, threaded=True)
