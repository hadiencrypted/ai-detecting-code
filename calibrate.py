import os
import json
import numpy as np

from PIL import Image

# Calibration script: looks for labeled files in uploads/
# Labels precedence:
# - If uploads/labels.csv exists, it should be: filename,label (label: ai or real)
# - Otherwise, infer from filename containing 'ai' or 'synthetic' => ai, 'real' or 'auth' => real

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ai_model.h5')
UPLOADS = os.path.join(os.path.dirname(__file__), 'uploads')

def find_labeled_files():
    labels = {}
    csvp = os.path.join(UPLOADS, 'labels.csv')
    if os.path.exists(csvp):
        with open(csvp, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    labels[parts[0].strip()] = parts[1].strip().lower()
    else:
        for fn in os.listdir(UPLOADS):
            low = fn.lower()
            if any(x in low for x in ['ai', 'synthetic']):
                labels[fn] = 'ai'
            elif any(x in low for x in ['real', 'auth']):
                labels[fn] = 'real'
    return labels


def load_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print('Loaded model for calibration:', MODEL_PATH)
        return model
    except Exception as e:
        print('Failed to load model for calibration:', e)
        return None


def predict_conf(model, pil_img):
    arr = np.array(pil_img.resize((224,224))).astype('float32')/255.0
    if arr.ndim == 2:
        arr = np.stack([arr,arr,arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr,0)
    preds = model.predict(arr)
    p = np.array(preds)
    if p.size == 1:
        prob = float(p.flatten()[0])
    elif p.ndim == 2 and p.shape[1] > 1:
        prob = float(p[0,1])
    else:
        prob = float(p.flatten()[0])
    if prob > 1 or prob < 0:
        try:
            prob = 1.0/(1.0+np.exp(-prob))
        except Exception:
            prob = max(0.0, min(1.0, prob))
    return int(round(prob*100))


def main():
    labels = find_labeled_files()
    if not labels:
        print('No labeled files found in uploads/. Add files named with ai/real in filename or provide uploads/labels.csv')
        return
    model = load_model()
    if model is None:
        print('Local model not available. Cannot calibrate.')
        return

    y_true = []
    scores = []
    for fn, lab in labels.items():
        p = os.path.join(UPLOADS, fn)
        if not os.path.exists(p):
            print('Missing file:', p)
            continue
        img = Image.open(p).convert('RGB')
        conf = predict_conf(model, img)
        scores.append(conf)
        y_true.append(1 if lab=='ai' else 0)
        print(fn, 'label=', lab, 'conf=', conf)

    if not scores:
        print('No valid labeled images to calibrate')
        return

    # find best threshold maximizing accuracy
    best_t = 0
    best_acc = -1
    for t in range(0,101):
        preds = [1 if s>=t else 0 for s in scores]
        acc = sum(int(p==y) for p,y in zip(preds,y_true))/len(y_true)
        if acc>best_acc:
            best_acc=acc
            best_t=t
    print('Best threshold:', best_t, 'accuracy:', best_acc)

    cfg = {'threshold': best_t}
    with open(os.path.join(os.path.dirname(__file__), 'detector_config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    print('Wrote detector_config.json with threshold', best_t)

if __name__=='__main__':
    main()
