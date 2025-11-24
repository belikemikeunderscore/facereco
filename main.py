import cv2
import os
import numpy as np
from pathlib import Path
import time
import datetime

# --- Config ---
DATASET_DIR = Path("dataset")           
MODEL_PATH = Path("lbph_model.yml")
CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_WIDTH, IMG_HEIGHT = 200, 200        
CAPTURE_PER_PRESS = 50                  
SLEEP_BETWEEN_CAPTURES = 0.08          

# typing UI state
typed_label = ""
is_typing = False
typing_stage =  None #label or birthday
birthdays = {} # label_name -> "DD/MM/YYYY"

# --- Ensure dataset dir exists ---
DATASET_DIR.mkdir(exist_ok=True)

# --- Load face detector ---
if not Path(CASCADE_PATH).exists():
    print(f"[!] não foi possivel encontrar o cascade '{CASCADE_PATH}'. mete o ficheiro cascade Haar no diretorio do programa.")
    raise SystemExit(1)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# --- Create LBPH recognizer ---
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except Exception as e:
    print("[!] falha ao criar o LBPH recognizer. por favor instala o opencv-contrib-python.")
    raise

# --- Helper functions ---

def calc_age(birthday_str):
    try:
        # Expected format: DD/MM/YYYY
        d, m, y = map(int, birthday_str.split("/"))
        birth = datetime.date(y, m, d)
        today = datetime.date.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return age
    except:
        return None
    
def list_labels():
    return sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])

def get_face(gray, rect):
    x, y, w, h = rect
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))

def start_typing_label(current_label):
    global typed_label, is_typing
    typed_label = ""
    is_typing = True
    return None  # clear current label

def capture_samples(label_name, cam, display_window="LBPH"):
    label_dir = DATASET_DIR / label_name
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(label_dir.glob("*.jpg")))
    print(f"[i] a gravar {CAPTURE_PER_PRESS} capturas em '{label_dir}' (no total existem {existing})")
    saved, tries = 0, 0

    while saved < CAPTURE_PER_PRESS and tries < CAPTURE_PER_PRESS * 8:
        ret, frame = cam.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
        display = frame.copy()

        if len(faces) > 0:
            face_rect = max(faces, key=lambda r: r[2]*r[3])
            face_img = get_face(gray, face_rect)
            filename = label_dir / f"{int(time.time()*1000)}_{saved+existing}.jpg"
            cv2.imwrite(str(filename), face_img)
            saved += 1
            x, y, w, h = face_rect
            cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(display, f"Saved {saved}/{CAPTURE_PER_PRESS}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # show the capture on main LBPH window
        cv2.imshow(display_window, display)

        tries += 1
        if cv2.waitKey(int(SLEEP_BETWEEN_CAPTURES*1000)) & 0xFF == 27: break

    print(f"[i] feito: salvas {saved} imagens para '{label_name}'.")


def prepare_training_data():
    label_names = list_labels()
    if not label_names:
        print("[!] sem labels encontradas.")
        return [], [], {}
    label2id = {name: idx for idx, name in enumerate(label_names)}
    faces, labels = [], []

    for name, id_ in label2id.items():
        for img_path in (DATASET_DIR / name).glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            if img.shape != (IMG_HEIGHT, IMG_WIDTH): img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            faces.append(img); labels.append(id_)

    print(f"[i] dados preparados para treino: {len(faces)} imagens, {len(label2id)} labels.")
    return faces, labels, label2id

def train_recognizer():
    faces, labels, label2id = prepare_training_data()
    if len(faces) < 5:
        print("[!] dados insuficientes.")
        return None, None
    recognizer.train(faces, np.array(labels))
    print("[i] treino finalizado.")
    return recognizer, label2id

def save_model():
    """Saves recognizer + labels + birthdays in consistent format."""
    recognizer.write(str(MODEL_PATH))
    label_file = MODEL_PATH.with_suffix(".labels.txt")

    with open(label_file, "w", encoding="utf-8") as f:
        for name in list_labels():
            # ALWAYS write name and comma (birthday may be empty)
            birthday = birthdays.get(name, "")
            f.write(f"{name},{birthday}\n")

    print("[i] Modelo e birthdays salvos.")

def load_model():
    """Loads recognizer + labels + birthdays robustly (even if blank)."""
    if not MODEL_PATH.exists():
        print("[!] modelo não encontrado.")
        return None, {}

    recognizer.read(str(MODEL_PATH))
    label2id = {}
    birthdays.clear()

    label_file = MODEL_PATH.with_suffix(".labels.txt")
    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                # Always split into name + birthday (even if blank)
                parts = line.split(",", 1)
                name = parts[0].strip()
                birthday = parts[1].strip() if len(parts) > 1 else ""

                # Store ID
                label2id[name] = idx

                # ALWAYS store a birthday key (even blank)
                birthdays[name] = birthday

    return recognizer, label2id

def input_birthday_for_label(label_name):
    global typed_label, is_typing
    typed_label = ""
    is_typing = True
    input_stage = "birthday"  # distinguish typing a name vs birthday
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [13,10]:  # Enter
            if typed_label.strip() != "":
                birthdays[label_name] = typed_label.strip()
                print(f"[i] Birthday for {label_name} set to {typed_label.strip()}")
            is_typing = False
            break
        elif key in [8,127]: typed_label = typed_label[:-1]
        elif key==27: 
            is_typing = False
            break
        elif 32<=key<=126: typed_label += chr(key)

# ------------------- MAIN LOOP -----------------------
def main():
    global typed_label, is_typing
    cam = cv2.VideoCapture(0)
    if not cam.isOpened(): return
    current_label = None
    label2id = {}
    mode = "idle"
    print("programa inicializado. (pressionar teclas na janela)")
    res = load_model()
    if res:
        _, label2id = res
        print("[i] modelo + aniversários carregados ao iniciar.")
    while True:
        ret, frame = cam.read()
        if not ret: break
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.2,5,minSize=(60,60))

        # recognition
        if mode == "recognize" and len(faces)>0:
            id2label = {v:k for k,v in label2id.items()}
            for r in faces:
                x,y,w,h = r
                face_img = get_face(gray,r)
                try:
                    label_id, conf = recognizer.predict(face_img)

                    CONF_THRESHOLD = 60

                    if conf <= 30:
                        color = (0, 255, 0)         # green → very confident
                        name = id2label.get(label_id, f"id_{label_id}")
                    elif conf <= 50:
                        color = (0, 200, 200)       # yellowish → medium confidence
                        name = id2label.get(label_id, f"id_{label_id}")
                    elif conf <= 60:
                        color = (0, 165, 255)       # orange → low confidence
                        name = id2label.get(label_id, f"id_{label_id}")
                    else:
                        color = (0, 0, 255)         # red → unknown
                        name = "desconhecido"

                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(display, f"{name} ({conf:.1f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # === Birthday & age display ===
                    bday = birthdays.get(name, "")

                    if bday:
                        age = calc_age(bday)
                        if age is not None:
                            birth_color = (255, 255, 255)
                            bday_text = f"{age} Anos ({bday})"
                        else:
                            birth_color = (100, 100, 100)
                            bday_text = f"{bday}"
                    else:
                        # no birthday text if unknown
                        bday_text = None

                    # Draw birthday text ONLY if recognized person
                    if bday_text:
                        cv2.putText(display, bday_text, (x, y+h+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, birth_color, 2)
                except:
                    pass

        # bottom text
        cv2.putText(display, f"modo: {mode} | pessoa/label: {current_label}",
                    (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # controls overlay
        for i, line in enumerate(["N: Novo nome  |  C: Capturar  |  T: Treinar",
                                  "R: Reconhecer  |  S: Salvar  |  L: Carregar  |  Q/ESC: Sair"]):
            cv2.putText(display, line, (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        # typing UI
        if is_typing:
            if typing_stage == "label":
                prompt = f"inserir nome: {typed_label}"
            elif typing_stage == "birthday":
                prompt = f"inserir aniversário (DD/MM/YYYY): {typed_label}"
            cv2.putText(display, prompt,
                        (10, display.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.imshow("LBPH", display)
        key = cv2.waitKey(1) & 0xFF
        # exit
        if key in [ord('q'), 27] and not is_typing: break

        # typing handler
        if is_typing:
            if key in [13,10]:  # Enter
                if typed_label.strip() != "":
                    if typing_stage == "label":
                        current_label = typed_label.strip()
                        print(f"[i] nome definido para: {current_label}")
                        # now start birthday input
                        typed_label = ""
                        is_typing = True
                        typing_stage = "birthday"
                        print(f"[i] Digite o aniversario para {current_label} (DD/MM/YYYY), ou deixe vazio se não quiser)")
                    elif typing_stage == "birthday":
                        if typed_label.strip() != "":
                            birthdays[current_label] = typed_label.strip()
                            print(f"[i] Birthday for {current_label} set to {typed_label.strip()}")
                        is_typing = False
                        typing_stage = None
                else:
                    is_typing = False
                    typing_stage = None
            elif key in [8,127]: typed_label = typed_label[:-1]
            elif key==27: is_typing=False; typing_stage=None
            elif 32<=key<=126: typed_label += chr(key)
            continue

        # commands
        if key == ord('n') and not is_typing:
            typed_label = ""
            is_typing = True
            typing_stage = "label"  # now typing a new label

        elif key==ord('c'):
            if current_label:
                mode="capture"
                capture_samples(current_label, cam, display_window="LBPH")
                mode="idle"
            else:
                print("[!] inserir label primeiro (usar N).")
        elif key==ord('t'):
            model,l2id=train_recognizer(); 
            if model: label2id=l2id; mode="idle"
        elif key==ord('s'): save_model()
        elif key==ord('l'):
            res=load_model()
            if res: _,label2id=res
        elif key==ord('r'):
            faces,labels,l2id=prepare_training_data()
            if faces: recognizer.train(faces,np.array(labels)); label2id=l2id; mode="recognize"
            else: print("[!] sem dados de treino.")

    cam.release()
    cv2.destroyAllWindows()

if __name__=="__main__": main()
