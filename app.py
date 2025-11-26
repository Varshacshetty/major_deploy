import os
import json
import logging
import time
import shutil
from datetime import datetime   
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress Torch warnings

# Optional genai (chatbot)
genai_available = False
try:
    import google.generativeai as genai
    genai_available = True
    print("genai imported successfully (full chatbot mode)")
except ImportError:
    print("genai not available - using enhanced fallbacks for chatbot")
    genai = None

# Load env vars
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'annotated'), exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'reports'), exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if genai_available and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini configured with key")
    except Exception as e:
        print(f"Gemini config failed: {e} - using fallbacks")
        genai_available = False

# Load herbicide data
HERBICIDE_DATA = {}
try:
    with open('herbicide_data.json', 'r') as f:
        HERBICIDE_DATA = json.load(f)
    logger.info("Herbicide data loaded")
except Exception as e:
    logger.error(f"Failed to load herbicide_data.json: {e}")
    HERBICIDE_DATA = {"fallbacks": {}}

# Enhanced fallbacks (EXPANDED: Added general agriculture topics to handle more queries without Gemini)
ENHANCED_FALLBACKS = HERBICIDE_DATA.get('fallbacks', {})
specific_fallbacks = {
    # Weed-specific (existing)
    "what is goosgrass": "Goosegrass (Eleusine indica) is a tough grassy weed in potato fields, thriving in compacted soil. Control with pre-emergent organics like corn gluten or mulch to prevent spread. Chemicals like quinclorac work but rotate to avoid resistance. IPM tip: Improve drainage. Consult local experts.",
    "goosgrass": "Goosegrass is a common grassy weed competing with potatoes. Use organic pre-emergents first. Consult local experts.",
    "why glyphosate": "Glyphosate kills systemically by blocking EPSPS enzyme. Effective for broad weeds in potatoes but can harm non-targets—use directed sprays and prefer organics. Consult local experts.",
    "glyphostae": "Likely 'glyphosate'. It's a common herbicide but prioritize IPM for sustainability. See 'why glyphosate'. Consult local experts.",
    "multiple weeds": "For mixed weeds (e.g., Goosegrass + Palmer Amaranth), start with IPM: Mulch for prevention, hand-pull small patches, and use selective organics. Monitor potato health weekly. Consult local experts.",
    "no detection": "No weeds—maintain with crop rotation and scouting. Consult local experts.",
    "best weed control": "IPM for potatoes: Cultural (rotation/mulch), mechanical (pulling), organic (corn gluten), chemical last resort. Consult local experts.",
    "weed control": "General weed control in potatoes: Scout weekly, mulch to suppress, hand-pull early. Use organics like corn gluten pre-emergent. Chemicals as last resort. Consult local experts.",
    "nutsedge": "Nutsedge is a perennial sedge weed in wet potato fields. Biology: Tubers regrow easily. Organic: Manual pulling or solarization. Chemical: Halosulfuron early. Improve drainage. Consult local experts.",
    "purslane": "Purslane is a succulent weed in dry potato soils. Biology: Drought-resistant, spreads via stems. Organic: Mulching to smother. Chemical: Spot glyphosate. Till to disrupt. Consult local experts.",
    "goosegrass control": "For Goosegrass in potatoes: Organic - Corn gluten pre-emergent or mulch. Why: Inhibits seeds in compacted soil. Chemical: Quinclorac selective. IPM: Improve soil drainage. Consult local experts.",
    # NEW: General agriculture fallbacks (broader coverage)
    "soil fertility": "To improve soil fertility: Test soil pH/NPK, add compost/manure for organics, rotate crops to prevent depletion. For potatoes, aim for pH 5.5-6.5 with balanced potassium. Sustainable tip: Cover crops like clover. Consult local experts.",
    "pest control": "General pest control in agriculture: Use IPM - scout early, encourage beneficial insects (e.g., ladybugs), apply neem oil organics first. For potatoes, target aphids/Colorado beetles with BT sprays. Avoid broad chemicals. Consult local experts.",
    "irrigation": "Efficient irrigation: Drip systems save water (30-50% less), water deeply but infrequently to encourage roots. For potatoes, maintain even moisture (1-2 inches/week) to avoid cracking. Mulch to retain soil moisture. Consult local experts.",
    "crop rotation": "Crop rotation benefits: Prevents soil diseases/pests, improves fertility (e.g., legumes fix nitrogen). For potatoes, rotate with grains/legumes every 3 years to break cycles. Example: Potatoes → Corn → Beans. Consult local experts.",
    "sustainable farming": "Sustainable agriculture: Integrate organics, reduce tillage (no-till preserves soil), diversify crops. For potatoes, use cover crops and IPM to minimize inputs. Benefits: Healthier soil, lower costs long-term. Consult local experts.",
    "potato farming": "Potato farming basics: Plant in cool weather (50-70°F), space 12 inches apart in loose soil. Fertilize with NPK (high potassium), harvest when vines die. Common issues: Blight - use resistant varieties. Consult local experts.",
    "general advice": "General agriculture advice: Focus on soil health (test annually), water wisely, use IPM for pests/weeds. Sustainable practices like rotation/composting yield better long-term. Tailor to your crop/climate. Consult local experts.",
    "how to": "For 'how to' in agriculture: Start with site assessment (soil/sun), choose suited crops, follow IPM/organics. Specifics depend on query - e.g., planting: Prepare soil, sow at right depth. Consult local experts."
}
ENHANCED_FALLBACKS.update(specific_fallbacks)

import torch
torch.serialization.add_safe_globals([__import__("ultralytics").nn.tasks.DetectionModel])

# YOLO Model
model = None
try:
    model = YOLO("weed_yolov8/weed_detection/weights/best.pt")
    logger.info("YOLO model loaded")
except Exception as e:
    logger.error(f"YOLO load failed: {e}")
    model = None

# Global for latest detection
latest_detection = None

# System Prompt for Gemini (EXPANDED: General agriculture expert, anti-bias for Palmer, potato focus)
SYSTEM_PROMPT_BASE = """You are FarmBot, an expert in all aspects of agriculture, with specialization in potato farming, weed control, IPM, soil management, pest control, irrigation, crop rotation, and sustainable practices. Provide accurate, practical advice for any agriculture-related query. For potato-specific questions, emphasize best practices like IPM and organics. CRITICAL RULE: Do NOT mention or default to Palmer Amaranth unless the query specifically includes 'palmer' or 'amaranth'. Ignore it completely otherwise.

Key Data (use only if relevant - tailor to query):
- Weeds in Potatoes: Goosegrass (organic: corn gluten; chemical: quinclorac). Nutsedge (organic: solarization; chemical: halosulfuron). Purslane (organic: mulching; chemical: glyphosate spot). PalmerAmaranth: [FORBIDDEN unless mentioned] Organic: vinegar sprays; Chemical: glyphosate (rotate).
- Soil: Test pH/NPK annually; add compost for fertility; potatoes prefer 5.5-6.5 pH.
- Pests: IPM first - scout, use BT/neem for organics; avoid broad sprays.
- Irrigation: Drip for efficiency; potatoes need consistent moisture.
- Rotation: 3-year cycle to prevent diseases; follow potatoes with non-solanaceous crops.
- Sustainability: Reduce tillage, use cover crops, minimize chemicals.

General Guidelines:
- STRICTLY base on SPECIFIC query: For general agriculture, give broad sustainable advice. For potatoes, focus on IPM/organics.
- Correct typos (e.g., 'goosgrass' → Goosegrass). Concise (<150 words), use bullets, practical/potato-focused where relevant, end 'Consult local experts.'.
- Stay on-topic: Agriculture only - soil, crops, pests, sustainability; no unrelated info."""

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_image():
    global latest_detection
    if model is None:
        return jsonify({"error": "YOLO model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
    file.save(image_path)

    try:
        annotated_dir = os.path.join(app.config['STATIC_FOLDER'], 'annotated')
        os.makedirs(annotated_dir, exist_ok=True)

        # Predict and save explicitly
        results = model.predict(source=image_path, save=True, save_txt=False, imgsz=640, project=app.config['STATIC_FOLDER'], name='annotated', exist_ok=True)
        
        # Force wait for file creation
        time.sleep(1)
        
        # Get annotated path
        annotated_files = [f for f in os.listdir(annotated_dir) if f.endswith(('.jpg', '.png'))]
        annotated_img_path = None
        if annotated_files:
            latest_file = sorted(annotated_files)[-1]
            annotated_img_path = os.path.join(annotated_dir, latest_file)
            shutil.copy2(annotated_img_path, os.path.join(annotated_dir, 'annotated.jpg'))
            annotated_img_path = os.path.join(annotated_dir, 'annotated.jpg')
        else:
            annotated_img_path = None

        detected = {}
        class_names = results[0].names
        for cls in results[0].boxes.cls:
            class_id = int(cls)
            class_name = class_names[class_id]
            detected[class_name] = detected.get(class_name, 0) + 1
        
        total_detections = len(results[0].boxes)
        crop_count = detected.get('PotatoPlant', 0)
        weed_count = total_detections - crop_count
        crop_ratio = round((crop_count / total_detections * 100), 2) if total_detections > 0 else 0
        weed_ratio = 100 - crop_ratio

        herbicide_recs = {weed: HERBICIDE_DATA.get(weed.replace(' ', ''), {}) for weed in detected if weed != 'PotatoPlant' and detected[weed] > 0}

        # Single unified best_overall
        best_overall = {}
        if herbicide_recs:
            most_common_weed = max(herbicide_recs, key=lambda w: detected.get(w, 0))
            primary_rec = herbicide_recs[most_common_weed]
            best_overall = {
                'organic': primary_rec.get('organic', 'Mulch and hand-pull (IPM priority)'),
                'chemical': primary_rec.get('chemical', 'N/A - Use organics first'),
                'why': f"Based on dominant weed ({most_common_weed}): {primary_rec.get('why_organic', 'Organic methods are sustainable and low-risk for potatoes.')}. Rotate treatments to prevent resistance. Consult local experts."
            }
        else:
            best_overall = {
                'organic': 'Preventive mulching and crop rotation',
                'chemical': 'N/A',
                'why': 'No weeds detected—focus on cultural practices to keep fields clean. Consult local experts.'
            }

        annotated_url = "/static/annotated/annotated.jpg" if annotated_img_path and os.path.exists(annotated_img_path) else None
        logger.info(f"Annotated files in dir: {annotated_files}, Final path exists: {os.path.exists(annotated_img_path) if annotated_img_path else 'None'}")

        response = {
            "annotated_img": annotated_url,
            "detected": detected,
            "herbicide": herbicide_recs,
            "best_overall": best_overall,
            "crop_count": crop_count,
            "weed_count": weed_count,
            "crop_ratio": crop_ratio,
            "weed_ratio": weed_ratio,
            "total_detections": total_detections
        }
        latest_detection = response
        logger.info(f"Detection: {detected}, Image: {annotated_url}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question"}), 400

    lower_q = question.lower().strip()
    
    # Enhanced fuzzy matching (specific overlap only - catches agriculture keywords)
    matched = None
    for key in ENHANCED_FALLBACKS:
        if any(word in lower_q for word in key.lower().split()) or lower_q in key.lower():
            matched = ENHANCED_FALLBACKS[key]
            break
    
    if matched:
        logger.info(f"Fallback matched for: {lower_q[:50]}")
        return jsonify({"answer": matched})

    # Dynamic prompt with context (tailors to detection if relevant)
    dynamic_prompt = SYSTEM_PROMPT_BASE
    detected_weeds = []
    if latest_detection and latest_detection.get('detected'):
        detected_weeds = [w for w in latest_detection['detected'] if w != 'PotatoPlant' and latest_detection['detected'][w] > 0]
        if detected_weeds:
            dynamic_prompt += f"\n\nContext: Recent potato field detection found weeds: {', '.join(detected_weeds)}. If query relates to field management, incorporate this (e.g., weed control in potatoes)."

    dynamic_prompt += f"\n\nUser  Query: {question}. Respond precisely and helpfully."

    # Log for debugging
    logger.info(f"Dynamic prompt for '{question[:50]}': {dynamic_prompt[-200:]}...")  # Last 200 chars

    # Try Gemini
    if genai_available and GEMINI_API_KEY:
        try:
            model_gem = genai.GenerativeModel('gemini-pro', system_instruction=dynamic_prompt)
            chat = model_gem.start_chat(history=[])
            response = chat.send_message(question)
            answer = response.text.strip()

            if not answer:
                raise Exception("Empty response")

            # Post-process: Trim, disclaimer, anti-bias
            if len(answer) > 150:
                answer = answer[:147] + "... Consult local experts."
            else:
                answer += " Consult local experts." if not answer.endswith('experts.') else ""

            # Anti-bias: Override if Palmer unexpected
            if ('palmer' in answer.lower() or 'amaranth' in answer.lower()) and ('palmer' not in lower_q and 'amaranth' not in lower_q):
                logger.warning(f"Bias detected - overriding")
                answer = "General agriculture advice: Focus on sustainable IPM, soil health, and crop rotation for your needs. Consult local experts."

            logger.info(f"Gemini OK: {answer[:50]}...")
            return jsonify({"answer": answer})

        except Exception as e:
            logger.error(f"Gemini error: {e} - alt model")
            try:
                alt_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=dynamic_prompt)
                chat = alt_model.start_chat(history=[])
                response = chat.send_message(question)
                answer = response.text.strip()
                if len(answer) > 150:
                    answer = answer[:147] + "... Consult local experts."
                else:
                    answer += " Consult local experts." if not answer.endswith('experts.') else ""
                # Anti-bias check
                if ('palmer' in answer.lower() or 'amaranth' in answer.lower()) and ('palmer' not in lower_q and 'amaranth' not in lower_q):
                    answer = "General agriculture advice: Focus on sustainable IPM, soil health, and crop rotation for your needs. Consult local experts."
                logger.info(f"Gemini alt OK: {answer[:50]}...")
                return jsonify({"answer": answer})
            except Exception as e2:
                logger.error(f"Alt model failed: {e2} - fallback")

    # General fallback (broad agriculture)
    fallback = f"For '{question}', general agriculture advice: Prioritize soil testing, IPM for pests/weeds, efficient irrigation, and crop rotation for sustainability. Tailor to your crop (e.g., potatoes: high potassium, even moisture). Consult local experts."
    logger.info(f"General fallback for: {lower_q[:50]}")
    return jsonify({"answer": fallback})

class PDF(FPDF):
    def footer(self):
        # Add page number in footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    global latest_detection
    if not latest_detection:
        return jsonify({"error": "Upload and detect first"}), 400

    try:
        pdf = PDF()
        pdf.add_page()
        # Set margins for better alignment (10mm left/right, 15mm top for more space)
        pdf.set_left_margin(10)
        pdf.set_right_margin(10)
        pdf.set_top_margin(15)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Weed Detection Report', ln=True, align='C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        pdf.ln(10)

        # Summary (aligned within margins)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Total Detections: {latest_detection['total_detections']}", ln=True)
        pdf.cell(0, 10, f"Crop (Potatoes): {latest_detection['crop_count']} ({latest_detection['crop_ratio']}%)", ln=True)
        pdf.cell(0, 10, f"Weeds: {latest_detection['weed_count']} ({latest_detection['weed_ratio']}%)", ln=True)
        pdf.ln(10)

        # Image (full width within margins, x=10 for left alignment)
        annotated_path = os.path.join(app.root_path, 'static', 'annotated', 'annotated.jpg')
        image_inserted = False
        if os.path.exists(annotated_path):
            # Get actual image dimensions using cv2 for dynamic scaling
            img = cv2.imread(annotated_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
                if img_width > 0:  # Valid image
                    scaled_height = (img_height / img_width) * 190  # Scale height based on 190mm width
                    pdf.image(annotated_path, x=10, y=pdf.get_y(), w=190)
                    pdf.ln(scaled_height + 20)  # Dynamic move down: scaled height + padding
                    image_inserted = True
                    logger.info(f"Image inserted with scaled height: {scaled_height}mm")
                else:
                    logger.warning("Invalid image dimensions")
            else:
                logger.warning("Could not read image with cv2")
        if not image_inserted:
            pdf.cell(0, 10, 'Annotated image not available', ln=True, align='C')
            pdf.ln(20)  # Extra space if no image

        # Weeds Table (FIXED: Fixed row height, truncation for alignment, no multi_cell misalignment)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Detected Weeds & Recommendations', ln=True)
        pdf.set_font('Arial', 'B', 10)
        # Column widths: sum=190mm, balanced
        col_widths = [50, 20, 60, 60]
        row_height = 10  # Fixed height per row for structure
        # Header row
        pdf.cell(col_widths[0], row_height, 'Weed Type', 1, 0, 'C')
        pdf.cell(col_widths[1], row_height, 'Count', 1, 0, 'C')
        pdf.cell(col_widths[2], row_height, 'Organic Rec', 1, 0, 'C')
        pdf.cell(col_widths[3], row_height, 'Chemical Rec', 1, 0, 'C')
        pdf.ln(row_height)

        pdf.set_font('Arial', '', 9)  # Font for content (fits better)
        detected_weeds = {k: v for k, v in latest_detection['detected'].items() if k != 'PotatoPlant' and v > 0}
        if detected_weeds:
            for weed, count in detected_weeds.items():
                rec = latest_detection['herbicide'].get(weed, {})
                organic = rec.get('organic', 'N/A')
                chemical = rec.get('chemical', 'N/A')
                # Truncate long text to fit column + "..." (prevents overflow/misalignment)
                weed_text = (weed[:47] + '...') if len(weed) > 47 else weed
                organic_text = (organic[:57] + '...') if len(organic) > 57 else organic
                chemical_text = (chemical[:57] + '...') if len(chemical) > 57 else chemical
                # Draw row with fixed height cells + borders
                pdf.cell(col_widths[0], row_height, weed_text, 1, 0, 'L')
                pdf.cell(col_widths[1], row_height, str(count), 1, 0, 'C')
                pdf.cell(col_widths[2], row_height, organic_text, 1, 0, 'L')
                pdf.cell(col_widths[3], row_height, chemical_text, 1, 0, 'L')
                pdf.ln(row_height)  # Move to next row
        else:
            # No weeds row (spans full width)
            pdf.cell(sum(col_widths), row_height, 'No weeds detected - Excellent field!', 1, 0, 'C')
            pdf.ln(row_height)

        # Best Overall (single, aligned below table)
        pdf.ln(10)  # Space after table
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Best Overall Recommendation', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Organic: {latest_detection['best_overall']['organic']}", ln=True)
        pdf.cell(0, 10, f"Chemical: {latest_detection['best_overall']['chemical']}", ln=True)
        pdf.set_font('Arial', '', 9)  # Smaller for why (if long)
        pdf.multi_cell(0, 6, f"Why: {latest_detection['best_overall']['why']}", 0, 'L')

        # Save PDF
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pdf_filename = f"weed_report_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['STATIC_FOLDER'], 'reports', pdf_filename)
        pdf.output(pdf_path)
        logger.info(f"PDF generated successfully: {pdf_path}")

        return jsonify({"pdf": f"/static/reports/{pdf_filename}"})

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({"error": f"PDF failed: {str(e)}. Ensure upload was done first."}), 500

# ---------------- GLOBAL JSON ERROR HANDLER ---------------- #


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for ALL errors (prevents JSON parse failures)."""
    try:
        return jsonify({"error": str(e)}), 500
    except:
        return '{"error": "Unknown server error"}', 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
