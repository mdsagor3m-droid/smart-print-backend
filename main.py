import os
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Smart Print Hub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_and_enhance_nid(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Safe Bounding Box Cropping (কোনো গ্লিচ বা বাঁকা distortion হবে না)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # যদি সাইজ ঠিক থাকে তবেই ক্রপ করবে, নতুবা পুরো ছবি রাখবে
        if w > 200 and h > 100:
            # একটু মার্জিন রেখে ক্রপ করা যাতে কার্ডের বর্ডার কেটে না যায়
            pad = 10
            h_img, w_img, _ = img.shape
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            img = img[y1:y2, x1:x2]

    # Professional Scan Clean (ব্যাকগ্রাউন্ড উজ্জ্বল এবং লেখা স্পষ্ট করা)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(enhanced_rgb)

    # Contrast & Sharpness Boost
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.25)
    enhancer_sharp = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer_sharp.enhance(1.4)

    return pil_img

@app.get("/")
def home():
    return {"message": "Smart Print Hub API Active"}

@app.post("/api/resize")
async def resize_image(file: UploadFile = File(...), preset: str = Form(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    sizes = {
        "passport": (413, 531),
        "stamp": (236, 295),
        "indian_visa": (600, 600),
        "teletalk_photo": (300, 300),
        "teletalk_signature": (300, 80)
    }
    
    target_size = sizes.get(preset, (300, 300))
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    output = io.BytesIO()
    resized_img.save(output, format="JPEG", quality=95)
    output.seek(0)
    
    return StreamingResponse(output, media_type="image/jpeg", headers={"Content-Disposition": f"attachment; filename=resized_{preset}.jpg"})

@app.post("/api/nid-fix")
async def fix_nid(front_file: UploadFile = File(...), back_file: UploadFile = File(None)):
    CARD_W, CARD_H = 1011, 637  # Standard NID physical size (85.6mm x 53.9mm @ 300 DPI)
    A4_W, A4_H = 2480, 3508
    
    front_bytes = await front_file.read()
    front_img = clean_and_enhance_nid(front_bytes)
    front_resized = front_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
    
    a4_canvas = Image.new('RGB', (A4_W, A4_H), color='white')
    margin_left = (A4_W - CARD_W) // 2
    gap = 140

    if back_file:
        back_bytes = await back_file.read()
        back_img = clean_and_enhance_nid(back_bytes)
        back_resized = back_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
        
        total_content_h = (CARD_H * 2) + gap
        start_y = (A4_H - total_content_h) // 2
        
        a4_canvas.paste(front_resized, (margin_left, start_y))
        a4_canvas.paste(back_resized, (margin_left, start_y + CARD_H + gap))
    else:
        start_y = (A4_H - CARD_H) // 2
        a4_canvas.paste(front_resized, (margin_left, start_y))
    
    output = io.BytesIO()
    a4_canvas.save(output, format="PDF", resolution=300.0)
    output.seek(0)
    
    return StreamingResponse(output, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=NID_A4_Print.pdf"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
