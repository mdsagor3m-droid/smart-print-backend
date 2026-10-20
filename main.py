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

def auto_crop_and_enhance(img_bytes):
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return pil_img

    # Convert to Grayscale & Blur to find edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding to detect document boundary
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour assuming it's the NID card
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop if valid size
        if w > 100 and h > 100:
            img = img[y:y+h, x:x+w]

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Enhance Brightness & Contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.05)

    return pil_img

@app.get("/")
def home():
    return {"message": "Smart Print Hub Backend Server Active!"}

@app.post("/api/resize")
async def resize_image(file: UploadFile = File(...), preset: str = Form(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    sizes = {
        "passport": (413, 531),        # 35mm x 45mm @ 300 DPI
        "stamp": (236, 295),           # 20mm x 25mm @ 300 DPI
        "indian_visa": (600, 600),     # 2x2 Inch @ 300 DPI
        "teletalk_photo": (300, 300),  # 300x300 px
        "teletalk_signature": (300, 80)# 300x80 px
    }
    
    target_size = sizes.get(preset, (300, 300))
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    output = io.BytesIO()
    resized_img.save(output, format="JPEG", quality=95)
    output.seek(0)
    
    return StreamingResponse(
        output, 
        media_type="image/jpeg", 
        headers={"Content-Disposition": f"attachment; filename=resized_{preset}.jpg"}
    )

@app.post("/api/nid-fix")
async def fix_nid(front_file: UploadFile = File(...), back_file: UploadFile = File(None)):
    CARD_W, CARD_H = 1011, 637  # 85.6mm x 53.9mm @ 300 DPI
    A4_W, A4_H = 2480, 3508     # A4 Canvas @ 300 DPI
    
    # Process Front
    front_bytes = await front_file.read()
    front_img = auto_crop_and_enhance(front_bytes)
    front_resized = front_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
    
    a4_canvas = Image.new('RGB', (A4_W, A4_H), color='white')
    
    margin_left = (A4_W - CARD_W) // 2
    gap = 120  # Gap between front and back
    
    if back_file:
        back_bytes = await back_file.read()
        back_img = auto_crop_and_enhance(back_bytes)
        back_resized = back_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
        
        # Total height of two cards + gap
        total_content_h = (CARD_H * 2) + gap
        start_y = (A4_H - total_content_h) // 2  # Exact Vertical Center
        
        a4_canvas.paste(front_resized, (margin_left, start_y))
        a4_canvas.paste(back_resized, (margin_left, start_y + CARD_H + gap))
    else:
        start_y = (A4_H - CARD_H) // 2  # Exact Vertical Center
        a4_canvas.paste(front_resized, (margin_left, start_y))
    
    output = io.BytesIO()
    a4_canvas.save(output, format="PDF", resolution=300.0)
    output.seek(0)
    
    return StreamingResponse(
        output, 
        media_type="application/pdf", 
        headers={"Content-Disposition": "attachment; filename=NID_A4_Print.pdf"}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
