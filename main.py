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

def process_nid_image(img_bytes):
    # Read image bytes using OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Convert to RGB for PIL processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Clean lighting, boost contrast and remove dull background noise to look like a scanned doc
    enhancer_contrast = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer_contrast.enhance(1.3)

    enhancer_sharp = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer_sharp.enhance(1.4)

    enhancer_bright = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer_bright.enhance(1.1)

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
    
    # Process Front Image
    front_bytes = await front_file.read()
    front_img = process_nid_image(front_bytes)
    front_resized = front_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
    
    # Create pure white A4 canvas
    a4_canvas = Image.new('RGB', (A4_W, A4_H), color='white')
    margin_left = (A4_W - CARD_W) // 2
    gap = 140

    if back_file:
        back_bytes = await back_file.read()
        back_img = process_nid_image(back_bytes)
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
