import os
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Smart Print Hub API")

# Blogger এবং যেকোনো ডোমেইন থেকে API কল করার অনুমতি (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Smart Print Hub Backend Server is Active on Render!"}

# ১. ফটো রিসাইজার API (পাসপোর্ট, স্ট্যাম্প, টেলিটক ইত্যাদি)
@app.post("/api/resize")
async def resize_image(file: UploadFile = File(...), preset: str = Form(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # প্রিসেট অনুযায়ী ডাইমেনশন (Pixel @ 300 DPI)
    sizes = {
        "passport": (413, 531),        # 35mm x 45mm
        "stamp": (236, 295),           # 20mm x 25mm
        "indian_visa": (600, 600),     # 2x2 Inch
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

# ২. NID ও আইডি কার্ড অটো-লেআউট ফিক্সার (A4 Printable Size)
@app.post("/api/nid-fix")
async def fix_nid(front_file: UploadFile = File(...), back_file: UploadFile = File(None)):
    # Standard NID Card Physical Dimensions at 300 DPI
    # 85.6mm x 53.9mm -> ~1011 x 637 pixels
    CARD_W, CARD_H = 1011, 637
    
    # Read Front Side
    front_bytes = await front_file.read()
    front_img = Image.open(io.BytesIO(front_bytes)).convert('RGB')
    front_resized = front_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
    
    # A4 Page Creation (2480 x 3508 pixels at 300 DPI)
    a4_canvas = Image.new('RGB', (2480, 3508), color='white')
    
    # Paste Front Side
    margin_left = (2480 - CARD_W) // 2
    margin_top = 300
    a4_canvas.paste(front_resized, (margin_left, margin_top))
    
    # Paste Back Side if provided
    if back_file:
        back_bytes = await back_file.read()
        back_img = Image.open(io.BytesIO(back_bytes)).convert('RGB')
        back_resized = back_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
        a4_canvas.paste(back_resized, (margin_left, margin_top + CARD_H + 150))
    
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
