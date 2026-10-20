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

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_clean_nid(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')

    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edged = cv2.Canny(blurred, 50, 200)
    
    # Find contours for card detection
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    card_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_cnt = approx
            break

    if card_cnt is not None:
        warped = four_point_transform(orig, card_cnt.reshape(4, 2))
    else:
        # Fallback to bounding box crop if 4 points contour fails
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_tb, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_tb:
            c = max(contours_tb, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > 100 and h > 100:
                warped = orig[y:y+h, x:x+w]
            else:
                warped = orig
        else:
            warped = orig

    # Color Adjustments (Remove shadow & increase sharpness)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(enhanced_rgb)

    # Sharpness & Contrast boost
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.25)
    
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
    CARD_W, CARD_H = 1011, 637  # Standard physical NID size (85.6mm x 53.9mm @ 300 DPI)
    A4_W, A4_H = 2480, 3508
    
    front_bytes = await front_file.read()
    front_img = auto_clean_nid(front_bytes)
    front_resized = front_img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
    
    a4_canvas = Image.new('RGB', (A4_W, A4_H), color='white')
    margin_left = (A4_W - CARD_W) // 2
    gap = 140

    if back_file:
        back_bytes = await back_file.read()
        back_img = auto_clean_nid(back_bytes)
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
    
    return StreamingResponse(output, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=NID_A4_Print.pdf"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
