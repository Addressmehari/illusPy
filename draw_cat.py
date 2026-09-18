import os
import math
from PIL import Image, ImageDraw

os.makedirs("visualizations", exist_ok=True)

def draw_cat(img_size=512, output_path="visualizations/programmatic_cat.png"):
    """
    Programmatically draws a cute cat image using Pillow (PIL.ImageDraw).
    Uses geometric shapes, curves, layers, and color palettes.
    """
    # 1. Create canvas with a soft pastel background
    img = Image.new("RGBA", (img_size, img_size), color=(240, 243, 246, 255))
    draw = ImageDraw.Draw(img)

    center_x = img_size // 2
    head_y = 220
    head_radius = 110

    # Color Palette
    fur_color = (255, 160, 80)       # Warm Orange Fur
    dark_fur = (220, 130, 50)        # Fur Outline / Stripes
    inner_ear = (255, 180, 190)      # Soft Pink
    eye_outer = (255, 255, 255)      # White Sclera
    eye_iris = (70, 180, 220)        # Sky Blue Iris
    eye_pupil = (20, 20, 30)         # Dark Pupil
    eye_shine = (255, 255, 255)      # Eye Reflection
    nose_color = (255, 120, 140)     # Pink Nose
    line_color = (60, 50, 50)        # Dark Charcoal Outlines
    body_color = (245, 150, 70)      # Body Fur

    # 2. Draw Tail (Curved stroke on background layer)
    tail_pts = []
    for t in range(0, 100):
        rad = (t / 100.0) * math.pi
        tx = center_x + 130 + 30 * math.sin(rad * 1.5)
        ty = 380 + (t * 0.9) - 40 * math.cos(rad)
        tail_pts.append((tx, ty))
    for i in range(len(tail_pts) - 1):
        draw.line([tail_pts[i], tail_pts[i+1]], fill=fur_color, width=22)
        draw.line([tail_pts[i], tail_pts[i+1]], fill=line_color, width=3)

    # 3. Draw Body & Paws
    body_bbox = [center_x - 110, 290, center_x + 110, 470]
    draw.ellipse(body_bbox, fill=body_color, outline=line_color, width=4)

    # Paws
    left_paw = [center_x - 70, 430, center_x - 15, 475]
    right_paw = [center_x + 15, 430, center_x + 70, 475]
    draw.ellipse(left_paw, fill=(255, 255, 255), outline=line_color, width=3)
    draw.ellipse(right_paw, fill=(255, 255, 255), outline=line_color, width=3)

    # Paw toes
    draw.line([(center_x - 52, 445), (center_x - 52, 470)], fill=line_color, width=2)
    draw.line([(center_x - 33, 445), (center_x - 33, 470)], fill=line_color, width=2)
    draw.line([(center_x + 33, 445), (center_x + 33, 470)], fill=line_color, width=2)
    draw.line([(center_x + 52, 445), (center_x + 52, 470)], fill=line_color, width=2)

    # 4. Draw Ears (Outer Orange + Inner Pink Triangles)
    left_ear_outer = [(center_x - 100, head_y - 30), (center_x - 40, head_y - 120), (center_x - 10, head_y - 65)]
    right_ear_outer = [(center_x + 100, head_y - 30), (center_x + 40, head_y - 120), (center_x + 10, head_y - 65)]
    draw.polygon(left_ear_outer, fill=fur_color, outline=line_color)
    draw.polygon(right_ear_outer, fill=fur_color, outline=line_color)

    left_ear_inner = [(center_x - 90, head_y - 35), (center_x - 45, head_y - 105), (center_x - 20, head_y - 65)]
    right_ear_inner = [(center_x + 90, head_y - 35), (center_x + 45, head_y - 105), (center_x + 20, head_y - 65)]
    draw.polygon(left_ear_inner, fill=inner_ear)
    draw.polygon(right_ear_inner, fill=inner_ear)

    # 5. Draw Head
    head_bbox = [center_x - head_radius, head_y - head_radius, center_x + head_radius, head_y + head_radius]
    draw.ellipse(head_bbox, fill=fur_color, outline=line_color, width=4)

    # Head Stripes (Tabby Cat Markings)
    draw.polygon([(center_x - 15, head_y - 105), (center_x, head_y - 75), (center_x + 15, head_y - 105)], fill=dark_fur)
    draw.polygon([(center_x - 35, head_y - 100), (center_x - 25, head_y - 75), (center_x - 15, head_y - 95)], fill=dark_fur)
    draw.polygon([(center_x + 35, head_y - 100), (center_x + 25, head_y - 75), (center_x + 15, head_y - 95)], fill=dark_fur)

    # 6. Draw Eyes (Large expressive cartoon eyes)
    eye_offset_x = 45
    eye_y = head_y - 10
    eye_r = 28

    for sign in [-1, 1]:
        ex = center_x + sign * eye_offset_x
        # Sclera (White)
        draw.ellipse([ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r], fill=eye_outer, outline=line_color, width=3)
        # Iris (Blue)
        draw.ellipse([ex - eye_r + 6, eye_y - eye_r + 4, ex + eye_r - 6, eye_y + eye_r - 4], fill=eye_iris)
        # Pupil (Dark Slit/Oval)
        draw.ellipse([ex - 9, eye_y - 16, ex + 9, eye_y + 16], fill=eye_pupil)
        # Highlights (Shine)
        draw.ellipse([ex - 12, eye_y - 16, ex - 4, eye_y - 8], fill=eye_shine)
        draw.ellipse([ex + 3, eye_y + 4, ex + 8, eye_y + 9], fill=eye_shine)

    # 7. Draw Nose & Mouth
    nose_y = head_y + 35
    nose_tri = [(center_x - 12, nose_y - 8), (center_x + 12, nose_y - 8), (center_x, nose_y + 6)]
    draw.polygon(nose_tri, fill=nose_color, outline=line_color)

    # Mouth Curves
    draw.line([(center_x, nose_y + 6), (center_x, nose_y + 16)], fill=line_color, width=3)
    draw.arc([center_x - 24, nose_y + 8, center_x, nose_y + 26], start=0, end=180, fill=line_color, width=3)
    draw.arc([center_x, nose_y + 8, center_x + 24, nose_y + 26], start=0, end=180, fill=line_color, width=3)

    # Cute Cheek Blush
    draw.ellipse([center_x - 85, head_y + 25, center_x - 55, head_y + 45], fill=(255, 160, 180, 140))
    draw.ellipse([center_x + 55, head_y + 25, center_x + 85, head_y + 45], fill=(255, 160, 180, 140))

    # 8. Draw Whiskers
    whisker_y = nose_y + 4
    # Left Whiskers
    draw.line([(center_x - 45, whisker_y - 10), (center_x - 125, whisker_y - 25)], fill=line_color, width=3)
    draw.line([(center_x - 45, whisker_y), (center_x - 130, whisker_y)], fill=line_color, width=3)
    draw.line([(center_x - 45, whisker_y + 10), (center_x - 120, whisker_y + 20)], fill=line_color, width=3)

    # Right Whiskers
    draw.line([(center_x + 45, whisker_y - 10), (center_x + 125, whisker_y - 25)], fill=line_color, width=3)
    draw.line([(center_x + 45, whisker_y), (center_x + 130, whisker_y)], fill=line_color, width=3)
    draw.line([(center_x + 45, whisker_y + 10), (center_x + 120, whisker_y + 20)], fill=line_color, width=3)

    # 9. Save image
    img = img.convert("RGB")
    img.save(output_path, quality=95)
    print(f"Programmatic Cat image drawn and saved to '{output_path}'!")

if __name__ == "__main__":
    draw_cat()
