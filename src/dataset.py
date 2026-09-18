import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 128

def create_messy_square(img_size=128):
    """Generates an imperfect, messy, hand-drawn-style square image."""
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)

    center_x = img_size / 2 + random.uniform(-10, 10)
    center_y = img_size / 2 + random.uniform(-10, 10)
    half_side = random.uniform(25, 40)
    angle = random.uniform(-0.3, 0.3)

    base_vertices = [
        (-half_side, -half_side),
        (half_side, -half_side),
        (half_side, half_side),
        (-half_side, half_side)
    ]

    vertices = []
    for vx, vy in base_vertices:
        rx = vx * math.cos(angle) - vy * math.sin(angle)
        ry = vx * math.sin(angle) + vy * math.cos(angle)
        jx = rx + center_x + random.uniform(-6, 6)
        jy = ry + center_y + random.uniform(-6, 6)
        vertices.append((jx, jy))

    stroke_width = random.randint(2, 5)
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        num_segments = random.randint(5, 10)
        pts = [p1]
        for s in range(1, num_segments):
            t = s / num_segments
            bx = p1[0] + t * (p2[0] - p1[0]) + random.uniform(-2, 2)
            by = p1[1] + t * (p2[1] - p1[1]) + random.uniform(-2, 2)
            pts.append((bx, by))
        pts.append(p2)
        
        for k in range(len(pts) - 1):
            draw.line([pts[k], pts[k+1]], fill=255, width=stroke_width)

    img_np = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 10, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def create_messy_circle(img_size=128):
    """Generates an imperfect, messy, hand-drawn-style circle image."""
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)

    center_x = img_size / 2 + random.uniform(-10, 10)
    center_y = img_size / 2 + random.uniform(-10, 10)
    rx = random.uniform(25, 40)
    ry = rx * random.uniform(0.8, 1.2)

    num_points = random.randint(30, 60)
    pts = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        r_jitter = random.uniform(-3, 3)
        px = center_x + (rx + r_jitter) * math.cos(angle)
        py = center_y + (ry + r_jitter) * math.sin(angle)
        pts.append((px, py))

    stroke_width = random.randint(2, 5)
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        draw.line([p1, p2], fill=255, width=stroke_width)

    img_np = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 10, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def create_programmatic_cat(img_size=128):
    """
    Generates highly diverse, distinct cat images with different face shapes,
    ear types (pointy, tall, folded), eye expressions, mouth styles, and markings.
    """
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)

    center_x = img_size / 2 + random.uniform(-12, 12)
    head_y = img_size / 2 + random.uniform(-8, 8)
    
    # 1. Cat Head Shape Variety (Chubby round, Slender oval, Fluffy wide)
    head_style = random.choice(["round", "oval", "chubby", "triangular"])
    rx = random.uniform(22, 36)
    ry = rx * random.uniform(0.75, 1.25)
    if head_style == "chubby":
        rx *= 1.25
    elif head_style == "oval":
        ry *= 1.2

    stroke_w = random.randint(2, 4)
    fur_fill = random.randint(80, 220)

    # 2. Ear Variety (Pointy tall, Short stubby, Folded down, Wide spread)
    ear_type = random.choice(["tall_pointy", "short_stubby", "folded", "wide_spread"])
    ear_h = random.uniform(16, 32)
    ear_w = random.uniform(12, 22)

    if ear_type == "tall_pointy":
        ear_h *= 1.3
    elif ear_type == "short_stubby":
        ear_h *= 0.7
    elif ear_type == "wide_spread":
        ear_w *= 1.4

    if ear_type == "folded":
        # Folded ears bend downward
        left_ear = [(center_x - rx + 4, head_y - ry * 0.2), (center_x - rx * 0.6, head_y - ry - 4), (center_x - 4, head_y - ry + 4)]
        right_ear = [(center_x + 4, head_y - ry + 4), (center_x + rx * 0.6, head_y - ry - 4), (center_x + rx - 4, head_y - ry * 0.2)]
    else:
        left_ear = [(center_x - rx + 4, head_y - ry * 0.3), (center_x - rx * 0.5 - ear_w / 2, head_y - ry - ear_h), (center_x - 4, head_y - ry * 0.8)]
        right_ear = [(center_x + 4, head_y - ry * 0.8), (center_x + rx * 0.5 + ear_w / 2, head_y - ry - ear_h), (center_x + rx - 4, head_y - ry * 0.3)]

    # Draw Ears
    draw.polygon(left_ear, outline=255, fill=fur_fill // 2, width=stroke_w)
    draw.polygon(right_ear, outline=255, fill=fur_fill // 2, width=stroke_w)

    # Inner Ear details
    if random.random() > 0.3:
        inner_l = [(p[0] * 0.85 + center_x * 0.15, p[1] * 0.85 + head_y * 0.15) for p in left_ear]
        inner_r = [(p[0] * 0.85 + center_x * 0.15, p[1] * 0.85 + head_y * 0.15) for p in right_ear]
        draw.polygon(inner_l, fill=230)
        draw.polygon(inner_r, fill=230)

    # Draw Head
    draw.ellipse([center_x - rx, head_y - ry, center_x + rx, head_y + ry], outline=255, fill=fur_fill, width=stroke_w)

    # 3. Tabby Forehead Stripes / Markings
    marking_style = random.choice(["m_stripe", "spots", "crown", "none"])
    if marking_style == "m_stripe":
        draw.line([(center_x - 8, head_y - ry + 6), (center_x - 4, head_y - ry + 16), (center_x, head_y - ry + 8), (center_x + 4, head_y - ry + 16), (center_x + 8, head_y - ry + 6)], fill=255, width=2)
    elif marking_style == "spots":
        draw.ellipse([center_x - 6, head_y - ry + 6, center_x + 6, head_y - ry + 16], fill=255)
    elif marking_style == "crown":
        draw.polygon([(center_x - 12, head_y - ry + 4), (center_x, head_y - ry + 18), (center_x + 12, head_y - ry + 4)], fill=255)

    # 4. Eye Expressions (Big round, Winking, Squinty happy, Sleepy)
    eye_style = random.choice(["big_round", "squinty_happy", "winking", "sleepy_slit"])
    eye_offset = rx * 0.45
    eye_y = head_y - ry * 0.1
    eye_r = random.uniform(4, 8)

    for idx_side, sign in enumerate([-1, 1]):
        ex = center_x + sign * eye_offset
        if eye_style == "big_round" or (eye_style == "winking" and sign == -1):
            draw.ellipse([ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r], fill=255)
            # Pupil shape
            pupil_w = random.choice([1, 2, 4])
            draw.ellipse([ex - pupil_w, eye_y - eye_r + 2, ex + pupil_w, eye_y + eye_r - 2], fill=0)
        elif eye_style == "squinty_happy" or (eye_style == "winking" and sign == 1):
            draw.arc([ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r], start=190, end=350, fill=255, width=3)
        elif eye_style == "sleepy_slit":
            draw.line([(ex - eye_r, eye_y), (ex + eye_r, eye_y)], fill=255, width=3)

    # 5. Nose & Mouth Styles
    nose_y = head_y + ry * 0.35
    nose_style = random.choice(["triangle", "dot", "heart"])
    if nose_style == "triangle":
        draw.polygon([(center_x - 4, nose_y - 3), (center_x + 4, nose_y - 3), (center_x, nose_y + 4)], fill=255)
    elif nose_style == "dot":
        draw.ellipse([center_x - 3, nose_y - 3, center_x + 3, nose_y + 3], fill=255)
    elif nose_style == "heart":
        draw.ellipse([center_x - 4, nose_y - 4, center_x, nose_y], fill=255)
        draw.ellipse([center_x, nose_y - 4, center_x + 4, nose_y], fill=255)
        draw.polygon([(center_x - 4, nose_y - 2), (center_x + 4, nose_y - 2), (center_x, nose_y + 4)], fill=255)

    # Mouth
    mouth_style = random.choice(["w_smile", "open_happy", "flat"])
    if mouth_style == "w_smile":
        draw.line([(center_x, nose_y + 3), (center_x, nose_y + 8)], fill=255, width=2)
        draw.arc([center_x - 10, nose_y + 4, center_x, nose_y + 14], start=0, end=180, fill=255, width=2)
        draw.arc([center_x, nose_y + 4, center_x + 10, nose_y + 14], start=0, end=180, fill=255, width=2)
    elif mouth_style == "open_happy":
        draw.chord([center_x - 8, nose_y + 4, center_x + 8, nose_y + 16], start=0, end=180, fill=255)
    elif mouth_style == "flat":
        draw.line([(center_x - 6, nose_y + 8), (center_x + 6, nose_y + 8)], fill=255, width=2)

    # 6. Whiskers (Angled, Straight, Droopy, Multi-line)
    whisker_style = random.choice(["angled_up", "straight", "droopy"])
    w_count = random.randint(2, 4)
    w_y = nose_y + 2

    for sign in [-1, 1]:
        for w in range(w_count):
            angle_offset = (w - (w_count - 1) / 2) * 12
            if whisker_style == "angled_up":
                angle_offset -= 8
            elif whisker_style == "droopy":
                angle_offset += 12

            length = rx * random.uniform(0.9, 1.4)
            rad = math.radians(angle_offset)
            end_x = center_x + sign * (rx * 0.4 + length * math.cos(rad))
            end_y = w_y + length * math.sin(rad)
            draw.line([(center_x + sign * rx * 0.4, w_y + w * 2), (end_x, end_y)], fill=255, width=2)

    # Mild noise
    img_np = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 10, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def create_non_shape(target_shape="square", img_size=128):
    """Generates control non-shape images."""
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)
    stroke_width = random.randint(2, 5)
    center_x = img_size / 2 + random.uniform(-10, 10)
    center_y = img_size / 2 + random.uniform(-10, 10)

    if target_shape == "square":
        shape_type = random.choice(["circle", "triangle", "lines", "noise"])
    elif target_shape == "circle":
        shape_type = random.choice(["square", "triangle", "lines", "noise"])
    elif target_shape == "cat":
        shape_type = random.choice(["square", "circle", "triangle", "lines", "noise"])
    else:
        shape_type = random.choice(["lines", "noise"])

    if shape_type == "circle":
        r = random.uniform(20, 40)
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], outline=255, width=stroke_width)
    elif shape_type == "square":
        h = random.uniform(25, 40)
        draw.rectangle([center_x - h, center_y - h, center_x + h, center_y + h], outline=255, width=stroke_width)
    elif shape_type == "triangle":
        r = random.uniform(25, 40)
        angles = [random.uniform(0, 2), random.uniform(2, 4), random.uniform(4, 6)]
        pts = [(center_x + r * math.cos(a), center_y + r * math.sin(a)) for a in angles]
        draw.polygon(pts, outline=255, width=stroke_width)
    elif shape_type == "lines":
        for _ in range(random.randint(2, 5)):
            x1, y1 = random.randint(10, 118), random.randint(10, 118)
            x2, y2 = random.randint(10, 118), random.randint(10, 118)
            draw.line([(x1, y1), (x2, y2)], fill=255, width=stroke_width)
    elif shape_type == "noise":
        img_np = np.random.randint(0, 150, (img_size, img_size), dtype=np.uint8)
        return Image.fromarray(img_np)

    img_np = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 10, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def generate_full_dataset(target_shape="square", base_dir="dataset", count=100):
    """Generates 100 positive images and 100 negative images for target_shape."""
    pos_dir = os.path.join(base_dir, f"{target_shape}s")
    neg_dir = os.path.join(base_dir, f"non_{target_shape}s")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    print(f"Generating {count} diverse {target_shape} images...")
    for i in range(count):
        if target_shape == "square":
            img = create_messy_square(IMG_SIZE)
        elif target_shape == "circle":
            img = create_messy_circle(IMG_SIZE)
        elif target_shape == "cat":
            img = create_programmatic_cat(IMG_SIZE)
        else:
            img = create_messy_square(IMG_SIZE)

        img.save(os.path.join(pos_dir, f"{target_shape}_{i:03d}.png"))

    print(f"Generating {count} non-{target_shape} control images...")
    for i in range(count):
        img = create_non_shape(target_shape, IMG_SIZE)
        img.save(os.path.join(neg_dir, f"non_{target_shape}_{i:03d}.png"))

    print(f"Dataset for '{target_shape}' successfully generated in '{base_dir}'!")
