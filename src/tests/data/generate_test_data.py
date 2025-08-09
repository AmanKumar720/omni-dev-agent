"""
Generate sample test data for AI vision tests
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_test_images():
    """Create sample test images for different vision tasks"""
    
    data_dir = Path(__file__).parent
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("Generating test images...")
    
    # 1. Classification test images
    create_cat_image().save(images_dir / "cat_sample.jpg")
    create_dog_image().save(images_dir / "dog_sample.jpg")
    create_car_image().save(images_dir / "car_sample.jpg")
    create_natural_scene().save(images_dir / "nature_sample.jpg")
    
    # 2. Object detection test images
    create_multi_object_scene().save(images_dir / "multi_object_scene.jpg")
    create_crowd_scene().save(images_dir / "crowd_scene.jpg")
    create_traffic_scene().save(images_dir / "traffic_scene.jpg")
    
    # 3. OCR test images
    create_text_document().save(images_dir / "text_document.jpg")
    create_receipt_image().save(images_dir / "receipt_sample.jpg")
    create_signage_image().save(images_dir / "signage_sample.jpg")
    
    # 4. Face recognition test images
    create_face_image("frontal").save(images_dir / "face_frontal.jpg")
    create_face_image("profile").save(images_dir / "face_profile.jpg")
    create_group_photo().save(images_dir / "group_photo.jpg")
    
    # 5. Low quality / edge case images
    create_blurry_image().save(images_dir / "blurry_sample.jpg")
    create_low_contrast_image().save(images_dir / "low_contrast_sample.jpg")
    create_noisy_image().save(images_dir / "noisy_sample.jpg")
    
    print(f"Generated {len(list(images_dir.glob('*.jpg')))} test images")


def create_cat_image():
    """Create a cat-like image for classification testing"""
    img = Image.new('RGB', (224, 224), color=(245, 245, 220))
    draw = ImageDraw.Draw(img)
    
    # Cat silhouette
    draw.ellipse([40, 80, 180, 180], fill=(255, 140, 0), outline='black', width=2)
    draw.ellipse([70, 40, 150, 120], fill=(255, 140, 0), outline='black', width=2)
    
    # Ears
    draw.polygon([(75, 50), (85, 20), (95, 50)], fill=(255, 140, 0), outline='black')
    draw.polygon([(125, 50), (135, 20), (145, 50)], fill=(255, 140, 0), outline='black')
    
    # Eyes
    draw.ellipse([90, 70, 100, 80], fill='green', outline='black')
    draw.ellipse([120, 70, 130, 80], fill='green', outline='black')
    
    # Nose and whiskers
    draw.polygon([(105, 85), (115, 95), (95, 95)], fill='pink')
    draw.line([(80, 90), (105, 85)], fill='black', width=1)
    draw.line([(105, 85), (135, 90)], fill='black', width=1)
    
    return img


def create_dog_image():
    """Create a dog-like image for classification testing"""
    img = Image.new('RGB', (224, 224), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Dog silhouette
    draw.ellipse([30, 90, 190, 190], fill=(139, 69, 19), outline='black', width=2)
    draw.ellipse([60, 50, 160, 130], fill=(139, 69, 19), outline='black', width=2)
    
    # Floppy ears
    draw.ellipse([45, 60, 75, 110], fill=(101, 67, 33), outline='black')
    draw.ellipse([145, 60, 175, 110], fill=(101, 67, 33), outline='black')
    
    # Eyes and nose
    draw.ellipse([85, 80, 95, 90], fill='black')
    draw.ellipse([125, 80, 135, 90], fill='black')
    draw.ellipse([105, 95, 115, 105], fill='black')
    
    # Mouth and tongue
    draw.arc([95, 105, 125, 125], start=0, end=180, fill='black', width=2)
    draw.ellipse([100, 105, 120, 120], fill='pink')
    
    return img


def create_car_image():
    """Create a car-like image for classification testing"""
    img = Image.new('RGB', (224, 224), color=(192, 192, 192))
    draw = ImageDraw.Draw(img)
    
    # Car body
    draw.rectangle([40, 100, 180, 160], fill='red', outline='darkred', width=2)
    draw.rectangle([60, 80, 160, 100], fill='red', outline='darkred', width=2)
    
    # Wheels
    draw.ellipse([50, 150, 80, 180], fill='black', outline='gray', width=2)
    draw.ellipse([140, 150, 170, 180], fill='black', outline='gray', width=2)
    draw.ellipse([60, 160, 70, 170], fill='silver')
    draw.ellipse([150, 160, 160, 170], fill='silver')
    
    # Windows and lights
    draw.rectangle([70, 85, 150, 95], fill='lightblue', outline='blue')
    draw.ellipse([175, 115, 185, 135], fill='yellow', outline='orange')
    draw.ellipse([175, 125, 185, 145], fill='yellow', outline='orange')
    
    return img


def create_natural_scene():
    """Create a natural scene for classification testing"""
    img = Image.new('RGB', (224, 224), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Ground and sky
    draw.rectangle([0, 150, 224, 224], fill=(34, 139, 34))
    
    # Tree
    draw.rectangle([90, 100, 110, 150], fill=(139, 69, 19))
    draw.ellipse([60, 60, 140, 120], fill=(0, 100, 0), outline='darkgreen')
    
    # Sun and clouds
    draw.ellipse([170, 20, 200, 50], fill='yellow', outline='orange')
    draw.ellipse([20, 30, 60, 50], fill='white', outline='lightgray')
    
    # Flowers
    for i in range(5):
        x = 20 + i * 30
        y = 160 + (i % 3) * 5
        draw.ellipse([x, y, x+10, y+10], fill='red', outline='darkred')
    
    return img


def create_multi_object_scene():
    """Create scene with multiple objects for detection testing"""
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Person 1
    draw.ellipse([100, 50, 140, 90], fill='peachpuff', outline='black')
    draw.rectangle([110, 90, 130, 150], fill='blue', outline='black')
    draw.rectangle([90, 100, 110, 120], fill='peachpuff', outline='black')
    draw.rectangle([130, 100, 150, 120], fill='peachpuff', outline='black')
    
    # Person 2 (smaller, in background)
    draw.ellipse([400, 80, 430, 110], fill='peachpuff', outline='black')
    draw.rectangle([405, 110, 425, 160], fill='red', outline='black')
    
    # Car
    draw.rectangle([250, 250, 450, 320], fill='green', outline='black')
    draw.ellipse([270, 310, 310, 350], fill='black', outline='gray')
    draw.ellipse([390, 310, 430, 350], fill='black', outline='gray')
    
    # Bicycle
    draw.ellipse([500, 200, 540, 240], fill=None, outline='black', width=3)
    draw.ellipse([540, 200, 580, 240], fill=None, outline='black', width=3)
    draw.line([(520, 220), (560, 220)], fill='black', width=2)
    draw.line([(520, 200), (540, 220)], fill='black', width=2)
    
    return img


def create_crowd_scene():
    """Create a scene with multiple people for detection testing"""
    img = Image.new('RGB', (640, 480), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Background
    draw.rectangle([0, 350, 640, 480], fill='darkgray')  # Ground
    
    # Multiple people at different scales and positions
    people_positions = [
        (80, 100, 1.0),   # (x, y, scale)
        (200, 120, 0.8),
        (320, 110, 0.9),
        (450, 130, 0.7),
        (550, 140, 0.6),
        (150, 200, 1.2),  # Closer person
        (380, 180, 1.1),
    ]
    
    for x, y, scale in people_positions:
        # Scale dimensions
        head_size = int(30 * scale)
        body_width = int(20 * scale)
        body_height = int(60 * scale)
        
        # Head
        draw.ellipse([x, y, x + head_size, y + head_size], 
                    fill='peachpuff', outline='black')
        
        # Body  
        draw.rectangle([x + head_size//4, y + head_size, 
                       x + 3*head_size//4, y + head_size + body_height],
                      fill='blue', outline='black')
    
    return img


def create_traffic_scene():
    """Create a traffic scene with cars and road signs"""
    img = Image.new('RGB', (640, 480), color=(135, 206, 235))  # Sky
    draw = ImageDraw.Draw(img)
    
    # Road
    draw.rectangle([0, 300, 640, 480], fill='darkgray')
    
    # Lane markings
    for x in range(0, 640, 80):
        draw.rectangle([x, 380, x + 40, 390], fill='white')
    
    # Cars at different positions and angles
    cars = [
        (100, 320, 'red'),
        (300, 340, 'blue'), 
        (480, 315, 'white'),
        (200, 420, 'yellow'),  # Oncoming traffic
    ]
    
    for x, y, color in cars:
        # Car body
        draw.rectangle([x, y, x + 80, y + 40], fill=color, outline='black')
        # Wheels
        draw.ellipse([x + 10, y + 35, x + 25, y + 50], fill='black')
        draw.ellipse([x + 65, y + 35, x + 80, y + 50], fill='black')
        # Windows
        draw.rectangle([x + 10, y + 5, x + 70, y + 15], fill='lightblue')
    
    # Traffic sign (stop sign)
    draw.regular_polygon((550, 150, 30), 8, fill='red', outline='white')
    draw.text((540, 145), "STOP", fill='white')
    
    return img


def create_text_document():
    """Create a document image for OCR testing"""
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to load a better font
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Document title
    draw.text((50, 50), "SAMPLE DOCUMENT", fill='black', font=font_large)
    draw.line([(50, 85), (550, 85)], fill='black', width=2)
    
    # Body text with different formatting
    text_lines = [
        "This is a sample document created for OCR testing purposes.",
        "",
        "It contains multiple lines of text with different formatting:",
        "• Bullet points like this one",
        "• Numbers: 123-456-7890",
        "• Special characters: @#$%^&*()",
        "• Mixed case: CamelCase and snake_case",
        "",
        "Date: January 15, 2024",
        "Reference: DOC-2024-001",
        "",
        "The quick brown fox jumps over the lazy dog.",
        "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS.",
        "",
        "Contact Information:",
        "Email: test@example.com",
        "Phone: +1 (555) 123-4567",
        "Website: https://www.example.com"
    ]
    
    y_pos = 120
    for line in text_lines:
        if line.startswith("•"):
            draw.text((70, y_pos), line, fill='black', font=font_small)
        elif line.isupper():
            draw.text((50, y_pos), line, fill='black', font=font_medium)
        else:
            draw.text((50, y_pos), line, fill='black', font=font_small)
        y_pos += 25
    
    return img


def create_receipt_image():
    """Create a receipt-like image for OCR testing"""
    img = Image.new('RGB', (400, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = font_small = font_bold = ImageFont.load_default()
    
    # Receipt header
    draw.text((150, 20), "QuickMart", fill='black', font=font_bold)
    draw.text((120, 45), "123 Main Street", fill='black', font=font_small)
    draw.text((110, 60), "Anytown, ST 12345", fill='black', font=font_small)
    draw.text((130, 75), "(555) 123-4567", fill='black', font=font_small)
    
    draw.line([(20, 100), (380, 100)], fill='black', width=1)
    
    # Receipt items
    y_pos = 120
    items = [
        ("Milk 2% 1 Gallon", "3.49"),
        ("Bread Whole Wheat", "2.99"),
        ("Eggs Large Dozen", "4.29"),
        ("Bananas 2.5 lbs", "1.87"),
        ("Coffee Ground 12oz", "8.99"),
        ("Tax", "1.08"),
    ]
    
    for item, price in items:
        draw.text((30, y_pos), item, fill='black', font=font_small)
        draw.text((320, y_pos), f"${price}", fill='black', font=font_small)
        y_pos += 20
    
    draw.line([(20, y_pos + 10), (380, y_pos + 10)], fill='black', width=1)
    
    # Total
    draw.text((250, y_pos + 25), "TOTAL: $21.71", fill='black', font=font_bold)
    
    # Footer
    draw.text((120, y_pos + 60), "Thank you for shopping!", fill='black', font=font_small)
    draw.text((80, y_pos + 80), f"Transaction ID: TXN-2024-0001", fill='black', font=font_small)
    
    return img


def create_signage_image():
    """Create a street sign image for OCR testing"""
    img = Image.new('RGB', (400, 300), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_medium = ImageFont.truetype("arial.ttf", 24)
    except:
        font_large = font_medium = ImageFont.load_default()
    
    # Stop sign
    draw.regular_polygon((150, 100, 60), 8, fill='red', outline='white')
    draw.text((125, 90), "STOP", fill='white', font=font_large)
    
    # Speed limit sign
    draw.rectangle([250, 50, 350, 150], fill='white', outline='black', width=3)
    draw.text((260, 60), "SPEED", fill='black', font=font_medium)
    draw.text((260, 85), "LIMIT", fill='black', font=font_medium)
    draw.text((285, 115), "25", fill='black', font=font_large)
    
    # Street name sign
    draw.rectangle([50, 200, 350, 250], fill='green', outline='white', width=2)
    draw.text((80, 215), "MAIN STREET", fill='white', font=font_large)
    
    return img


def create_face_image(pose="frontal"):
    """Create face images for recognition testing"""
    img = Image.new('RGB', (300, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    if pose == "frontal":
        # Centered face
        face_center = (150, 200)
        face_width, face_height = 120, 160
    else:  # profile
        face_center = (180, 200)
        face_width, face_height = 80, 160
    
    # Face outline
    face_box = [
        face_center[0] - face_width//2,
        face_center[1] - face_height//2,
        face_center[0] + face_width//2,
        face_center[1] + face_height//2
    ]
    draw.ellipse(face_box, fill='peachpuff', outline='brown', width=2)
    
    if pose == "frontal":
        # Eyes
        draw.ellipse([130, 170, 150, 190], fill='white', outline='black')
        draw.ellipse([170, 170, 190, 190], fill='white', outline='black')
        draw.ellipse([137, 177, 143, 183], fill='black')  # Pupils
        draw.ellipse([177, 177, 183, 183], fill='black')
        
        # Nose
        draw.polygon([(145, 200), (155, 200), (150, 220)], fill='brown')
        
        # Mouth
        draw.arc([135, 230, 185, 250], start=0, end=180, fill='red', width=3)
    
    # Hair
    draw.ellipse([
        face_center[0] - face_width//2 - 10,
        face_center[1] - face_height//2 - 30,
        face_center[0] + face_width//2 + 10,
        face_center[1] - face_height//4
    ], fill='brown', outline='#8B4513')
    
    return img


def create_group_photo():
    """Create a group photo with multiple faces"""
    img = Image.new('RGB', (600, 400), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Background
    draw.rectangle([0, 300, 600, 400], fill='green')  # Ground
    
    # Multiple people
    people_positions = [
        (100, 150, 'peachpuff'),
        (250, 140, 'wheat'),
        (400, 160, 'burlywood'),
        (500, 155, 'peachpuff')
    ]
    
    for x, y, skin_tone in people_positions:
        # Head
        draw.ellipse([x, y, x + 60, y + 80], fill=skin_tone, outline='brown', width=2)
        
        # Eyes
        draw.ellipse([x + 15, y + 25, x + 25, y + 35], fill='white', outline='black')
        draw.ellipse([x + 35, y + 25, x + 45, y + 35], fill='white', outline='black')
        draw.ellipse([x + 18, y + 28, x + 22, y + 32], fill='black')
        draw.ellipse([x + 38, y + 28, x + 42, y + 32], fill='black')
        
        # Nose and mouth
        draw.polygon([(x + 27, y + 40), (x + 33, y + 40), (x + 30, y + 50)], fill='brown')
        draw.arc([x + 20, y + 55, x + 40, y + 65], start=0, end=180, fill='red', width=2)
        
        # Body (simple rectangle)
        draw.rectangle([x + 10, y + 80, x + 50, y + 150], fill='blue', outline='black')
    
    return img


def create_blurry_image():
    """Create a blurry image for edge case testing"""
    img = create_cat_image()
    # Simulate blur by resizing down and up
    img = img.resize((56, 56), Image.LANCZOS)
    img = img.resize((224, 224), Image.LANCZOS)
    return img


def create_low_contrast_image():
    """Create a low contrast image for edge case testing"""
    img = create_car_image()
    # Convert to array and reduce contrast
    img_array = np.array(img)
    img_array = img_array.astype(np.float32)
    img_array = (img_array - 127.5) * 0.3 + 127.5  # Reduce contrast
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def create_noisy_image():
    """Create a noisy image for edge case testing"""
    img = create_dog_image()
    img_array = np.array(img)
    # Add random noise
    noise = np.random.randint(-30, 30, img_array.shape)
    img_array = img_array.astype(np.int16) + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def create_test_video():
    """Create a simple test video (placeholder - would need opencv for actual video)"""
    print("Video generation would require opencv-python for actual implementation")
    print("Creating video placeholder files...")
    
    videos_dir = Path(__file__).parent / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Create placeholder files
    (videos_dir / "README.txt").write_text(
        "This directory would contain test videos.\n"
        "For actual video generation, use opencv-python:\n"
        "import cv2\n"
        "# Create video frames and write with cv2.VideoWriter\n"
    )


if __name__ == "__main__":
    print("Generating AI Vision test data...")
    create_test_images()
    create_test_video()
    print("Test data generation complete!")
