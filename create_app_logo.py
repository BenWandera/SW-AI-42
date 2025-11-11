from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_modern_logo(size=512):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    
    gradient_bg = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(gradient_bg)
    
    for i in range(size):
        ratio = i / size
        r = int(10 + (30 - 10) * ratio)
        g = int(40 + (100 - 40) * ratio)
        b = int(80 + (200 - 80) * ratio)
        bg_draw.rectangle([(0, i), (size, i+1)], fill=(r, g, b, 180))
    
    img = Image.alpha_composite(img, gradient_bg)
    draw = ImageDraw.Draw(img)
    
    outer_radius = size * 0.42
    inner_radius = size * 0.28
    
    blue_glow = (0, 150, 255, 200)
    cyan_glow = (0, 255, 255, 150)
    
    for offset in range(8, 0, -1):
        alpha = int(30 * (offset / 8))
        glow_color = (0, 150, 255, alpha)
        draw.ellipse(
            [(center - outer_radius - offset, center - outer_radius - offset),
             (center + outer_radius + offset, center + outer_radius + offset)],
            fill=None,
            outline=glow_color,
            width=3
        )
    
    draw.ellipse(
        [(center - outer_radius, center - outer_radius),
         (center + outer_radius, center + outer_radius)],
        fill=(15, 60, 120, 220),
        outline=(0, 200, 255, 255),
        width=4
    )
    
    draw.ellipse(
        [(center - inner_radius, center - inner_radius),
         (center + inner_radius, center + inner_radius)],
        fill=(5, 30, 80, 255),
        outline=(0, 180, 255, 255),
        width=3
    )
    
    arrow_size = size * 0.15
    arrow_thickness = size * 0.05
    arrow_offset = size * 0.35
    
    angles = [30, 150, 270]
    
    for angle in angles:
        rad = np.radians(angle)
        x = center + arrow_offset * np.cos(rad)
        y = center + arrow_offset * np.sin(rad)
        
        next_rad = np.radians(angle + 120)
        
        start_x = x - arrow_size * 0.4 * np.cos(next_rad)
        start_y = y - arrow_size * 0.4 * np.sin(next_rad)
        end_x = x + arrow_size * 0.6 * np.cos(next_rad)
        end_y = y + arrow_size * 0.6 * np.sin(next_rad)
        
        for glow_offset in range(6, 0, -1):
            glow_alpha = int(40 * (glow_offset / 6))
            glow_width = int(arrow_thickness + glow_offset * 2)
            draw.line(
                [(start_x, start_y), (end_x, end_y)],
                fill=(0, 200, 255, glow_alpha),
                width=glow_width
            )
        
        gradient_steps = 10
        for i in range(gradient_steps):
            ratio = i / gradient_steps
            line_x = start_x + (end_x - start_x) * ratio
            line_y = start_y + (end_y - start_y) * ratio
            next_line_x = start_x + (end_x - start_x) * (ratio + 1/gradient_steps)
            next_line_y = start_y + (end_y - start_y) * (ratio + 1/gradient_steps)
            
            r = int(0 + (0 - 0) * ratio)
            g = int(180 + (255 - 180) * ratio)
            b = int(255 + (255 - 255) * ratio)
            
            draw.line(
                [(line_x, line_y), (next_line_x, next_line_y)],
                fill=(r, g, b, 255),
                width=int(arrow_thickness)
            )
        
        tip_x = end_x
        tip_y = end_y
        
        perp_rad = next_rad + np.pi / 2
        arrow_head_size = arrow_size * 0.4
        
        left_x = tip_x - arrow_head_size * np.cos(next_rad) - arrow_head_size * 0.5 * np.cos(perp_rad)
        left_y = tip_y - arrow_head_size * np.sin(next_rad) - arrow_head_size * 0.5 * np.sin(perp_rad)
        right_x = tip_x - arrow_head_size * np.cos(next_rad) + arrow_head_size * 0.5 * np.cos(perp_rad)
        right_y = tip_y - arrow_head_size * np.sin(next_rad) + arrow_head_size * 0.5 * np.sin(perp_rad)
        
        arrow_head = [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]
        draw.polygon(arrow_head, fill=(0, 255, 255, 255), outline=(0, 200, 255, 255))
    
    center_dot_radius = size * 0.08
    for offset in range(6, 0, -1):
        alpha = int(50 * (offset / 6))
        draw.ellipse(
            [(center - center_dot_radius - offset, center - center_dot_radius - offset),
             (center + center_dot_radius + offset, center + center_dot_radius + offset)],
            fill=(0, 255, 255, alpha)
        )
    
    draw.ellipse(
        [(center - center_dot_radius, center - center_dot_radius),
         (center + center_dot_radius, center + center_dot_radius)],
        fill=(0, 255, 255, 255),
        outline=(255, 255, 255, 255),
        width=2
    )
    
    particle_count = 30
    for _ in range(particle_count):
        px = np.random.randint(0, size)
        py = np.random.randint(0, size)
        particle_size = np.random.randint(1, 3)
        alpha = np.random.randint(50, 150)
        draw.ellipse(
            [(px, py), (px + particle_size, py + particle_size)],
            fill=(0, 200, 255, alpha)
        )
    
    return img

def create_rounded_square_logo(size=512):
    img = create_modern_logo(size)
    
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    corner_radius = size // 8
    mask_draw.rounded_rectangle([(0, 0), (size, size)], radius=corner_radius, fill=255)
    
    output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    output.paste(img, (0, 0), mask)
    
    return output

print("Creating EcoWaste AI Logo...")

logo_512 = create_modern_logo(512)
logo_512.save('app_logo_512.png')
print("✅ Created: app_logo_512.png (512x512)")

logo_1024 = create_modern_logo(1024)
logo_1024.save('app_logo_1024.png')
print("✅ Created: app_logo_1024.png (1024x1024 - High Quality)")

logo_rounded = create_rounded_square_logo(512)
logo_rounded.save('app_logo_rounded.png')
print("✅ Created: app_logo_rounded.png (Rounded corners)")

icon_sizes = [
    (192, 'app_icon_192.png'),
    (128, 'app_icon_128.png'),
    (96, 'app_icon_96.png'),
    (72, 'app_icon_72.png'),
    (48, 'app_icon_48.png'),
]

for size, filename in icon_sizes:
    icon = create_modern_logo(size)
    icon.save(filename)
    print(f"✅ Created: {filename} ({size}x{size})")

android_sizes = [
    (192, 'android/res/mipmap-xxxhdpi/ic_launcher.png'),
    (144, 'android/res/mipmap-xxhdpi/ic_launcher.png'),
    (96, 'android/res/mipmap-xhdpi/ic_launcher.png'),
    (72, 'android/res/mipmap-hdpi/ic_launcher.png'),
    (48, 'android/res/mipmap-mdpi/ic_launcher.png'),
]

import os
for size, path in android_sizes:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    icon = create_rounded_square_logo(size)
    icon.save(path)
    print(f"✅ Created: {path} ({size}x{size})")

print("\n🎨 Logo creation complete!")
print("\nFiles created:")
print("  - app_logo_512.png (Main logo)")
print("  - app_logo_1024.png (High quality)")
print("  - app_logo_rounded.png (Rounded corners)")
print("  - app_icon_*.png (Various sizes)")
print("  - android/res/mipmap-*/ic_launcher.png (Android icons)")
