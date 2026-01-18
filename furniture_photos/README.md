# 📸 Put Your Furniture Photos Here

## Instructions

1. **Take photos** of individual furniture pieces
   - Bed
   - Desk
   - Chair
   - Nightstand
   - etc.

2. **Crop each piece** (optional but recommended)
   - Use any image editor
   - Or use: `python3 ../scripts/crop_furniture.py your_photo.jpg`

3. **Save photos here** with clear names:
   - `bed.jpg`
   - `desk.jpg`
   - `chair.jpg`

4. **Process them:**
   ```bash
   ./process_to_3d.sh furniture_photos/bed.jpg
   ```

5. **Get 3D models** in `../processed_models/` folder!

---

## Tips

- ✅ Good lighting = better 3D models
- ✅ Centered furniture = better results
- ✅ One object per photo = best quality
- ✅ Clear background = easier processing

---

## Example Workflow

```bash
# 1. Put your bedroom photo here
cp ~/Desktop/bedroom.jpg furniture_photos/

# 2. Crop just the bed (optional)
python3 ../scripts/crop_furniture.py furniture_photos/bedroom.jpg

# 3. Process it
../process_to_3d.sh furniture_photos/bed_cropped.jpg

# 4. View result
# Upload ../processed_models/bed_cropped/bed_cropped_3d.obj to https://3dviewer.net/
```

