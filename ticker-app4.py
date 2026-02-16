import yt_dlp
import cv2
import os
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image
import json
import time
from threading import Lock
import numpy as np
from datetime import datetime
import easyocr
import shutil

# Configure Gemini API
GEMINI_API_KEY = "api-key"
genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Initialize EasyOCR reader (supports English and Urdu)
# Set gpu=True if you have CUDA-capable GPU for faster processing
easyocr_reader = easyocr.Reader(['en', 'ur'], gpu=False)

# Rate limiting
rate_limit_lock = Lock()
last_request_time = 0
MIN_REQUEST_INTERVAL = 1.5

# Processing settings
TARGET_WIDTH = 640
TARGET_HEIGHT = 360
JPEG_QUALITY = 85
CHANGE_DETECTION_THRESHOLD = 0.15

# Files
METADATA_DICT_FILE = "video_metadata_dict.json"

def load_metadata_dict():
    """Load the metadata dictionary from file"""
    if os.path.exists(METADATA_DICT_FILE):
        with open(METADATA_DICT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata_dict(metadata_dict):
    """Save the metadata dictionary to file"""
    with open(METADATA_DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

def get_video_metadata(video_path):
    """
    Extract metadata from video file
    Tries to get YouTube metadata if video has standard YouTube filename
    """
    print("\n" + "="*50)
    print("📊 EXTRACTING VIDEO METADATA")
    print("="*50)
    
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    
    # Try to get YouTube metadata if it looks like a YouTube ID (11 chars)
    if len(video_id) == 11:
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            ydl_opts = {'quiet': True, 'no_warnings': True}
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                upload_date = info.get('upload_date', '')
                if upload_date:
                    upload_datetime = datetime.strptime(upload_date, '%Y%m%d')
                    upload_month = upload_datetime.strftime('%B %Y')
                    upload_year = upload_datetime.year
                    upload_month_num = upload_datetime.month
                else:
                    upload_month = "Unknown"
                    upload_year = None
                    upload_month_num = None
                
                metadata = {
                    'video_id': info.get('id'),
                    'title': info.get('title'),
                    'channel': info.get('channel'),
                    'channel_id': info.get('channel_id'),
                    'upload_month': upload_month,
                    'upload_year': upload_year,
                    'upload_month_number': upload_month_num,
                    'duration_seconds': info.get('duration'),
                }
                
                print(f"✅ Video ID: {metadata['video_id']}")
                print(f"✅ Channel: {metadata['channel']}")
                print(f"✅ Channel ID: {metadata['channel_id']}")
                print(f"✅ Upload Month: {metadata['upload_month']}")
                
                return metadata
                
        except Exception as e:
            print(f"⚠️  Could not fetch YouTube metadata: {str(e)[:100]}")
    
    # Fallback: basic metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    metadata = {
        'video_id': video_id,
        'title': video_filename,
        'channel': 'Unknown',
        'channel_id': 'Unknown',
        'upload_month': 'Unknown',
        'upload_year': None,
        'upload_month_number': None,
        'duration_seconds': int(duration),
    }
    
    print(f"⚠️  Using basic metadata for: {video_filename}")
    return metadata

def check_metadata_match(video_metadata, metadata_dict):
    """
    Check if video's channel_id and upload_month match any entry in dictionary
    
    Returns:
        tuple: (match_found, matching_key, bbox_data, missing_overlays)
        - missing_overlays: list of overlay types that need detection (e.g., ['ticker', 'lower_third', 'headline'])
    """
    channel_id = video_metadata.get('channel_id')
    upload_month = video_metadata.get('upload_month')
    
    if not channel_id or not upload_month or channel_id == 'Unknown':
        return False, None, None, ['ticker', 'lower_third', 'headline']
    
    # Search for matching channel_id and upload_month
    for key, data in metadata_dict.items():
        if (data.get('channel_id') == channel_id and 
            data.get('upload_month') == upload_month):
            
            print(f"\n✅ MATCH FOUND!")
            print(f"   Key: {key}")
            print(f"   Channel ID: {channel_id}")
            print(f"   Upload Month: {upload_month}")
            
            bbox_data = data.get('bounding_boxes', {})
            
            # Check which overlays are missing
            missing_overlays = []
            for overlay_type in ['ticker', 'lower_third', 'headline']:
                overlay_bbox = bbox_data.get(overlay_type)
                # Consider it missing if it's None or if it's a dict with null x coordinate
                if overlay_bbox is None or (isinstance(overlay_bbox, dict) and overlay_bbox.get('x') is None):
                    missing_overlays.append(overlay_type)
            
            if missing_overlays:
                print(f"⚠️  Missing bounding boxes for: {', '.join(missing_overlays)}")
            
            return True, key, bbox_data, missing_overlays
    
    return False, None, None, ['ticker', 'lower_third', 'headline']

def extract_sample_frames(video_path, num_samples=5):
    """Extract sample frames for bbox detection"""
    print(f"\n🎬 Extracting {num_samples} sample frames for bbox detection...")
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    output_folder = "temp_frames"
    os.makedirs(output_folder, exist_ok=True)
    
    sample_frames = []
    
    for idx, frame_number in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            timestamp = frame_number / video_fps
            frame_filename = f"{output_folder}/sample_{idx:02d}.jpg"
            cv2.imwrite(frame_filename, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            sample_frames.append({'path': frame_filename, 'timestamp': timestamp})
    
    cap.release()
    print(f"✅ Extracted {len(sample_frames)} frames")
    return sample_frames

def detect_bounding_boxes_with_api(sample_frames, detect_overlays=['ticker', 'lower_third', 'headline']):
    """
    Use Gemini API to detect bounding boxes
    
    Args:
        sample_frames: List of sample frame paths
        detect_overlays: List of overlay types to detect (e.g., ['ticker'], ['headline'], or ['ticker', 'lower_third', 'headline'])
    """
    global last_request_time
    
    print(f"\n🔍 Detecting bounding boxes for: {', '.join(detect_overlays)}")
    
    # Rate limiting
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        last_request_time = time.time()
    
    try:
        images = [Image.open(frame['path']) for frame in sample_frames]
        img_width, img_height = images[0].size
        
        # Build overlay descriptions based on what we're detecting
        overlay_descriptions = []
        if 'ticker' in detect_overlays:
            overlay_descriptions.append("1. TICKER: Scrolling news text at the VERY BOTTOM of the screen (thin bar, usually 20-40px height)")
        if 'lower_third' in detect_overlays:
            overlay_descriptions.append("2. LOWER_THIRD: The persistent graphics bar/box above the ticker (contains channel logo, breaking news banner, etc. - usually 30-50px height)")
        if 'headline' in detect_overlays:
            overlay_descriptions.append("3. HEADLINE: Main news headline text overlay (can be within the lower-third or separate - look for the largest, most prominent text)")
        
        # Build JSON structure description
        json_fields = []
        for overlay in detect_overlays:
            json_fields.append(f'  "{overlay}": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>, "confidence": <0-100>, "description": "<text>"}}')
        
        prompt = f"""
I'm showing you {len(images)} frames from a news broadcast video.

Analyze ALL the frames and identify the SINGLE, CONSISTENT bounding box location for these text overlays:
{chr(10).join(overlay_descriptions)}

Image dimensions: {img_width}x{img_height}

IMPORTANT GUIDELINES:
- Return ONE bounding box for each overlay type, not per-frame
- The boxes should work for ALL frames
- TICKER is at the very bottom (scrolling text)
- LOWER_THIRD is the bar/graphic above ticker (DO NOT INCLUDE LOGOS - only text areas)
- HEADLINE is the main news text
- These are separate elements that may appear independently

Return ONLY this JSON structure (no markdown, no code blocks, no extra text):
{{
{','.join(json_fields)}
}}

If an overlay is not visible in ANY frame, set all its values to null.
"""
        
        content = [prompt] + images
        response = gemini_model.generate_content(content)
        
        if response and hasattr(response, 'text'):
            response_text = response.text.strip()
            
            # Remove markdown code blocks more aggressively
            if '```' in response_text:
                # Extract content between first ``` and last ```
                start = response_text.find('```')
                end = response_text.rfind('```')
                if start != -1 and end != -1 and start < end:
                    response_text = response_text[start+3:end].strip()
                    # Remove 'json' keyword if present at start
                    if response_text.startswith('json'):
                        response_text = response_text[4:].strip()
            
            # Find the JSON object/array
            # Look for opening brace or bracket
            json_start = -1
            for char in ['{', '[']:
                idx = response_text.find(char)
                if idx != -1 and (json_start == -1 or idx < json_start):
                    json_start = idx
            
            if json_start != -1:
                response_text = response_text[json_start:]
            
            # Parse JSON
            bboxes_data = json.loads(response_text)
            
            # Handle if response is an array (take first element which represents all frames)
            if isinstance(bboxes_data, list):
                if len(bboxes_data) > 0:
                    print(f"📝 API returned array of {len(bboxes_data)} items, using first one")
                    bboxes_data = bboxes_data[0]
                else:
                    print(f"❌ API returned empty array")
                    return None
            
            # Validate that it's a dict
            if not isinstance(bboxes_data, dict):
                print(f"❌ Expected dict, got {type(bboxes_data)}")
                print(f"Response: {str(bboxes_data)[:500]}")
                return None
            
            # Validate that requested overlays are present
            for overlay in detect_overlays:
                if overlay not in bboxes_data:
                    print(f"❌ Missing {overlay} key in response")
                    print(f"Response: {response_text[:500]}")
                    return None
            
            # Display detected bboxes
            for overlay_type in detect_overlays:
                if overlay_type in bboxes_data:
                    bbox = bboxes_data[overlay_type]
                    if isinstance(bbox, dict) and bbox.get('x') is not None:
                        print(f"✅ {overlay_type.upper()}: ({bbox['x']}, {bbox['y']}) - {bbox['width']}x{bbox['height']}")
                    else:
                        print(f"⚠️  {overlay_type.upper()}: Not detected")
            
            return bboxes_data
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {str(e)}")
        print(f"Response text: {response_text[:500]}")
        return None
    except Exception as e:
        print(f"❌ Bbox detection failed: {str(e)[:200]}")
        return None

def extract_all_frames(video_path, fps=0.5):
    """Extract all frames at specified FPS"""
    print(f"\n🎬 Extracting frames at {fps} FPS...")
    
    frames_folder = "frames"
    os.makedirs(frames_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = max(1, int(video_fps / fps))
    
    frames_saved = []
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                timestamp = frame_count / video_fps
                frame_filename = f"{frames_folder}/frame_{saved_count:05d}.jpg"
                cv2.imwrite(frame_filename, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                
                frames_saved.append({
                    'path': frame_filename,
                    'timestamp': timestamp,
                    'frame_number': frame_count,
                    'index': saved_count
                })
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"✅ Extracted {saved_count} frames")
    return frames_saved

def crop_region(frame_path, bbox):
    """Crop region from frame using bounding box"""
    img = cv2.imread(frame_path)
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    cropped = img[y:y+h, x:x+w]
    return cropped

def is_frame_different(img1, img2, threshold=0.15):
    """Check if two images are significantly different using grayscale"""
    if img1 is None or img2 is None:
        return True
    
    if img1.shape != img2.shape:
        return True
    
    # Convert to grayscale for comparison (like chyron_extractor's B&W method)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray1, gray2)
    diff_ratio = np.sum(diff > 30) / diff.size
    
    return diff_ratio > threshold

def process_overlay_frames(frames, bbox, overlay_type, output_folder):
    """Process frames for a specific overlay type (ticker, lower_third, or headline)"""
    print(f"\n📦 Processing {overlay_type} frames...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    unique_frames = []
    previous_img = None
    
    for frame in tqdm(frames, desc=f"Processing {overlay_type}"):
        cropped_img = crop_region(frame['path'], bbox)
        
        if previous_img is None or is_frame_different(previous_img, cropped_img, CHANGE_DETECTION_THRESHOLD):
            unique_id = len(unique_frames)
            filename = f"{output_folder}/{overlay_type}_{unique_id:04d}_t{frame['timestamp']:.2f}s.jpg"
            
            cv2.imwrite(filename, cropped_img)
            
            unique_frames.append({
                'id': unique_id,
                'timestamp': frame['timestamp'],
                'image_path': filename,
            })
            
            previous_img = cropped_img.copy()
    
    print(f"✅ Found {len(unique_frames)} unique {overlay_type} frames")
    return unique_frames

def extract_text_easyocr(unique_frames, overlay_type):
    """
    Extract text from unique frames using EasyOCR (local processing, no API calls)
    
    Args:
        unique_frames: List of unique frame dicts with image_path
        overlay_type: 'ticker', 'lower_third', or 'headline'
    
    Returns:
        List of frames with extracted text added
    """
    if not unique_frames:
        return unique_frames
    
    print(f"\n📝 Extracting text from {len(unique_frames)} {overlay_type} frames using EasyOCR...")
    
    frames_with_text = []
    
    for frame_info in tqdm(unique_frames, desc=f"OCR {overlay_type}"):
        try:
            # Read image
            img = cv2.imread(frame_info['image_path'])
            
            # Use EasyOCR to extract text
            results = easyocr_reader.readtext(img)
            
            # Combine all detected text
            if results:
                # Extract text from results (each result is [bbox, text, confidence])
                texts = [text for (bbox, text, conf) in results]
                extracted_text = ' '.join(texts)
            else:
                extracted_text = "NO_TEXT"
            
            # Add to frame info
            frame_with_text = frame_info.copy()
            frame_with_text['extracted_text'] = extracted_text
            frames_with_text.append(frame_with_text)
            
        except Exception as e:
            # Handle errors gracefully
            frame_with_text = frame_info.copy()
            frame_with_text['extracted_text'] = f"ERROR: {str(e)[:100]}"
            frames_with_text.append(frame_with_text)
    
    print(f"✅ Text extraction complete for {overlay_type}")
    return frames_with_text

def process_with_known_bbox(video_path, bboxes):
    """
    Process video using known bounding boxes
    - Uses OpenCV for frame extraction and deduplication (NO API)
    - Uses EasyOCR for text extraction (local processing, no API calls)
    """
    print("\n" + "="*50)
    print("🚀 PROCESSING WITH KNOWN BOUNDING BOXES")
    print("="*50)
    print("✨ Visual processing: OpenCV only (no API)")
    print("📝 Text extraction: EasyOCR (local, no API cost)")
    
    # Validate bboxes structure
    if not isinstance(bboxes, dict):
        print(f"❌ Invalid bboxes type: {type(bboxes)}")
        return None
    
    # Extract all frames
    frames = extract_all_frames(video_path, fps=0.5)
    
    results = {}
    
    # Process ticker if bbox exists
    if bboxes.get('ticker') and isinstance(bboxes['ticker'], dict) and bboxes['ticker'].get('x') is not None:
        ticker_frames = process_overlay_frames(
            frames, 
            bboxes['ticker'], 
            'ticker',
            'unique_ticker_frames'
        )
        
        # Extract text from ticker frames using EasyOCR
        ticker_frames_with_text = extract_text_easyocr(
            ticker_frames,
            'ticker'
        )
        
        results['ticker'] = {
            'unique_frames_count': len(ticker_frames_with_text),
            'unique_frames': ticker_frames_with_text
        }
    
    # Process lower_third if bbox exists
    if bboxes.get('lower_third') and isinstance(bboxes['lower_third'], dict) and bboxes['lower_third'].get('x') is not None:
        lower_third_frames = process_overlay_frames(
            frames,
            bboxes['lower_third'],
            'lower_third',
            'unique_lower_third_frames'
        )
        
        # Extract text from lower_third frames using EasyOCR
        lower_third_frames_with_text = extract_text_easyocr(
            lower_third_frames,
            'lower_third'
        )
        
        results['lower_third'] = {
            'unique_frames_count': len(lower_third_frames_with_text),
            'unique_frames': lower_third_frames_with_text
        }
    
    # Process headline if bbox exists
    if bboxes.get('headline') and isinstance(bboxes['headline'], dict) and bboxes['headline'].get('x') is not None:
        headline_frames = process_overlay_frames(
            frames,
            bboxes['headline'],
            'headline',
            'unique_headline_frames'
        )
        
        # Extract text from headline frames using EasyOCR
        headline_frames_with_text = extract_text_easyocr(
            headline_frames,
            'headline'
        )
        
        results['headline'] = {
            'unique_frames_count': len(headline_frames_with_text),
            'unique_frames': headline_frames_with_text
        }
    
    return results

def process_with_api_detection(video_path, video_metadata, existing_bboxes=None, missing_overlays=None):
    """
    Process video by detecting missing bounding boxes with API
    Then merge with existing bboxes and process normally
    
    Args:
        video_path: Path to video file
        video_metadata: Video metadata dict
        existing_bboxes: Existing bounding boxes dict (if any)
        missing_overlays: List of overlay types to detect (e.g., ['ticker', 'lower_third', 'headline'])
    """
    if missing_overlays is None:
        missing_overlays = ['ticker', 'lower_third', 'headline']
    
    print("\n" + "="*50)
    print(f"🔍 DETECTING MISSING BOUNDING BOXES: {', '.join(missing_overlays)}")
    print("="*50)
    
    # Extract sample frames for bbox detection
    sample_frames = extract_sample_frames(video_path, num_samples=5)
    
    try:
        # Detect only the missing bounding boxes with API
        new_bboxes = detect_bounding_boxes_with_api(sample_frames, missing_overlays)
        
        if not new_bboxes:
            print("❌ Failed to detect bounding boxes")
            return None
    finally:
        # Clean up temp_frames folder
        if os.path.exists('temp_frames'):
            shutil.rmtree('temp_frames')
            print("🗑️  Cleaned up temp_frames/")
    
    # Merge with existing bboxes
    if existing_bboxes:
        merged_bboxes = existing_bboxes.copy()
        merged_bboxes.update(new_bboxes)
        print(f"\n✅ Merged new detections with existing bboxes")
    else:
        merged_bboxes = new_bboxes
    
    # Update metadata dictionary
    metadata_dict = load_metadata_dict()
    
    dict_key = f"{video_metadata['channel_id']}_{video_metadata['upload_month'].replace(' ', '_')}"
    
    # If entry exists, update it; otherwise create new
    if dict_key in metadata_dict:
        metadata_dict[dict_key]['bounding_boxes'] = merged_bboxes
        metadata_dict[dict_key]['updated_at'] = datetime.now().isoformat()
        print(f"\n✅ Updated dictionary entry: {dict_key}")
    else:
        metadata_dict[dict_key] = {
            'channel_id': video_metadata['channel_id'],
            'channel': video_metadata['channel'],
            'upload_month': video_metadata['upload_month'],
            'upload_year': video_metadata['upload_year'],
            'bounding_boxes': merged_bboxes,
            'added_at': datetime.now().isoformat()
        }
        print(f"\n✅ Added to dictionary: {dict_key}")
    
    save_metadata_dict(metadata_dict)
    
    # Now process with the complete bboxes
    results = process_with_known_bbox(video_path, merged_bboxes)
    
    return results

def main(video_path):
    """
    Main processing function
    
    Logic:
    1. Check if video file exists
    2. Extract video metadata
    3. Check if channel_id + upload_month match dictionary
    4. Check which overlays are missing (ticker, lower_third, headline)
    5. If all overlays exist: process with known bboxes (NO API)
    6. If some/all overlays missing: detect only missing ones, merge with existing, then process
    """
    print("\n" + "🎥"*25)
    print("SMART VIDEO PROCESSOR")
    print("🎥"*25)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    print(f"✅ Video found: {video_path}")
    
    # Get video metadata
    video_metadata = get_video_metadata(video_path)
    
    # Load metadata dictionary
    metadata_dict = load_metadata_dict()
    print(f"\n📚 Loaded dictionary with {len(metadata_dict)} entries")
    
    # Check for match and missing overlays
    match_found, match_key, bbox_data, missing_overlays = check_metadata_match(video_metadata, metadata_dict)
    
    if match_found and not missing_overlays:
        # All overlays exist - no API needed
        print("\n✨ Using cached bounding boxes - NO API CALL NEEDED!")
        results = process_with_known_bbox(video_path, bbox_data)
    elif match_found and missing_overlays:
        # Some overlays exist, need to detect missing ones
        print(f"\n🔄 Found partial cache - detecting missing: {', '.join(missing_overlays)}")
        results = process_with_api_detection(video_path, video_metadata, bbox_data, missing_overlays)
    else:
        # No match - detect all overlays
        print("\n🆕 New video configuration - detecting all bounding boxes")
        results = process_with_api_detection(video_path, video_metadata, None, ['ticker', 'lower_third', 'headline'])
    
    # Save final results
    if results:
        output_file = f"results_{video_metadata['video_id']}.json"
        output_data = {
            'video_metadata': video_metadata,
            'processing_results': results,
            'processed_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Clean up all temporary image folders (data is already in JSON)
        folders_to_cleanup = ['frames', 'unique_ticker_frames', 'unique_lower_third_frames', 'unique_headline_frames']
        for folder in folders_to_cleanup:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        print("🗑️  Cleaned up temporary image folders")
        
        print("\n" + "="*50)
        print("✅ PROCESSING COMPLETE")
        print("="*50)
        print(f"📄 Results saved to: {output_file}")
        
        # Display summary
        if 'ticker' in results:
            print(f"\n📰 TICKER:")
            print(f"   Unique frames: {results['ticker']['unique_frames_count']}")
        
        if 'lower_third' in results:
            print(f"\n📊 LOWER THIRD:")
            print(f"   Unique frames: {results['lower_third']['unique_frames_count']}")
        
        if 'headline' in results:
            print(f"\n📺 HEADLINE:")
            print(f"   Unique frames: {results['headline']['unique_frames_count']}")
    
    return results

if __name__ == "__main__":
    # Process a video file in the current directory
    video_path = "downloads/1K5JRL2wS8c.mp4"  # Change this to your video filename
    
    print("\n💡 HOW THIS WORKS:")
    print("="*50)
    print("1. Checks video metadata (channel_id + upload_month)")
    print("2. Checks which overlays already detected (ticker/lower_third/headline)")
    print("3. If all cached → Skip bbox detection entirely")
    print("4. If partially cached → Detect only missing overlays")
    print("5. If nothing cached → Detect all three overlay types")
    print("6. Extracts frames using OpenCV (free)")
    print("7. Extracts text using EasyOCR (local, free)")
    print()
    print("💰 COST BREAKDOWN:")
    print("- Bbox detection (first time only): ~$0.00015 per channel/month")
    print("- Text extraction: $0.00 (EasyOCR is local and free!)")
    print("- Total per video after first: $0.00 (completely free!)")
    print()
    print("📊 EXAMPLE SCENARIOS:")
    print("- Video 1 (lower_third only): Detects lower_third → saves")
    print("- Video 2 (ticker + headline): Detects both → merges with cached lower_third")
    print("- Video 3 (all three): Uses all cached → NO detection needed!")
    print("="*50)
    print()
    
    results = main(video_path)
