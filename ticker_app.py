import yt_dlp
import cv2
import os
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image
import json
from difflib import SequenceMatcher
import time
from threading import Lock

# Configure Gemini API
GEMINI_API_KEY = "your-api-key"
genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Rate limiting
rate_limit_lock = Lock()
last_request_time = 0
MIN_REQUEST_INTERVAL = 1.5

# ULTRA-EFFICIENT BATCH PROCESSING SETTINGS
FRAMES_PER_REQUEST = 30   # Send 30 frames in 1 API request
REQUESTS_PER_BATCH = 10   # 10 requests per batch (300 frames)
TARGET_WIDTH = 640        # Reduce resolution for ticker
TARGET_HEIGHT = 360
JPEG_QUALITY = 85

RESUME_FILE = "processing_progress.json"

def load_progress():
    """Load processing progress from file"""
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'last_processed_index': -1, 'results': []}

def save_progress(last_index, results):
    """Save processing progress"""
    with open(RESUME_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'last_processed_index': last_index,
            'results': results
        }, f, indent=2, ensure_ascii=False)

def download_video(video_url, output_path="downloads"):
    """Download YouTube video"""
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info['id']
        filename = f"{output_path}/{video_id}.mp4"
        
    print(f"Downloaded: {filename}")
    return filename, info

def extract_frames(video_path, fps=1, output_folder="frames", target_width=640, target_height=360):
    """Extract frames from video at specified FPS with resolution reduction"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video FPS: {video_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Target resolution: {target_width}x{target_height}")
    
    frame_interval = max(1, int(video_fps / fps))
    print(f"Extracting at {fps} FPS (every {frame_interval} frames)")
    
    frames_saved = []
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, (target_width, target_height))
                timestamp = frame_count / video_fps
                frame_filename = f"{output_folder}/frame_{saved_count:05d}_t{timestamp:.2f}s.jpg"
                cv2.imwrite(frame_filename, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                
                frames_saved.append({
                    'path': frame_filename,
                    'timestamp': timestamp,
                    'frame_number': frame_count
                })
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {saved_count} frames at {fps} FPS (resized to {target_width}x{target_height})")
    
    if frames_saved:
        sample_size = os.path.getsize(frames_saved[0]['path']) / 1024
        total_size = sample_size * len(frames_saved) / 1024
        print(f"Average frame size: ~{sample_size:.1f}KB")
        print(f"Total frames size: ~{total_size:.1f}MB")
    
    return frames_saved

def extract_ticker_text_batch_gemini(frame_batch, max_retries=3):
    """Extract ticker text from multiple frames in ONE API request"""
    global last_request_time
    
    for attempt in range(max_retries):
        try:
            # Rate limiting
            with rate_limit_lock:
                current_time = time.time()
                time_since_last = current_time - last_request_time
                if time_since_last < MIN_REQUEST_INTERVAL:
                    time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
                last_request_time = time.time()
            
            # Load all images
            images = [Image.open(frame['path']) for frame in frame_batch]
            
            # Create optimized prompt
            prompt = f"""
            I'm showing you {len(images)} sequential frames from a news broadcast video.
            
            For EACH frame, extract ONLY the ticker text at the bottom of the screen.
            
            Return ONLY a valid JSON object with this structure:
            {{
              "frame_0": "ticker text or NO_TICKER",
              "frame_1": "ticker text or NO_TICKER",
              ...
              "frame_{len(images)-1}": "ticker text or NO_TICKER"
            }}
            
            Rules:
            - Extract ONLY ticker text (scrolling text at bottom)
            - If no ticker visible in a frame, use "NO_TICKER"
            - Preserve exact Urdu text
            - Return ONLY the JSON object, no markdown, no explanation
            - Be concise - these are consecutive frames, ticker may repeat
            """
            
            # Send all images in one request
            content = [prompt] + images
            response = gemini_model.generate_content(content)
            
            if response and hasattr(response, 'text'):
                response_text = response.text.strip()
                
                # Remove markdown code blocks
                if '```' in response_text:
                    parts = response_text.split('```')
                    for part in parts:
                        part = part.strip()
                        if part.startswith('json'):
                            part = part[4:].strip()
                        if part.startswith('{') and part.endswith('}'):
                            response_text = part
                            break
                
                ticker_data = json.loads(response_text)
                
                # Build results
                results = []
                for i, frame in enumerate(frame_batch):
                    ticker_text = (
                        ticker_data.get(f'frame_{i}') or 
                        ticker_data.get(str(i)) or 
                        ticker_data.get(f'Frame {i}') or
                        'NO_TICKER'
                    )
                    results.append({
                        'frame_number': frame['frame_number'],
                        'timestamp': frame['timestamp'],
                        'frame_path': frame['path'],
                        'ticker_text': ticker_text
                    })
                
                return results
            else:
                return [{
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'frame_path': frame['path'],
                    'ticker_text': 'NO_TICKER'
                } for frame in frame_batch]
                
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parsing error: {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return [{
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'frame_path': frame['path'],
                    'ticker_text': f'ERROR: JSON_PARSE_ERROR'
                } for frame in frame_batch]
                
        except Exception as e:
            error_msg = str(e)
            
            if "quota" in error_msg.lower() or "429" in error_msg:
                print(f"\n⚠️  API QUOTA EXCEEDED")
                return [{
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'frame_path': frame['path'],
                    'ticker_text': 'ERROR: QUOTA_EXCEEDED'
                } for frame in frame_batch]
            
            if "payload" in error_msg.lower() or "too large" in error_msg.lower():
                print(f"\n⚠️  PAYLOAD TOO LARGE")
                return [{
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'frame_path': frame['path'],
                    'ticker_text': 'ERROR: PAYLOAD_TOO_LARGE'
                } for frame in frame_batch]
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return [{
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'frame_path': frame['path'],
                    'ticker_text': f'ERROR: {error_msg[:100]}'
                } for frame in frame_batch]
    
    return [{
        'frame_number': frame['frame_number'],
        'timestamp': frame['timestamp'],
        'frame_path': frame['path'],
        'ticker_text': 'ERROR: Max retries exceeded'
    } for frame in frame_batch]

def deduplicate_tickers(results, similarity_threshold=0.85):
    """Remove duplicate/similar consecutive ticker texts"""
    if not results:
        return []
    
    deduplicated = [results[0]] if results[0]['ticker_text'] not in ('NO_TICKER', 'ERROR') else []
    
    for current in results[1:]:
        if current['ticker_text'].startswith('NO_TICKER') or current['ticker_text'].startswith('ERROR'):
            continue
        
        if deduplicated:
            previous = deduplicated[-1]
            similarity = SequenceMatcher(None, previous['ticker_text'], current['ticker_text']).ratio()
            
            if similarity < similarity_threshold:
                deduplicated.append(current)
        else:
            deduplicated.append(current)
    
    print(f"Deduplication: {len(results)} -> {len(deduplicated)} unique tickers")
    return deduplicated

def finalize_results(all_results, video_info, similarity_threshold=0.85):
    """Generate final deduplicated results and save to JSON"""
    print("\n" + "=" * 50)
    print("📊 GENERATING FINAL RESULTS")
    print("=" * 50)
    
    deduplicated = deduplicate_tickers(all_results, similarity_threshold)
    
    output_data = {
        'video_info': {
            'title': video_info.get('title', 'Unknown'),
            'channel': video_info.get('channel', 'Unknown'),
            'duration': video_info.get('duration'),
        },
        'deduplicated_tickers': deduplicated,
        'all_tickers': all_results,
        'total_processed': len(all_results),
        'unique_count': len(deduplicated)
    }
    
    with open('ticker_results_final.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
    
    print(f"✅ Total frames processed: {len(all_results)}")
    print(f"✅ Unique tickers found: {len(deduplicated)}")
    print(f"✅ Results saved to: ticker_results_final.json")
    
    return deduplicated

def process_video_auto_loop(video_url, fps=0.5, frames_per_request=30, requests_per_batch=10, similarity_threshold=0.85):
    """
    Auto-looping batch processor: Runs until complete or quota exceeded
    
    Key improvements:
    - Automatically loops through all batches
    - Handles quota gracefully mid-batch
    - Generates final JSON when complete
    - No need to manually rerun script
    """
    
    # Load previous progress
    progress = load_progress()
    all_results = progress['results']
    
    # Step 1: Download video
    print("=" * 50)
    print("STEP 1: Checking video")
    print("=" * 50)
    
    video_id = video_url.split('v=')[1].split('&')[0]
    video_path = f"downloads/{video_id}.mp4"
    
    if not os.path.exists(video_path):
        video_path, video_info = download_video(video_url)
    else:
        print(f"Video already downloaded: {video_path}")
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            video_info = ydl.extract_info(video_url, download=False)
    
    # Step 2: Extract frames
    print("\n" + "=" * 50)
    print("STEP 2: Checking frames")
    print("=" * 50)
    
    frames_folder = "frames"
    
    if not os.path.exists(frames_folder):
        frames = extract_frames(video_path, fps=fps, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT)
    else:
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith('frame_')])
        
        if not frame_files:
            frames = extract_frames(video_path, fps=fps, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT)
        else:
            print(f"Frames already extracted: {len(frame_files)} frames")
            sample_frame = Image.open(os.path.join(frames_folder, frame_files[0]))
            if sample_frame.width != TARGET_WIDTH or sample_frame.height != TARGET_HEIGHT:
                print(f"⚠️  Re-extracting at {TARGET_WIDTH}x{TARGET_HEIGHT}...")
                frames = extract_frames(video_path, fps=fps, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT)
            else:
                print(f"✅ Frames already optimized ({TARGET_WIDTH}x{TARGET_HEIGHT})")
                frames = []
                for f in frame_files:
                    frame_num = int(f.split('_')[1])
                    timestamp = float(f.split('_')[2].replace('t', '').replace('s.jpg', ''))
                    frames.append({
                        'path': os.path.join(frames_folder, f),
                        'timestamp': timestamp,
                        'frame_number': frame_num * 50
                    })
    
    # Step 3: AUTO-LOOP through batches
    print("\n" + "=" * 50)
    print("🔄 STARTING AUTO-LOOP PROCESSING")
    print("=" * 50)
    print(f"Total frames: {len(frames)}")
    print(f"Frames per request: {frames_per_request}")
    print(f"Requests per batch: {requests_per_batch}")
    print(f"Processing will continue automatically until complete or quota exceeded")
    print("=" * 50)
    
    last_processed = progress['last_processed_index']
    batch_number = 0
    quota_exceeded = False
    
    while last_processed + 1 < len(frames):
        batch_number += 1
        start_index = last_processed + 1
        frames_to_process = min(requests_per_batch * frames_per_request, len(frames) - start_index)
        end_index = start_index + frames_to_process
        
        print(f"\n{'='*50}")
        print(f"📦 BATCH {batch_number}")
        print(f"{'='*50}")
        print(f"Processing frames {start_index}-{end_index-1}")
        print(f"Progress: {start_index}/{len(frames)} ({100*start_index/len(frames):.1f}%)")
        
        frames_to_process_list = frames[start_index:end_index]
        
        # Group frames for API requests
        frame_groups = []
        for i in range(0, len(frames_to_process_list), frames_per_request):
            frame_groups.append(frames_to_process_list[i:i+frames_per_request])
        
        print(f"API requests in this batch: {len(frame_groups)}")
        
        # Process each group
        for group_idx, frame_group in enumerate(tqdm(frame_groups, desc=f"Batch {batch_number}")):
            results = extract_ticker_text_batch_gemini(frame_group)
            
            # Check each result for errors
            for result_idx, result in enumerate(results):
                # Check for quota exceeded BEFORE adding to results
                if "QUOTA_EXCEEDED" in result['ticker_text']:
                    # Calculate the index of the LAST SUCCESSFUL frame
                    current_index = start_index + (group_idx * frames_per_request) + result_idx - 1
                    
                    # Don't add failed frames to results - they'll be retried
                    save_progress(current_index, all_results)
                    
                    print(f"\n{'='*50}")
                    print(f"⏸️  QUOTA EXCEEDED - PAUSING")
                    print(f"{'='*50}")
                    print(f"✅ Processed: {current_index + 1}/{len(frames)} frames ({100*(current_index+1)/len(frames):.1f}%)")
                    print(f"❌ Frame {current_index + 2} failed (will retry on next run)")
                    print(f"✅ Progress saved successfully")
                    print(f"\n⏰ Wait ~1 hour for quota to reset, then run this script again")
                    print(f"   Script will automatically resume from frame {current_index + 1}")
                    
                    quota_exceeded = True
                    break
                
                # Only add successful results
                all_results.append(result)
                
                # Check for payload too large
                if "PAYLOAD_TOO_LARGE" in result['ticker_text']:
                    print(f"\n❌ PAYLOAD TOO LARGE")
                    print(f"   Current: FRAMES_PER_REQUEST = {frames_per_request}")
                    print(f"   Try: FRAMES_PER_REQUEST = {frames_per_request // 2}")
                    current_index = start_index + (group_idx * frames_per_request) + result_idx - 1
                    save_progress(current_index, all_results)
                    return None
            
            if quota_exceeded:
                break
        
        if quota_exceeded:
            break
        
        # Save progress after each batch
        last_processed = end_index - 1
        save_progress(last_processed, all_results)
        
        print(f"✅ Batch {batch_number} complete")
        print(f"✅ Remaining: {len(frames) - end_index} frames")
    
    # Check if we completed or hit quota
    if not quota_exceeded and last_processed + 1 >= len(frames):
        # ALL FRAMES PROCESSED - Generate final results
        print(f"\n{'='*50}")
        print(f"🎉 ALL FRAMES PROCESSED!")
        print(f"{'='*50}")
        
        deduplicated = finalize_results(all_results, video_info, similarity_threshold)
        
        # Show sample results
        print(f"\n📋 Sample unique tickers:")
        for i, result in enumerate(deduplicated[:5], 1):
            print(f"\n[{i}] Timestamp: {result['timestamp']:.2f}s")
            print(f"    Text: {result['ticker_text'][:100]}...")
        
        if len(deduplicated) > 5:
            print(f"\n... and {len(deduplicated) - 5} more unique tickers")
        
        return deduplicated
    else:
        # Quota exceeded - partial completion
        return None

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=1K5JRL2wS8c"
    
    print("\n" + "⚡" * 25)
    print("AUTO-LOOP BATCH PROCESSING MODE")
    print("⚡" * 25)
    print(f"Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT} (optimized)")
    print(f"Frames per API request: {FRAMES_PER_REQUEST}")
    print(f"API requests per batch: {REQUESTS_PER_BATCH}")
    print(f"Frames per batch: {FRAMES_PER_REQUEST * REQUESTS_PER_BATCH}")
    print()
    print("🔄 Script will automatically loop through all frames")
    print("⏸️  Pauses if quota exceeded, resume by running again")
    print("✅ Generates final JSON automatically when complete")
    print()
    
    result = process_video_auto_loop(
        video_url, 
        fps=0.5,
        frames_per_request=FRAMES_PER_REQUEST,
        requests_per_batch=REQUESTS_PER_BATCH,
        similarity_threshold=0.85
    )
    
    if result:
        print(f"\n{'='*50}")
        print(f"✅ PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Check 'ticker_results_final.json' for full results")
    else:
        print(f"\n{'='*50}")
        print(f"⏸️  PROCESSING PAUSED")
        print(f"{'='*50}")
        print(f"Run this script again to continue")