import os
import threading
import sys
import threading
import subprocess
import time
import zipfile
import shutil
import datetime
import logging
import json
import re
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_from_directory, Response

from extract_buildings import BuildingExtractor, Config


# ==========================================
# Helper Functions
# ==========================================
def safe_filename(filename):
    """
    Custom filename sanitizer that preserves non-ASCII characters (e.g. Chinese).
    Removes Windows invalid characters: < > : " / \ | ? *
    """
    # Get basename to avoid directory traversal
    filename = os.path.basename(filename)
    
    # Remove Windows invalid characters and control characters
    # Invalid: < > : " / \ | ? *
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'[\x00-\x1f]', '', filename)
    
    # Strip leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Fallback if empty
    if not filename:
        filename = "unnamed_file"
        
    return filename
# ==========================================
class LogCapture:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = []
        self.lock = threading.Lock()
        self.max_chars = 100000  # Keep last 100k chars approx
        self.log_file_path = None

    def set_log_file(self, path):
        with self.lock:
            self.log_file_path = path
            if path:
                try:
                    # Create/Overwrite the file with a header
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(f"=== Log Started at {datetime.datetime.now()} ===\n")
                except Exception:
                    pass

    def write(self, message):
        # Handle bytes
        if isinstance(message, bytes):
            try:
                message = message.decode('utf-8', errors='replace')
            except Exception:
                message = str(message)

        # Write to original stream (console)
        try:
            self.original_stream.write(message)
            self.original_stream.flush()
        except Exception:
            pass
        
        # Filter unwanted startup messages from UI logs and File logs
        if (
            "Starting Web UI at" in message or 
            "Serving Flask app" in message or 
            "Debug mode: off" in message or
            "Running on http" in message
        ):
             return

        # Write to file if set
        # Note: We access self.log_file_path without lock for speed, 
        # but since it's atomic assignment it's mostly fine. 
        # For strict correctness we could lock, but might slow down console.
        # Let's just use the value.
        path = self.log_file_path
        if path:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(message)
            except Exception:
                pass
        
        # Store in buffer
        with self.lock:
            self.buffer.append(message)
            # Simple cleanup strategy: if too long, cut first half
            current_len = sum(len(s) for s in self.buffer)
            if current_len > self.max_chars:
                # Remove from beginning until under limit
                while self.buffer and current_len > self.max_chars:
                    removed = self.buffer.pop(0)
                    current_len -= len(removed)

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
            
    def get_content(self):
        with self.lock:
            return "".join(self.buffer)

# Redirect stdout and stderr
log_capture = LogCapture(sys.stdout)
sys.stdout = log_capture
sys.stderr = log_capture


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB upload limit
app.config['JSON_AS_ASCII'] = False  # Ensure JSON responses use UTF-8 instead of escaping
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.getLogger("werkzeug").setLevel(logging.ERROR)

task_state = {
    "running": False,
    "progress": 0,
    "step_progress": 0,
    "status": "空闲",
    "output_path": None,
    "debug_image": None,
    "download_url": None,
    "task_start_time": None,
    "task_end_time": None,
    "step_start_time": None,
    "step_end_time": None,
    "pause_requested": False,
    "stop_requested": False,
    "task_name": None,
}

state_lock = threading.Lock()


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>建筑提取控制台</title>
  <style>
    :root {
      /* Professional Slate/Blue Theme - Unified */
      --primary-color: #2563eb; /* Blue 600 */
      --primary-dark: #1d4ed8;  /* Blue 700 */
      --primary-light: #eff6ff; /* Blue 50 */
      
      --secondary-color: #64748b; /* Slate 500 */
      --secondary-bg: #f1f5f9;    /* Slate 100 */
      
      --success-color: #10b981; /* Emerald 500 */
      --warning-color: #f59e0b; /* Amber 500 */
      --danger-color: #ef4444;  /* Red 500 */
      
      --bg-color: #f8fafc;      /* Slate 50 */
      --card-bg: #ffffff;
      --surface-color: #ffffff;
      --text-color: #0f172a;    /* Slate 900 */
      --text-muted: #64748b;    /* Slate 500 */
      
      --border-color: #e2e8f0;  /* Slate 200 */
      --radius-md: 6px;
      --radius-lg: 8px;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
      
      /* Legacy compat */
      --primary-hover: var(--primary-dark);
      --input-bg: #f8fafc;
    }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg-color);
      color: var(--text-color);
      margin: 0;
      padding: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .header {
      background: #fff;
      padding: 0.75rem 1.5rem;
      border-bottom: 1px solid var(--border-color);
      box-shadow: var(--shadow-sm);
      z-index: 20;
      display: flex;
      align-items: center;
      gap: 1.5rem;
    }
    .header-title {
      font-size: 1.4rem;
      font-weight: 700;
      color: #111827;
      white-space: nowrap;
    }
    .project-selector {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex: 1; /* Take up space */
    }
    .project-selector label {
      margin: 0;
      font-size: 1rem;
      font-weight: 500;
      margin-left: auto; /* Push to right */
    }
    .project-selector select {
      padding: 0.5rem 1rem;
      font-size: 0.8rem;
      border-radius: 0.375rem;
      background: #fff;
      min-width: 300px;
      max-width: 600px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      cursor: pointer;
      font-weight: 300;
      color: #1f2937;
    }
    .project-selector select:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    .project-selector button {
      padding: 0.4rem 0.8rem;
      font-size: 0.85rem;
    }
    .conn-status {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.85rem;
      color: #6b7280;
    }
    .conn-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #9ca3af;
      box-shadow: 0 0 0 1px rgba(0,0,0,0.05);
    }
    .conn-dot.online {
      background: #22c55e;
    }
    .conn-dot.offline {
      background: #ef4444;
    }
    .main-layout {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    /* Left Panel: Settings */
    .left-panel {
      width: clamp(340px, 20vw, 500px);
      min-width: 340px;
      background: #fff;
      border-right: 1px solid var(--border-color);
      overflow-y: auto;
      padding: 1rem;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      box-shadow: 4px 0 24px rgba(0,0,0,0.02);
      z-index: 10;
    }
    /* Right Panel: Preview */
    .right-panel {
      flex: 1;
      overflow-y: hidden; /* Changed from auto to hidden to manage internal scrolling */
      display: flex;
      flex-direction: column;
      background: #f8fafc;
      height: 100%; /* Ensure full height */
      box-sizing: border-box;
      padding: 0.5rem;
    }

    /* Status Bar */
    .status-bar {
        width: 100%;
        margin: 0 auto 0.5rem auto;
        padding: 0.5rem;
        background: #fff;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        box-sizing: border-box;
    }
    .status-meta {
        margin-top: 0.25rem;
        font-size: 0.75rem;
        color: #6b7280;
        display: flex;
        justify-content: space-between;
    }

    /* Preview Card */
    .preview-card {
        flex: 1; /* Take up available space */
        width: 100%;
        margin: 0 auto 0.5rem auto;
        background: #fff;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        padding: 0;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        min-height: 0; /* Allow shrinking */
        overflow: hidden;
    }
    .preview-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        color: #111827;
        flex-shrink: 0;
    }
    .preview-card img {
        flex: 1;
        width: 100%;
        height: 100%;
        object-fit: contain; /* Contain image within area */
        background: #f0f0f0;
        border-radius: 0.25rem;
    }

    /* Logs Panel (Moved to Right Panel) */
    .logs-panel {
        width: 100%;
        height: 25vh; /* Responsive height */
        min-height: 150px;
        max-height: 400px;
        background: #fff;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        display: flex;
        flex-direction: column;
        flex-shrink: 0; /* Don't shrink */
        box-sizing: border-box; /* Ensure padding/border doesn't overflow */
    }
    .logs-header {
        width: 100%;
        padding: 0.75rem 1rem;
        background: #f9fafb;
        border-bottom: 1px solid var(--border-color);
        font-weight: 600;
        font-size: 0.875rem;
        color: #374151;
        box-sizing: border-box;
    }
    .logs-body {
        width: 100%;
        flex: 1;
        overflow-y: auto;
        padding: 0;
        background: #0d1117; /* Dark terminal background */
        color: #c9d1d9;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        border-bottom-left-radius: var(--radius-md);
        border-bottom-right-radius: var(--radius-md);
        box-sizing: border-box;
    }
    #log-container {
        padding: 1rem;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .log-line {
        margin-bottom: 2px;
        line-height: 1.4;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      margin-bottom: 1rem;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
      transition: box-shadow 0.2s, transform 0.2s;
    }
    .card:hover {
      box-shadow: var(--shadow-md);
    }
    .card-header {
      background: #f9fafb;
      padding: 0.75rem 1rem;
      font-weight: 600;
      font-size: 0.875rem;
      color: #374151;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
      transition: background-color 0.2s;
    }
    .card-header:hover {
      background: #f3f4f6;
    }
    .card-body {
      padding: 1rem;
      display: none;
      border-top: 1px solid var(--border-color);
    }
    .card-body.open {
      display: block;
      animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-5px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Form Elements */
    .form-group {
      margin-bottom: 1rem;
    }
    .form-group:last-child {
      margin-bottom: 0;
    }
    label {
      display: block;
      margin-bottom: 0.375rem;
      font-weight: 500;
      font-size: 0.8rem;
      color: #4b5563;
    }
    input[type="text"],
    input[type="number"],
    select {
      width: 100%;
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      font-size: 0.875rem;
      background-color: var(--input-bg);
      box-sizing: border-box;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    input[type="text"]:focus,
    input[type="number"]:focus,
    select:focus {
      border-color: var(--primary-color);
      outline: none;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
      background-color: #fff;
    }
    
    /* Upload Area Styling */
    .upload-area {
      border: 2px dashed var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: #fafafa;
    }
    .upload-area:hover {
      border-color: var(--primary-color);
      background: #eff6ff;
    }
    .upload-area p {
      margin: 0.5rem 0 0;
      font-size: 0.875rem;
      color: #6b7280;
    }
    .upload-icon {
      font-size: 24px;
      color: #9ca3af;
      margin-bottom: 0.5rem;
    }

    /* Input Group for File Selection */
    /* Gallery Styles */
    .gallery-layout {
      display: flex;
      gap: 1rem;
      flex: 1;
      overflow: hidden;
      min-height: 0;
      padding: 0.75rem;
    }
    .gallery-list {
      width: 240px;
      min-width: 220px;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      background: #fff;
      overflow-y: auto;
      padding: 0.4rem;
      box-sizing: border-box;
      min-height: 0;
    }
    .gallery-item {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      align-items: center;
      padding: 0.35rem;
      border: 1px solid transparent;
      border-radius: 0.5rem;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
      margin-bottom: 0.25rem;
    }
    .gallery-item:hover {
      background: #f0f9ff;
      border-color: var(--primary-color);
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }
    .gallery-item.selected {
      border-color: var(--primary-color);
      box-shadow: var(--shadow-sm);
      background: rgba(59, 130, 246, 0.06);
    }
    .gallery-thumb {
      width: 56px;
      height: 56px;
      object-fit: cover;
      background: #f0f0f0;
      border-radius: 0.5rem;
      flex: 0 0 auto;
    }
    .gallery-caption {
      font-size: 0.78rem;
      color: #374151;
      line-height: 1.25;
      word-break: break-word;
      text-align: center;
    }
    .preview-pane {
      flex: 1;
      min-width: 0;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      background: #fff;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .preview-pane-header {
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border-color);
      background: #f9fafb;
      color: #374151;
      font-size: 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      min-height: 40px;
      box-sizing: border-box;
    }
    .preview-pane-body {
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f3f4f6;
    }
    .preview-pane-body img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #f3f4f6;
    }

    /* Modal Styles */
    .modal {
      display: none;
      position: fixed;
      z-index: 2000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: rgba(0,0,0,0.9);
    }
    .modal-content-container {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .modal-image {
      max-width: 90%;
      max-height: 90%;
      transition: transform 0.1s ease;
      cursor: grab;
      box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    .modal-image:active {
      cursor: grabbing;
    }
    .modal-close {
      position: absolute;
      top: 20px;
      right: 35px;
      color: #f1f1f1;
      font-size: 40px;
      font-weight: bold;
      cursor: pointer;
      z-index: 2001;
      text-shadow: 0 0 5px #000;
    }
    .modal-close:hover,
    .modal-close:focus {
      color: #bbb;
      text-decoration: none;
      cursor: pointer;
    }
    .modal-caption {
      position: absolute;
      bottom: 20px;
      left: 0;
      width: 100%;
      text-align: center;
      color: #fff;
      padding: 10px;
      pointer-events: none;
      text-shadow: 0 1px 3px rgba(0,0,0,0.8);
      font-size: 1.1rem;
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .btn-small {
      background-color: #fff;
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      padding: 0.375rem 0.75rem;
      cursor: pointer;
      font-size: 0.8rem;
      color: #374151;
      transition: all 0.2s;
    }
    .btn-small:hover {
      background-color: #f3f4f6;
      border-color: #d1d5db;
    }
    
    /* Merged Rows */
    .row-group {
      display: flex;
      gap: 0.75rem;
    }
    .row-group .form-group {
      flex: 1;
    }
    
    /* Range Inputs Wrapper */
    .range-wrapper {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .range-wrapper input {
      text-align: center;
    }
    .range-separator {
      color: #9ca3af;
    }

    /* Checkbox */
    .checkbox-label {
      display: flex;
      align-items: center;
      cursor: pointer;
      font-size: 0.875rem;
      color: #374151;
      user-select: none;
    }
    input[type="checkbox"] {
      width: 1rem;
      height: 1rem;
      margin-right: 0.5rem;
      accent-color: var(--primary-color);
      cursor: pointer;
    }

    /* Buttons */
    .btn-primary {
      background-color: var(--primary-color);
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.375rem;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      transition: background-color 0.2s, transform 0.1s;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .btn-secondary {
      background-color: #e5e7eb;
      color: #374151;
      border: none;
      padding: 0.55rem 1.1rem;
      border-radius: 0.375rem;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: background-color 0.15s, transform 0.05s, box-shadow 0.05s;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .btn-secondary:hover {
      background-color: #d1d5db;
      transform: translateY(-1px);
    }
    .btn-primary:hover {
      background-color: var(--primary-hover);
      transform: translateY(-1px);
    }
    .btn-primary:active {
      transform: translateY(0);
    }
    .btn-primary:disabled {
      background-color: #9ca3af;
      cursor: not-allowed;
      transform: none;
    }
    
    .btn-download {
      background-color: var(--success-color) !important;
    }
    .btn-download:hover {
      background-color: #059669 !important;
    }

    /* Control Buttons */
    .control-bar {
        display: flex;
        gap: 0.5rem; /* Reduced gap to prevent wrapping */
        padding: 1rem 0;
        background: transparent;
        justify-content: center;
        align-items: center;
        width: 100%;
        flex-wrap: nowrap; /* Prevent wrapping */
    }

    .btn-control {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.25rem;
        padding: 0.5rem 1rem; /* More padding */
        font-size: 0.85rem;
        font-weight: 500;
        border-radius: var(--radius-md); 
        border: 1px solid transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        min-width: 80px; 
        box-shadow: var(--shadow-sm);
        height: 36px;
        white-space: nowrap; 
        /* flex: 1; Removed to prevent stretching */
        color: white; 
    }

    .btn-control:disabled {
        background-color: var(--secondary-bg) !important;
        border-color: var(--border-color) !important;
        color: var(--text-muted) !important;
        cursor: not-allowed;
        box-shadow: none;
        transform: none !important;
        opacity: 0.6;
    }
    .btn-control:disabled .btn-icon {
        fill: var(--text-muted);
    }

    .btn-control:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        filter: brightness(1.1); /* Slight brighten on hover */
    }

    .btn-control:active:not(:disabled) {
        transform: translateY(0);
        opacity: 0.95;
    }

    /* Unified Style: Solid Buttons for Primary Actions */
    
    /* Start - Blue */
    .btn-start {
        background-color: var(--primary-color);
        border-color: var(--primary-color);
    }

    /* Pause - Amber */
    .btn-pause {
        background-color: var(--warning-color);
        border-color: var(--warning-color);
    }

    /* Resume - Emerald */
    .btn-resume {
        background-color: var(--success-color);
        border-color: var(--success-color);
    }

    /* Stop - Red */
    .btn-stop {
        background-color: var(--danger-color);
        border-color: var(--danger-color);
    }

    .btn-icon {
        width: 18px;
        height: 18px;
        fill: currentColor;
        transition: fill 0.2s;
    }

    /* Preview Area */
    .preview-card {
      background: #fff;
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-md);
      width: 100%;
      box-sizing: border-box;
      text-align: center;
      transition: box-shadow 0.3s;
    }
    .preview-card:hover {
        box-shadow: var(--shadow-lg);
    }
    .preview-card h3 {
      margin-top: 0;
      margin-bottom: 1rem;
      font-size: 1rem;
      color: #111827;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 0.75rem;
      text-align: left;
    }
    #preview-img {
      max-width: 100%;
      max-height: 75vh;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      background: #f3f4f6;
      box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* Progress */
    .status-bar {
      margin-bottom: 0.5rem;
      width: 100%;
      max-width: 98%;
      padding: 1rem;
    }
    .progress-container {
      height: 0.6rem;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 0.5rem;
    }
    .progress-fill {
      height: 100%;
      width: 0;
      background: var(--primary-color);
      transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .status-text {
      font-size: 0.875rem;
      color: #4b5563;
      display: flex;
      justify-content: space-between;
      font-weight: 500;
    }

    /* Tooltip */
    .tooltip {
      position: absolute;
      background: rgba(17, 24, 39, 0.9);
      color: #fff;
      padding: 0.5rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      z-index: 1000;
      pointer-events: none;
      max-width: 250px;
      opacity: 0;
      transform: translateY(5px);
      transition: opacity 0.2s, transform 0.2s;
      line-height: 1.4;
      box-shadow: var(--shadow-lg);
    }
    .tooltip.show {
      opacity: 1;
      transform: translateY(0);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: transparent; 
    }
    ::-webkit-scrollbar-thumb {
      background: #d1d5db; 
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #9ca3af; 
    }
    
    /* Log Panel */
    #log-container {
        background: #1e1e1e !important;
        color: #d4d4d4 !important;
        border: none !important;
        padding: 1rem !important;
        font-family: 'Consolas', 'Monaco', monospace;
        line-height: 1.5;
        height: 100%;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 0.8rem;
        box-sizing: border-box;
    }
    .bottom-panel {
      height: 250px;
      flex-shrink: 0;
      background: #fff;
      z-index: 30;
      width: 100%;
    }
    /* Step Indicator */
    .step-indicator {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-top: 1.5rem;
        position: relative;
        padding: 0 0.5rem;
    }
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        z-index: 2;
        min-width: 60px;
    }
    .step-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #f3f4f6;
        color: #9ca3af;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 600;
        transition: all 0.4s ease;
        border: 2px solid #fff;
        box-shadow: 0 0 0 2px #e5e7eb;
        position: relative;
    }
    .step-text {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 8px;
        font-weight: 500;
        text-align: center;
        transition: all 0.3s;
        white-space: nowrap;
    }
    
    /* Active State */
    .step.active .step-icon {
        background: #fff;
        color: var(--primary-color);
        box-shadow: 0 0 0 2px var(--primary-color);
    }
    .step.active .step-text {
        color: var(--primary-color);
        font-weight: 700;
        transform: translateY(-2px);
    }
    
    /* Loading Spinner Overlay for Active */
    .step.active .step-icon::before {
        content: '';
        position: absolute;
        top: -4px;
        left: -4px;
        right: -4px;
        bottom: -4px;
        border-radius: 50%;
        border: 2px solid transparent;
        border-top-color: var(--primary-color);
        border-right-color: var(--primary-color); 
        animation: spin 1.2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Completed State */
    .step.completed .step-icon {
        background: var(--primary-color);
        color: #fff;
        box-shadow: 0 0 0 2px var(--primary-color);
        border-color: var(--primary-color);
    }
    .step.completed .step-text {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Error State */
    .step.error .step-icon {
        background: #ef4444; /* Red */
        color: #fff;
        box-shadow: 0 0 0 2px #ef4444;
        border-color: #ef4444;
    }
    .step.error .step-text {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Line */
    .step-line {
        flex: 1;
        height: 3px;
        background: #e5e7eb;
        margin-top: 16px; /* Center of 32px icon */
        margin-left: -10px;
        margin-right: -10px;
        z-index: 1;
        border-radius: 1px;
        transition: background 0.4s ease;
    }
    .step-line.completed {
        background: var(--primary-color);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* Toast Notifications */
    .toast-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 3000;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .toast {
        background: #fff;
        border-left: 4px solid var(--primary-color);
        padding: 12px 16px;
        border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 250px;
        transform: translateX(120%);
        transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
        font-size: 0.9rem;
        color: #374151;
    }
    .toast.show {
        transform: translateX(0);
    }
    .toast.success { border-left-color: var(--success-color); }
    .toast.error { border-left-color: var(--danger-color); }
    .toast.info { border-left-color: var(--primary-color); }

    /* Pulse Animation for Active Step */
    .step.active .step-icon {
        box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.7);
        animation: pulse-blue 2s infinite;
    }
    @keyframes pulse-blue {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
    }
  </style>
</head>
<body>
  <div class="toast-container" id="toast-container"></div>
  <div class="header">
  <div class="header-title">建筑提取控制台</div>
    <div class="project-selector">
      <label for="task-select"> 当前任务 :</label>
      <select id="task-select" onchange="onTaskSelectChange(this.value)">
        <option value="">-- 选择或新建任务 --</option>
      </select>
      <button type="button" class="btn-secondary" onclick="refreshTasks()">刷新</button>
      <button type="button" class="btn-secondary" onclick="clearHistoryTasks()">清空历史</button>
    </div>
    <div class="conn-status">
      <span class="conn-dot" id="conn-dot"></span>
      <span id="conn-text">连接检测中...</span>
    </div>
  </div>
  
  <div class="main-layout">
    <!-- LEFT PANEL: CONFIG -->
    <div class="left-panel">
      <form id="config-form">
        
        <!-- Files -->
        <div class="card">
          <div class="card-header" onclick="toggleSection('sec-files')">
            <span>影像输入</span>
            <span>▼</span>
          </div>
          <div class="card-body open" id="sec-files">
            <div class="form-group">
              <label>1. 选择正射影像源 (TIFF)</label>
              
              <div style="margin-bottom: 10px; display: flex; gap: 5px; align-items: center;">
                  <select id="uploaded-history-select" class="form-control" onchange="onHistorySelectChange(this.value)" style="flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 6px; background-color: #fff;">
                      <option value="">-- 从记录中选择 --</option>
                  </select>
                  <button type="button" class="btn-secondary" onclick="clearUploadHistory()" style="padding: 8px 12px; white-space: nowrap; line-height: 1.2;">清除全部</button>
              </div>

              <div class="upload-area" onclick="document.getElementById('file-input').click()" id="drop-zone">
                <div class="upload-icon">📁</div>
                <p id="upload-text" style="font-weight:500; color:#4b5563;">点击选择文件</p>
                <p id="file-info" style="font-size:0.75rem; color:#9ca3af; margin-top:0.25rem;">支持 .tif, .tiff (Max 16GB)</p>
                <input type="file" id="file-input" accept=".tif,.tiff" style="display:none" onchange="handleFileSelect()">
              </div>

              <div id="upload-progress-container" style="display:none; margin-top:1rem;">
                 <div class="progress-container">
                    <div id="upload-progress-bar" class="progress-fill" style="width: 0%"></div>
                 </div>
                 <div class="status-text">
                    <span style="font-size:0.75rem;">上传中...</span>
                    <span id="upload-status-text" style="font-size:0.75rem;">0%</span>
                 </div>
              </div>

              <button type="button" id="btn-upload" class="btn-primary" style="margin-top:1rem;" onclick="uploadFile()" disabled>
                上传到服务器
              </button>
              <input type="hidden" name="uploaded_filename" id="input-uploaded-filename">
              <div id="upload-success-msg" style="display:none; color:var(--success-color); font-size:0.875rem; margin-top:0.5rem; font-weight:500; text-align:center;">
                ✓ 上传成功，准备就绪
              </div>
            </div>
            
            <div class="form-group" style="margin-top:1.5rem; border-top: 1px dashed var(--border-color); padding-top:1rem; display:none;">
                <label style="color:#6b7280;">2. 处理结果 (Moved to bottom)</label>
            </div>
          </div>
        </div>

        <!-- AI Model -->
        <div class="card">
          <div class="card-header" onclick="toggleSection('sec-ai')">
            <span>AI 模型设置</span>
            <span>▼</span>
          </div>
          <div class="card-body" id="sec-ai">
            <div class="row-group">
                <div class="form-group">
                  <label>分块大小 (Crop)</label>
                  <input type="number" name="BLOCK_SIZE" value="1024"
                    onfocus="showTooltip(this, '预测窗口大小 (Crop Size)')" onblur="hideTooltip()">
                </div>
                <div class="form-group">
                  <label>滑动步长 (Stride)</label>
                  <input type="number" name="BLOCK_OVERLAP" value="512"
                    onfocus="showTooltip(this, '滑动窗口步长 (Stride)')" onblur="hideTooltip()">
                </div>
            </div>
            
            <div class="row-group">
              <div class="form-group">
                <label>推理模型</label>
                <select name="AI_MODEL_PATH" onfocus="showTooltip(this, '选择 AI 模型')" onblur="hideTooltip()">
                    <option value="model/iter_40000.pth" selected>HRNet-W32</option>
                    <option value="model/hrnet_w48_iter_6000.pth">HRNet-W48</option>
                </select>
              </div>
              <div class="form-group">
                <label>模型推理尺寸</label>
                <input type="number" name="AI_IMG_SIZE" value="1024"
                  onfocus="showTooltip(this, 'MMSeg 推理输入尺寸')" 
                  onblur="hideTooltip()">
              </div>
            </div>
            
            <div class="form-group">
                <label>推理设备</label>
                <select name="AI_DEVICE" onfocus="showTooltip(this, '选择 CPU 或 CUDA (GPU)')" onblur="hideTooltip()">
                  <option value="auto" selected>自动 (Auto)</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA (GPU)</option>
                </select>
            </div>
          </div>
        </div>

        <!-- Post Process -->
        <div class="card">
          <div class="card-header" onclick="toggleSection('sec-post')">
            <span>后处理与过滤</span>
            <span>▼</span>
          </div>
          <div class="card-body" id="sec-post">
             <div class="form-group">
                <label class="checkbox-label">
                  <input type="checkbox" name="ENABLE_GLOBAL_FILL" checked>
                  启用全局孔洞填充
                </label>
             </div>
             <div class="form-group">
                <label class="checkbox-label">
                  <input type="checkbox" name="ENABLE_FILTER_AREA" checked>
                  启用面积过滤
                </label>
                 <input type="number" step="0.1" name="MIN_AREA_M2" value="100.0" style="width: 100px; margin-left: 10px;"
                    onfocus="showTooltip(this, '最小建筑面积 (平方米)')" onblur="hideTooltip()">
             </div>
             
             <div class="form-group">
               <label class="checkbox-label">
                 <input type="checkbox" name="ENABLE_SMOOTHING" checked>
                 平滑处理 (米)
               </label>
                <div class="range-wrapper">
                    <input type="number" step="0.1" name="BUFFER_EXPAND_M" value="0.6" placeholder="膨胀"
                       onfocus="showTooltip(this, '先膨胀')" onblur="hideTooltip()">
                    <span class="range-separator">/</span>
                    <input type="number" step="0.1" name="BUFFER_SHRINK_M" value="0.3" placeholder="腐蚀"
                       onfocus="showTooltip(this, '后腐蚀')" onblur="hideTooltip()">
                </div>
             </div>
             
             <div class="form-group">
               <label class="checkbox-label">
                 <input type="checkbox" name="ENABLE_SIMPLIFY" checked>
                 最终简化 (Douglas-Peucker)
               </label>
                <div class="row-group">
                    <input type="number" step="0.01" name="SIMPLIFY_TOL" value="0.3" placeholder="公差"
                       onfocus="showTooltip(this, '简化公差')" onblur="hideTooltip()">
                    <input type="number" step="0.01" name="RECT_IOU_THRESH" value="0.90" placeholder="矩形IoU"
                       onfocus="showTooltip(this, '若IoU>阈值则转为矩形')" onblur="hideTooltip()">
                </div>
             </div>
          </div>
        </div>
        <!-- Control & Result Panel (Moved to Bottom) -->
        <div class="card">
          <div class="card-header" onclick="toggleSection('sec-control')">
            <span>操作与结果</span>
            <span>▼</span>
          </div>
          <div class="card-body open" id="sec-control">
             <div class="form-group">
                <div class="control-bar">
                    <button type="button" class="btn-control btn-start" onclick="startJob()" id="btn-start" disabled>
                        <svg class="btn-icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                        <span>开始处理</span>
                    </button>
                    <button type="button" class="btn-control btn-pause" onclick="pauseJob()" id="btn-pause" style="display:none;">
                        <svg class="btn-icon" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                        <span>暂停</span>
                    </button>
                    <button type="button" class="btn-control btn-resume" onclick="resumeJob()" id="btn-resume" style="display:none;">
                        <svg class="btn-icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                        <span>继续</span>
                    </button>
                    <button type="button" class="btn-control btn-stop" onclick="stopJob()" id="btn-stop" style="display:none;">
                        <svg class="btn-icon" viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>
                        <span>停止</span>
                    </button>
                </div>
                <div style="font-size:0.875rem; color:#6b7280; line-height:1.5; background:#f9fafb; padding:0.75rem; border-radius:0.375rem; margin-bottom: 0.75rem;">
                    任务完成后，将自动打包生成 ZIP 文件供下载。
                </div>
             </div>
          </div>
        </div>

        <div style="height: 50px;"></div> <!-- Spacer -->


      </form>
    </div>

    <!-- RIGHT PANEL: PREVIEW -->
    <div class="right-panel">
      <div class="status-bar">
         <div class="status-item">
             <div class="status-text">
                <span>处理流程</span>
                <span id="total-time-label">总耗时 --:--</span>
             </div>
             <!-- Step Indicator -->
             <div class="step-indicator">
                  <div class="step" data-step="1">
                    <div class="step-icon">1</div>
                    <div class="step-text">输入数据</div>
                  </div>
                  <div class="step-line" data-step="1"></div>
                  
                  <div class="step" data-step="2">
                    <div class="step-icon">2</div>
                    <div class="step-text">AI预测</div>
                  </div>
                  <div class="step-line" data-step="2"></div>
                  
                  <div class="step" data-step="3">
                    <div class="step-icon">3</div>
                    <div class="step-text">面积过滤</div>
                  </div>
                  <div class="step-line" data-step="3"></div>
                  
                  <div class="step" data-step="4">
                    <div class="step-icon">4</div>
                    <div class="step-text">孔洞填充</div>
                  </div>
                  <div class="step-line" data-step="4"></div>
                  
                  <div class="step" data-step="5">
                    <div class="step-icon">5</div>
                    <div class="step-text">平滑处理</div>
                  </div>
                  <div class="step-line" data-step="5"></div>

                  <div class="step" data-step="6">
                    <div class="step-icon">6</div>
                    <div class="step-text">简化处理</div>
                  </div>
                  <div class="step-line" data-step="6"></div>

                  <div class="step" data-step="7">
                    <div class="step-icon">7</div>
                    <div class="step-text">压缩输出</div>
                  </div>
             </div>
         </div>
         
         <div class="status-item" style="margin-top: 10px;">
              <div class="status-text">
                 <span id="status-label">当前步骤 (Step)</span>
                 <span id="step-percent-label">0%</span>
             </div>
             <div class="progress-container">
                 <div class="progress-fill" id="step-progress-bar" style="background-color: #10b981;"></div>
             </div>
             <div class="status-meta">
                 <span id="step-time-label">当前步骤耗时 --:--</span>
             </div>
         </div>
      </div>
      
      <div class="preview-card">
         <div class="card-header" style="border-bottom: 1px solid var(--border-color);">
            <span>实时预览</span>
            <span class="badge" id="gallery-count" style="background: var(--primary-color); color: white; padding: 0.1rem 0.4rem; border-radius: 99px; font-size: 0.75rem;">0</span>
         </div>
         <div class="gallery-layout">
             <div id="gallery-container" class="gallery-list">
                 <div style="color: #9ca3af; text-align: center; padding: 2rem 0.5rem; font-size: 0.875rem;">
                     等待任务产出中间结果...<br>(Waiting for outputs...)
                 </div>
             </div>
             <div class="preview-pane">
                 <div class="preview-pane-header">
                     <span id="preview-title">未选择图片</span>
                     <a id="preview-open-link" href="#" target="_blank" style="display:none; color: var(--primary-color); text-decoration: none;">新窗口打开</a>
                 </div>
                 <div class="preview-pane-body">
                     <img id="preview-main-img" alt="preview" src="">
                 </div>
             </div>
         </div>
      </div>
      
      <!-- Logs Panel -->
      <div class="logs-panel">
          <div class="logs-header" style="display: flex; justify-content: space-between; align-items: center;">
              <span>运行日志</span>
              <div style="display: flex; gap: 5px;">
                  <button type="button" class="btn-small" onclick="copyLogs()" title="复制日志">📋</button>
                  <button type="button" class="btn-small" onclick="clearLogs()" title="清空日志">🗑️</button>
              </div>
          </div>
          <div class="logs-body">
              <div id="log-container"></div>
          </div>
      </div>
    </div>
  </div>
  
  <div id="tooltip" class="tooltip"></div>

  <script>
  let polling = null;
  let tasksCache = [];
  let selectedTaskId = "";
  let wasOnline = true;
  let wasRunning = false;
  let lastGalleryCount = 0;

  // Helper Functions
  function showToast(message, type = 'info') {
      const container = document.getElementById('toast-container');
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      
      let icon = 'ℹ️';
      if (type === 'success') icon = '✅';
      if (type === 'error') icon = '❌';
      
      toast.innerHTML = `<span style="font-size: 1.2rem;">${icon}</span><span>${message}</span>`;
      
      container.appendChild(toast);
      
      // Trigger animation
      requestAnimationFrame(() => {
          toast.classList.add('show');
      });
      
      setTimeout(() => {
          toast.classList.remove('show');
          setTimeout(() => {
              toast.remove();
          }, 300);
      }, 6000);
  }

  function copyLogs() {
      const container = document.getElementById('log-container');
      if (!container) return;
      const text = container.innerText;
      navigator.clipboard.writeText(text).then(() => {
          showToast('日志已复制到剪贴板', 'success');
      }).catch(err => {
          console.error('Copy failed', err);
          showToast('复制失败', 'error');
      });
  }

  function clearLogs() {
      const container = document.getElementById('log-container');
      if (container) {
          container.innerText = '';
          showToast('日志已清空', 'info');
      }
  }

  function ensureDownloadButton(url, sizeStr) {
      if (!url) return;
      let dlBtn = document.getElementById('btn-download-result');
      const startBtn = document.getElementById('btn-start');
      if (!dlBtn && startBtn && startBtn.parentNode) {
          dlBtn = document.createElement('a');
          dlBtn.id = 'btn-download-result';
          dlBtn.className = 'btn-primary btn-download';
          dlBtn.style.display = 'block';
          dlBtn.style.marginTop = '10px';
          dlBtn.style.textAlign = 'center';
          dlBtn.style.textDecoration = 'none';
          dlBtn.style.boxSizing = 'border-box';
          startBtn.parentNode.insertBefore(dlBtn, startBtn.nextSibling);
      }
      
      let text = '下载处理结果 (ZIP)';
      if (sizeStr) {
          text += ` (${sizeStr})`;
      }
      
      if (dlBtn) {
          dlBtn.innerText = text;
          dlBtn.href = url;
      }
  }

  function onTaskSelectChange(value) {
      selectedTaskId = value || "";
      selectedPreviewUrl = ""; // Reset selected preview
      setMainPreview(null); // Clear current preview
      pollGallery();
      
      // Reset download button
      const oldDl = document.getElementById('btn-download-result');
      if (oldDl) oldDl.remove();

      if (!selectedTaskId) {
          // Switching to New Task: Reset UI to allow new upload
          // We sync with the actual file input state
          handleFileSelect();
          // Reset history selection
          const historySelect = document.getElementById('uploaded-history-select');
          if (historySelect) historySelect.value = "";
          return;
      }

      const info = tasksCache.find(t => t.id === selectedTaskId);
      if (info) {
          // 1. Restore Image Info
          const historySelect = document.getElementById('uploaded-history-select');
          if (historySelect) {
              // Try to select the history item if it exists
              historySelect.value = info.uploaded_filename || "";
          }

          if (info.uploaded_filename) {
              const uploadText = document.getElementById('upload-text');
              const uploadBtn = document.getElementById('btn-upload');
              const startBtn = document.getElementById('btn-start');
              const uploadedInput = document.getElementById('input-uploaded-filename');
              const successMsg = document.getElementById('upload-success-msg');

              if (uploadedInput) uploadedInput.value = info.uploaded_filename;
              if (uploadText) {
                  uploadText.innerText = info.uploaded_filename;
                  uploadText.style.color = '#374151';
              }
              
              if (uploadBtn) {
                  uploadBtn.disabled = true; 
                  uploadBtn.innerText = "已关联影像";
              }
              
              if (startBtn) {
                  startBtn.disabled = false;
                  startBtn.innerText = "开始处理";
              }
              
              if (successMsg) successMsg.style.display = 'none';
          } else {
              const uploadText = document.getElementById('upload-text');
              const uploadBtn = document.getElementById('btn-upload');
              const startBtn = document.getElementById('btn-start');
              const uploadedInput = document.getElementById('input-uploaded-filename');
              const successMsg = document.getElementById('upload-success-msg');
              
              if (uploadedInput) uploadedInput.value = "";
              if (uploadText) {
                  uploadText.innerText = "未关联影像 (旧任务)";
                  uploadText.style.color = '#9ca3af';
              }
              if (uploadBtn) {
                  uploadBtn.disabled = true;
                  uploadBtn.innerText = "无法上传";
              }
              if (startBtn) {
                  startBtn.disabled = true;
                  startBtn.innerText = "无法重新开始";
              }
              if (successMsg) successMsg.style.display = 'none';
          }

          // 2. Restore Download Button
          if (info.has_zip) {
              const url = '/download/' + encodeURIComponent(info.id) + '.zip';
              ensureDownloadButton(url);
          }

          // 3. Restore Task Info (Logs, Config, Status)
          fetch('/task_info?task_id=' + encodeURIComponent(info.id))
            .then(r => r.json())
            .then(data => {
                // Restore Config
                if (data.meta && data.meta.config) {
                    const cfg = data.meta.config;
                    for (const key in cfg) {
                        const el = document.querySelector(`[name="${key}"]`);
                        if (el) {
                            if (el.type === 'checkbox') {
                                el.checked = cfg[key];
                                // Trigger change event for toggles
                                const event = new Event('change');
                                el.dispatchEvent(event);
                            } else {
                                el.value = cfg[key];
                            }
                        }
                    }
                }
                
                // Restore Logs
                if (data.logs) {
                    const logsPanel = document.getElementById('log-container');
                    if (logsPanel) {
                        logsPanel.innerText = data.logs;
                        logsPanel.scrollTop = logsPanel.scrollHeight;
                    }
                }
                
                // Restore Status Steps
                if (data.meta && data.meta.status === 'completed') {
                    // Mark all as completed
                    for (let i = 1; i <= 7; i++) {
                        const stepEl = document.querySelector(`.step[data-step="${i}"]`);
                        const lineEl = document.querySelector(`.step-line[data-step="${i}"]`);
                        if (stepEl) {
                            stepEl.classList.remove('active', 'error');
                            stepEl.classList.add('completed');
                        }
                        if (lineEl) lineEl.classList.add('completed');
                    }
                    
                    // Update Progress Bars
                    const pb = document.getElementById('progress-bar');
                    const spb = document.getElementById('step-progress-bar');
                    const sl = document.getElementById('status-label');
                    if (pb) pb.style.width = '100%';
                    if (spb) spb.style.width = '100%';
                    if (sl) sl.innerText = '已完成 (历史任务)';
                } else if (data.meta && data.meta.status === 'failed') {
                     const sl = document.getElementById('status-label');
                     if (sl) sl.innerText = '失败 (历史任务)';
                } else if (data.meta && data.meta.status === 'stopped') {
                     const sl = document.getElementById('status-label');
                     if (sl) sl.innerText = '已停止 (历史任务)';
                }
            })
            .catch(console.error);
      }
  }

  function refreshTasks(silent) {
      fetch('/tasks')
        .then(r => r.json())
        .then(list => {
            tasksCache = Array.isArray(list) ? list : [];
            const sel = document.getElementById('task-select');
            if (!sel) return;
            const prev = sel.value;
            sel.innerHTML = '';
            const optCurrent = document.createElement('option');
            optCurrent.value = '';
            optCurrent.textContent = '当前任务 (自动)';
            sel.appendChild(optCurrent);
            tasksCache.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t.id;
                const ts = typeof t.mtime === 'number' ? new Date(t.mtime * 1000) : null;
                let label = t.name || t.id;
                if (ts && !isNaN(ts.getTime())) {
                    label += '  ' + ts.toLocaleString();
                }
                if (t.is_current) label += ' [当前]';
                if (t.has_zip) label += ' [ZIP]';
                opt.textContent = label;
                sel.appendChild(opt);
            });
            let newVal = prev;
            if (prev && !tasksCache.some(t => t.id === prev)) {
                newVal = '';
            }
            sel.value = newVal;
            
            // Only trigger change handler if value actually changed AND not silent
            // This prevents resetting the Upload UI state (e.g. "Upload Complete")
            // when starting a new task (which calls refreshTasks but stays on "")
            if (newVal !== prev && !silent) {
                onTaskSelectChange(newVal);
            }
        })
        .catch(console.error);
  }

  function clearHistoryTasks() {
      if (!confirm('确定要清空历史任务吗？当前任务的结果会保留。')) return;
      fetch('/tasks/clear', {method: 'POST'})
        .then(r => r.json())
        .then(() => {
            const sel = document.getElementById('task-select');
            if (sel) sel.value = '';
            selectedTaskId = '';
            refreshTasks();
            pollGallery();
            showToast('历史任务已清除', 'success');
        })
        .catch(console.error);
  }

  function toggleSection(id) {
      const el = document.getElementById(id);
      if (el.classList.contains('open')) {
        el.classList.remove('open');
      } else {
        el.classList.add('open');
      }
    }

    function bindToggleToInputs(toggleSelector, inputSelectors) {
      const toggle = document.querySelector(toggleSelector);
      const inputs = inputSelectors.map(function (sel) {
        return document.querySelector(sel);
      }).filter(function (el) { return !!el; });
      if (!toggle || inputs.length === 0) return;
      function update() {
        inputs.forEach(function (el) {
          el.disabled = !toggle.checked;
        });
      }
      toggle.addEventListener('change', update);
      update();
    }

    const tooltipEl = document.getElementById('tooltip');

    function showTooltip(el, text) {
      tooltipEl.innerText = text;
      tooltipEl.classList.add('show');
      
      const rect = el.getBoundingClientRect();
      // Position tooltip to the right of the element if space allows, or top
      // Simplified to top for now
      const top = rect.top - tooltipEl.offsetHeight - 8; 
      const left = rect.left;
      
      tooltipEl.style.top = Math.max(0, top) + 'px';
      tooltipEl.style.left = left + 'px';
    }

    function hideTooltip() {
      tooltipEl.classList.remove('show');
    }

    // Drag and Drop Support
    const dropZone = document.getElementById('drop-zone');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
      dropZone.style.borderColor = 'var(--primary-color)';
      dropZone.style.backgroundColor = '#eff6ff';
    }

    function unhighlight(e) {
      dropZone.style.borderColor = 'var(--border-color)';
      dropZone.style.backgroundColor = '#fafafa';
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      const fileInput = document.getElementById('file-input');
      
      if (files.length > 0) {
          fileInput.files = files;
          handleFileSelect();
      }
    }

    function initSectionToggles() {
      bindToggleToInputs('input[name="ENABLE_FILTER_AREA"]', ['input[name="MIN_AREA_M2"]']);
      bindToggleToInputs('input[name="ENABLE_SMOOTHING"]', ['input[name="BUFFER_EXPAND_M"]', 'input[name="BUFFER_SHRINK_M"]']);
      bindToggleToInputs('input[name="ENABLE_SIMPLIFY"]', ['input[name="RECT_IOU_THRESH"]', 'input[name="SIMPLIFY_TOL"]']);
    }

    function handleFileSelect() {
      const fileInput = document.getElementById('file-input');
      const uploadText = document.getElementById('upload-text');
      const uploadBtn = document.getElementById('btn-upload');
      const startBtn = document.getElementById('btn-start');
      const uploadedInput = document.getElementById('input-uploaded-filename');
      const dropZone = document.getElementById('drop-zone');
      
      // Remove existing clear button if any
      const existingClear = document.getElementById('btn-clear-selection');
      if (existingClear) existingClear.remove();

      if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        uploadText.innerText = "已选择: " + file.name;
        uploadText.style.color = 'var(--primary-color)';
        uploadText.style.fontWeight = '600';
        
        uploadBtn.disabled = false;
        uploadBtn.innerText = "上传到服务器";
        const successMsg = document.getElementById('upload-success-msg');
        if (successMsg) successMsg.style.display = 'none';
        
        // Highlight Drop Zone
        if (dropZone) {
            dropZone.style.borderColor = 'var(--primary-color)';
            dropZone.style.backgroundColor = 'var(--primary-light)';
            dropZone.style.position = 'relative'; // Ensure relative for absolute positioning
            
            // Add Clear Button (X)
            const clearBtn = document.createElement('div');
            clearBtn.id = 'btn-clear-selection';
            clearBtn.innerHTML = '×'; // Times symbol
            clearBtn.style.position = 'absolute';
            clearBtn.style.top = '5px';
            clearBtn.style.right = '10px';
            clearBtn.style.fontSize = '24px';
            clearBtn.style.color = 'var(--danger-color)';
            clearBtn.style.cursor = 'pointer';
            clearBtn.style.lineHeight = '1';
            clearBtn.title = '清除选择';
            
            clearBtn.onclick = function(e) {
                e.stopPropagation();
                fileInput.value = '';
                handleFileSelect();
            };
            dropZone.appendChild(clearBtn);
        }

        if (startBtn) {
          startBtn.disabled = true;
        }
        if (uploadedInput) {
          uploadedInput.value = "";
        }
      } else {
        uploadText.innerText = "点击选择文件";
        uploadText.style.color = 'var(--text-muted)';
        uploadText.style.fontWeight = 'normal';
        uploadBtn.disabled = true;
        uploadBtn.innerText = "上传到服务器"; // Reset text
        
        if (dropZone) {
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.backgroundColor = '#fafafa';
        }

        if (startBtn) {
          startBtn.disabled = true;
        }
        if (uploadedInput) {
          uploadedInput.value = "";
        }
      }
    }

    function uploadFile() {
      const fileInput = document.getElementById('file-input');
      if (fileInput.files.length === 0) return;
      
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append('file', file);
      
      const btn = document.getElementById('btn-upload');
      btn.disabled = true;
      btn.innerText = "上传中...";
      
      const progressContainer = document.getElementById('upload-progress-container');
      const progressBar = document.getElementById('upload-progress-bar');
      const statusText = document.getElementById('upload-status-text');
      
      progressContainer.style.display = 'block';
      
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/upload', true);
      
      xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
          const percent = (e.loaded / e.total) * 100;
          progressBar.style.width = percent + '%';
          statusText.innerText = Math.round(percent) + '%';
        }
      };
      
      xhr.onload = function() {
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText);
          document.getElementById('input-uploaded-filename').value = data.filename;
          document.getElementById('upload-success-msg').style.display = 'block';
          btn.innerText = "上传完成";
          
          showToast('文件上传成功！', 'success');
          
          const startBtn = document.getElementById('btn-start');
          if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerText = "开始处理";
          }
          
          // Switch to New Task context to indicate new workflow
          const taskSelect = document.getElementById('task-select');
          if (taskSelect && taskSelect.value !== "") {
              taskSelect.value = "";
              onTaskSelectChange("");
          }
        } else {
          showToast('上传失败: ' + xhr.statusText, 'error');
          btn.disabled = false;
          btn.innerText = "重试上传";
        }
      };
      
      xhr.onerror = function() {
        showToast('上传发生网络错误', 'error');
        btn.disabled = false;
        btn.innerText = "重试上传";
      };
      
      xhr.send(formData);
    }

    function syncAiModelPath() {
      const sel = document.getElementById('ai-model-select');
      const hidden = document.getElementById('ai-model-path-hidden');
      const custom = document.getElementById('ai-model-path-custom');
      if (!sel || !hidden || !custom) return;
      const v = sel.value;
      if (v === 'custom') {
        custom.style.display = 'block';
        const customVal = custom.value.trim();
        hidden.value = customVal ? customVal : '';
      } else {
        custom.style.display = 'none';
        hidden.value = v;
      }
    }

    function initAiModelSelector() {
      const sel = document.getElementById('ai-model-select');
      const custom = document.getElementById('ai-model-path-custom');
      if (!sel || !custom) return;
      sel.addEventListener('change', syncAiModelPath);
      custom.addEventListener('input', syncAiModelPath);
      syncAiModelPath();
    }

    function startJob() {
      const uploaded = document.getElementById('input-uploaded-filename').value;
      if (!uploaded) {
        showToast("请先上传影像文件！", "error");
        return;
      }

      // Reset Context for New Run
      selectedTaskId = "";
      const taskSelect = document.getElementById('task-select');
      if (taskSelect) taskSelect.value = "";
      
      // Clear Gallery immediately
      const gallery = document.getElementById('gallery-container');
      if (gallery) gallery.innerHTML = '<div style="color:#9ca3af; text-align: center; padding: 20px;">等待生成...</div>';
      const badge = document.getElementById('gallery-count');
      if (badge) badge.innerText = '0';
      setMainPreview(null);

      const btn = document.getElementById('btn-start');
      btn.disabled = true;
      btn.innerText = '启动中...';
      
      // Remove old download button if exists
      const oldDl = document.getElementById('btn-download-result');
      if (oldDl) oldDl.remove();
      
      const form = document.getElementById('config-form');
      const fd = new FormData(form);
      
      fetch('/start', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
          if (data.ok) {
            updateStatus(data.message || '已启动', 0, 0, 0, 0);
            showToast('任务已启动', 'success');
            lastGalleryCount = 0;
            refreshTasks(true);
            if (!polling) {
              polling = setInterval(pollStatus, 1500);
            }
          } else {
            updateStatus(data.error || '启动失败', 0);
            resetBtn();
          }
        })
        .catch(err => {
          updateStatus('启动失败 ' + err, 0, 0, 0, 0);
          resetBtn();
        });
    }

    function resetBtn() {
       const btn = document.getElementById('btn-start');
       const uploadedInput = document.getElementById('input-uploaded-filename');
       const hasUploaded = uploadedInput && uploadedInput.value;
       if (btn) {
         btn.disabled = !hasUploaded;
         btn.innerText = '开始处理';
       }
       const btnPause = document.getElementById('btn-pause');
       const btnResume = document.getElementById('btn-resume');
       const btnStop = document.getElementById('btn-stop');
       if (btnPause) btnPause.style.display = 'none';
       if (btnResume) btnResume.style.display = 'none';
       if (btnStop) btnStop.style.display = 'none';
    }

    function pauseJob() {
      fetch('/control', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: 'pause'})
      }).catch(console.error);
    }

    function resumeJob() {
      fetch('/control', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: 'resume'})
      }).catch(console.error);
    }

    function stopJob() {
      if (!confirm('确定要停止当前任务吗？')) return;
      fetch('/control', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: 'stop'})
      }).catch(console.error);
    }

    function formatDuration(sec) {
       if (sec === undefined || sec === null || isNaN(sec)) return '--:--';
       const total = Math.floor(sec);
       const h = Math.floor(total / 3600);
       const m = Math.floor((total % 3600) / 60);
       const s = total % 60;
       const pad = n => n.toString().padStart(2, '0');
       if (h > 0) return pad(h) + ':' + pad(m) + ':' + pad(s);
       return pad(m) + ':' + pad(s);
    }

    function updateStatus(text, percent, stepPercent, taskElapsed, stepElapsed) {
       document.getElementById('status-label').innerText = '状态: ' + text;
       
       const fmt = (n) => {
           if (n === undefined || n === null) return '0.0';
           return Number(n).toFixed(1);
       };

       // Update Step Indicator
       let currentStep = 0;
       let isError = text.includes("出错") || text.includes("错误") || text.includes("失败") || text.includes("缺失");

       if (text.includes("输入数据") || text.includes("初始化")) currentStep = 1;
       else if (text.includes("AI") || text.includes("推理") || text.includes("预测")) currentStep = 2;
       else if (text.includes("面积") || text.includes("过滤") || text.includes("轮廓")) currentStep = 3;
       else if (text.includes("孔洞") || text.includes("填充")) currentStep = 4;
       else if (text.includes("平滑")) currentStep = 5;
       else if (text.includes("简化")) currentStep = 6;
       else if (text.includes("压缩") || text.includes("打包")) currentStep = 7;
       else if (text.includes("完成")) currentStep = 8;

       for (let i = 1; i <= 7; i++) {
            const stepEl = document.querySelector(`.step[data-step="${i}"]`);
            const lineEl = document.querySelector(`.step-line[data-step="${i}"]`);
            
            if (!stepEl) continue;

            stepEl.classList.remove('active', 'completed', 'error');
            if (lineEl) lineEl.classList.remove('completed');
            
            if (isError && i === currentStep) {
                stepEl.classList.add('error');
            } else if (i < currentStep) {
                stepEl.classList.add('completed');
                if (lineEl) lineEl.classList.add('completed');
            } else if (i === currentStep) {
                stepEl.classList.add('active');
            }
       }
       
       if (stepPercent !== undefined) {
           document.getElementById('step-percent-label').innerText = fmt(stepPercent) + '%';
           document.getElementById('step-progress-bar').style.width = stepPercent + '%';
       }

       const totalLabel = document.getElementById('total-time-label');
       const stepLabel = document.getElementById('step-time-label');
       if (totalLabel) totalLabel.innerText = '总耗时 ' + formatDuration(taskElapsed);
       if (stepLabel) stepLabel.innerText = '当前步骤耗时 ' + formatDuration(stepElapsed);
    }

    function setConnectionState(isOnline) {
       const dot = document.getElementById('conn-dot');
       const text = document.getElementById('conn-text');
       if (!dot || !text) return;
       dot.classList.remove('online', 'offline');
       if (isOnline) {
           dot.classList.add('online');
           text.innerText = '连接正常';
       } else {
           dot.classList.add('offline');
           text.innerText = '连接中断';
       }
    }

    function pollStatus() {
      fetch('/status')
        .then(r => {
          if (!r.ok) throw new Error('status ' + r.status);
          return r.json();
        })
        .then(data => {
          setConnectionState(true);
          if (!wasOnline) {
              showToast('已重新连接后端服务', 'success');
              wasOnline = true;
          }
          
          // If viewing a history task that is NOT the currently running task, skip live updates
          if (selectedTaskId && selectedTaskId !== data.task_id) {
              return;
          }

          updateStatus(
            data.status || '',
            data.progress || 0,
            data.step_progress || 0,
            data.task_elapsed,
            data.step_elapsed
          );
          
          const btn = document.getElementById('btn-start');
          const btnPause = document.getElementById('btn-pause');
          const btnResume = document.getElementById('btn-resume');
          const btnStop = document.getElementById('btn-stop');
          const isRunning = !!data.running;
          const isPaused = !!data.pause_requested;

          if (isRunning) {
             if (btn) {
               btn.disabled = true;
               btn.innerText = isPaused ? '已暂停' : '处理中...';
             }

             if (btnPause) btnPause.style.display = isPaused ? 'none' : 'inline-block';
             if (btnResume) btnResume.style.display = isPaused ? 'inline-block' : 'none';
             if (btnStop) btnStop.style.display = 'inline-block';
          } else {
             const uploadedInput = document.getElementById('input-uploaded-filename');
             const hasUploaded = uploadedInput && uploadedInput.value;
             if (btn) {
               btn.disabled = !hasUploaded;
               btn.innerText = '开始处理';
             }

             if (btnPause) btnPause.style.display = 'none';
             if (btnResume) btnResume.style.display = 'none';
             if (btnStop) btnStop.style.display = 'none';
             
             if (!selectedTaskId && data.download_url) {
                 ensureDownloadButton(data.download_url, data.zip_size);
             }
             
             if (polling) {
                clearInterval(polling);
                polling = null;
                
                // Task Finished Notification
                if (wasRunning) {
                   if (data.status && data.status.includes('完成')) {
                       showToast('处理流程结束', 'success');
                   } else if (data.status && (data.status.includes('失败') || data.status.includes('错误'))) {
                       showToast('处理流程失败: ' + data.status, 'error');
                   } else {
                       showToast('处理流程结束', 'info');
                   }
                }
             }
          }
          wasRunning = isRunning;
          
          pollGallery(isRunning);

          // Poll Logs (Only if we are updating status)
           fetch('/logs?t=' + Date.now())
             .then(r => r.json())
             .then(data => {
                 if (data.logs !== undefined) {
                     const el = document.getElementById('log-container');
                     if (el) {
                         // Check if we should auto-scroll (user is at bottom)
                         const isNearBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 50;
                         
                         if (el.textContent !== data.logs) {
                             el.textContent = data.logs;
                             if (isNearBottom) {
                                 el.scrollTop = el.scrollHeight;
                             }
                         }
                     }
                 }
             })
            .catch(console.error);
        })
        .catch(err => {
          console.error(err);
          setConnectionState(false);
          if (wasOnline) {
              showToast('与后端服务连接不正常', 'error');
              wasOnline = false;
          }
        });
    }

    // --- Gallery & Preview Pane Logic ---
    let selectedPreviewUrl = null;

    function setMainPreview(url, desc) {
        const img = document.getElementById('preview-main-img');
        const title = document.getElementById('preview-title');
        const link = document.getElementById('preview-open-link');

        if (!url) {
            selectedPreviewUrl = null;
            img.src = '';
            title.innerText = '未选择图片';
            link.style.display = 'none';
            link.href = '#';
            return;
        }

        selectedPreviewUrl = url;
        img.src = url;
        title.innerText = desc || url;
        link.href = url;
        link.style.display = 'inline';
    }

    function pollGallery(isRunning) {
        let url = '/debug_gallery';
        if (selectedTaskId) {
            url += '?task_id=' + encodeURIComponent(selectedTaskId);
        }
        fetch(url)
        .then(r => r.json())
        .then(files => {
            const container = document.getElementById('gallery-container');
            const countBadge = document.getElementById('gallery-count');
            
            if (!files || files.length === 0) return;

            if (isRunning && files.length > lastGalleryCount) {
                 showToast('实时预览已生成', 'success');
            }
            lastGalleryCount = files.length;

            countBadge.innerText = files.length;
            
            // Clear placeholder if data exists
            if (container.querySelector('div[style*="text-align: center"]')) {
                container.innerHTML = '';
            }
            
            const currentItems = container.querySelectorAll('.gallery-item');
            if (currentItems.length !== files.length) {
                container.innerHTML = '';
                files.forEach(file => {
                    const item = document.createElement('div');
                    item.className = 'gallery-item';
                    item.dataset.url = file.url;
                    item.onclick = () => {
                        setMainPreview(file.url, file.desc);
                        syncSelectedItem();
                    };
                    item.innerHTML = `
                        <img src="${file.url}" class="gallery-thumb" loading="lazy">
                        <div class="gallery-caption">${file.desc}</div>
                    `;
                    container.appendChild(item);
                });
            }

            const selectedStillExists = selectedPreviewUrl && files.some(f => f.url === selectedPreviewUrl);
            if (!selectedPreviewUrl || !selectedStillExists) {
                setMainPreview(files[0].url, files[0].desc);
            }

            syncSelectedItem();
        })
        .catch(console.error);
    }

    function syncSelectedItem() {
        const container = document.getElementById('gallery-container');
        const items = container.querySelectorAll('.gallery-item');
        items.forEach(item => {
            const isSelected = selectedPreviewUrl && item.dataset.url === selectedPreviewUrl;
            if (isSelected) item.classList.add('selected');
            else item.classList.remove('selected');
        });
    }


    function clearUploadHistory() {
        if (!confirm('确定要清除所有上传的影像文件吗？此操作无法撤销。')) return;
        fetch('/uploads/clear', {method: 'POST'})
            .then(r => r.json())
            .then(() => {
                loadUploadHistory();
                // Reset selection if needed
                const sel = document.getElementById('uploaded-history-select');
                if (sel) sel.value = "";
                onHistorySelectChange("");
                showToast('全部正射影像已清除', 'success');
            })
            .catch(console.error);
    }

    function loadUploadHistory() {
        fetch('/uploads')
            .then(r => r.json())
            .then(files => {
                const sel = document.getElementById('uploaded-history-select');
                if (!sel) return;
                sel.innerHTML = '<option value="">-- 从历史上传记录中选择 --</option>';
                files.forEach(f => {
                    const opt = document.createElement('option');
                    opt.value = f.name;
                    const date = new Date(f.mtime * 1000).toLocaleString();
                    opt.textContent = `${f.name} (${date})`;
                    sel.appendChild(opt);
                });
                
                // Attempt to restore selection if a task is already selected and we just populated the list
                if (selectedTaskId && tasksCache.length > 0) {
                     const info = tasksCache.find(t => t.id === selectedTaskId);
                     if (info && info.uploaded_filename) {
                         sel.value = info.uploaded_filename;
                     }
                }
            })
            .catch(console.error);
    }

    function onHistorySelectChange(val) {
        if (!val) return;
        
        const uploadedInput = document.getElementById('input-uploaded-filename');
        const uploadText = document.getElementById('upload-text');
        const uploadBtn = document.getElementById('btn-upload');
        const startBtn = document.getElementById('btn-start');
        const successMsg = document.getElementById('upload-success-msg');
        
        if (uploadedInput) uploadedInput.value = val;
        if (uploadText) {
            uploadText.innerText = val;
            uploadText.style.color = '#374151';
        }
        
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerText = "已在服务器";
        }
        
        if (successMsg) successMsg.style.display = 'none';
        
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerText = "开始处理";
        }
        
        // Switch to New Task context
        const taskSelect = document.getElementById('task-select');
        if (taskSelect && taskSelect.value !== "") {
            taskSelect.value = "";
            onTaskSelectChange("");
        }
    }

    window.onload = function () {
      refreshTasks();
      loadUploadHistory();
      pollStatus();
      polling = setInterval(pollStatus, 2000);
      initSectionToggles();
      initAiModelSelector();
    };
  </script>
</body>
</html>
"""


def update_state(**kwargs):
    with state_lock:
        task_state.update(kwargs)


def get_latest_debug_png(debug_dir):
    latest = None
    latest_mtime = None
    for root, _, files in os.walk(debug_dir):
        for name in files:
            if not name.lower().endswith(".png"):
                continue
            path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
                latest = path
    return latest


def list_tasks():
    tasks = []
    current_id = None
    with state_lock:
        current_id = task_state.get("task_id")
    
    if not os.path.isdir(OUTPUT_FOLDER):
        return tasks
    
    for name in os.listdir(OUTPUT_FOLDER):
        dir_path = os.path.join(OUTPUT_FOLDER, name)
        if not os.path.isdir(dir_path):
            continue
        zip_path = os.path.join(OUTPUT_FOLDER, f"{name}.zip")
        meta_path = os.path.join(dir_path, "meta.json")
        display_name = name
        created_ts = None
        uploaded_filename = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                display_name = meta.get("name") or display_name
                created_ts = meta.get("created_at") or None
                uploaded_filename = meta.get("uploaded_filename")
            except Exception:
                uploaded_filename = None
        try:
            mtime = created_ts or os.path.getmtime(dir_path)
        except OSError:
            mtime = created_ts or 0
        
        zip_size_str = None
        if os.path.exists(zip_path):
            try:
                size_bytes = os.path.getsize(zip_path)
                if size_bytes < 1024 * 1024:
                    zip_size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    zip_size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            except OSError:
                pass

        tasks.append({
            "id": name,
            "name": display_name,
            "uploaded_filename": uploaded_filename,
            "mtime": mtime,
            "has_zip": os.path.exists(zip_path),
            "zip_size": zip_size_str,
            "is_current": (name == current_id),
        })
    
    tasks.sort(key=lambda t: t["mtime"], reverse=True)
    return tasks

def run_extraction(dom_path, output_dir, overrides, task_id=None, task_name=None):
    # ==========================
    # 1. Setup Logging & Print Config
    # ==========================
    log_path = os.path.join(output_dir, "console_log.txt") if output_dir else None
    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_capture.set_log_file(log_path)
        except Exception as e:
            print(f"[WARN] Failed to setup log file: {e}")

    print("\n" + "="*60)
    print(f"  任务启动 (Task Started)")
    print(f"  名称 (Name): {task_name or 'Unknown'}")
    print(f"  ID: {task_id}")
    print(f"  时间 (Time): {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录 (Output): {output_dir}")
    print("-" * 60)
    print("  [配置参数 Configuration]")
    if overrides:
        for k in sorted(overrides.keys()):
            print(f"    {k:<35}: {overrides[k]}")
    else:
        print("    (使用默认配置 / Default)")
    print("="*60 + "\n")

    try:
        for k, v in overrides.items():
            setattr(Config, k, v)
    except Exception:
        pass

    start_time = time.time()

    def wait_if_paused_or_stopped():
        while True:
            with state_lock:
                running = task_state.get("running")
                paused = task_state.get("pause_requested")
                stopping = task_state.get("stop_requested")
            if stopping:
                raise RuntimeError("Task stopped by user")
            if not running:
                return
            if not paused:
                return
            time.sleep(0.5)
    update_state(
        running=True,
        progress=0,
        step_progress=0,
        status="输入数据",
        output_path=None,
        debug_image=None,
        download_url=None,
        task_id=task_id,
        task_name=task_name,
        task_start_time=start_time,
        task_end_time=None,
        step_start_time=start_time,
        step_end_time=None,
        pause_requested=False,
        stop_requested=False
    )

    def update_task_meta_file(status_str, progress_val):
        if not output_dir: return
        meta_path = os.path.join(output_dir, "meta.json")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {}
            
            meta["status"] = status_str
            meta["progress"] = progress_val
            meta["end_time"] = time.time()
            if status_str == "completed":
                meta["step_progress"] = 100
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to update meta.json: {e}")

    try:
        # Check if input file exists
        if not dom_path or not os.path.exists(dom_path):
            raise FileNotFoundError(f"输入数据异常: 影像源文件缺失 {dom_path}")

        extractor = BuildingExtractor(dom_path, output_dir or None)
        
        # Step 1: Input Data (Preview)
        extractor.save_source_preview()

        def step_callback(p):
            wait_if_paused_or_stopped()
            update_state(step_progress=p * 100)

        # Step 2: AI Prediction
        update_state(status="AI预测 (AI Prediction)", progress=10, step_progress=0, step_start_time=time.time(), step_end_time=None)
        wait_if_paused_or_stopped()
        global_ai_mask = extractor.run_tiled_inference(progress_callback=step_callback)
        extractor._save_merged_ai_mask_debug(global_ai_mask)

        # Step 3: Area Filter
        update_state(status="面积过滤 (Area Filter)", progress=35, step_progress=0, step_start_time=time.time(), step_end_time=None)
        wait_if_paused_or_stopped()
        # Collect
        candidate_contours = extractor._collect_candidate_contours(global_ai_mask)
        # Filter
        filtered_polygons, _ = extractor._verify_contours(candidate_contours, progress_callback=step_callback)
        extractor.save_debug_polygons(filtered_polygons, "02_Area_Filtered")

        # Step 4: Hole Filling
        update_state(status="孔洞填充 (Hole Filling)", progress=50, step_progress=0, step_start_time=time.time(), step_end_time=None)
        wait_if_paused_or_stopped()
        filled_polygons = extractor._fill_polygon_holes(filtered_polygons, progress_callback=step_callback)
        extractor.save_debug_polygons(filled_polygons, "03_Holes_Filled")

        # Step 5: Smoothing (Before Simplify, as per user list order request? Or logic?)
        # Logic: Smooth (Buffer) -> Simplify (DP) usually gives cleaner results than Simplify -> Smooth.
        # User list: "平滑处理、简化处理" (Smooth, Simplify).
        update_state(status="平滑处理 (Smoothing)", progress=65, step_progress=0, step_start_time=time.time(), step_end_time=None)
        wait_if_paused_or_stopped()
        smoothed_polys = extractor._smooth_polygons(filled_polygons, progress_callback=step_callback)
        extractor.save_debug_polygons(smoothed_polys, "04_Smoothed")

        # Step 6: Simplification
        update_state(status="简化处理 (Simplification)", progress=80, step_progress=0, step_start_time=time.time(), step_end_time=None)
        wait_if_paused_or_stopped()
        simplified_polys = extractor._simplify_polygons(smoothed_polys, progress_callback=step_callback)
        extractor.save_debug_polygons(simplified_polys, "05_Simplified")

        # Step 7: Output Compression
        update_state(status="压缩输出 (Compression)", progress=95, step_progress=0, step_start_time=time.time(), step_end_time=None)
        wait_if_paused_or_stopped()
        
        output_path = extractor._write_shapefile(simplified_polys)

        debug_image = None
        if getattr(extractor, "debug_dir", None) and os.path.isdir(extractor.debug_dir):
            debug_image = get_latest_debug_png(extractor.debug_dir)
            
        # Zip the result for download
        download_url = None
        zip_size_str = None
        if task_id and output_dir and os.path.isdir(output_dir):
             base_name = os.path.join(OUTPUT_FOLDER, task_id)
             shutil.make_archive(base_name, 'zip', output_dir)
             download_url = f"/download/{task_id}.zip"
             
             zip_path = base_name + ".zip"
             if os.path.exists(zip_path):
                try:
                    size_bytes = os.path.getsize(zip_path)
                    if size_bytes < 1024 * 1024:
                        zip_size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        zip_size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                except OSError:
                    pass

        update_task_meta_file("completed", 100)

        update_state(
            status="完成",
            progress=100,
            step_progress=100,
            running=False,
            output_path=output_path,
            debug_image=debug_image,
            download_url=download_url,
            zip_size=zip_size_str,
            task_end_time=time.time(),
            step_end_time=time.time()
        )
    except Exception as e:
        if isinstance(e, RuntimeError) and "Task stopped by user" in str(e):
            update_task_meta_file("stopped", task_state.get("progress", 0))
            update_state(status="已停止", running=False, task_end_time=time.time(), step_end_time=time.time())
        else:
            import traceback
            traceback.print_exc()
            update_task_meta_file("failed", task_state.get("progress", 0))
            update_state(status=f"出错: {e}", running=False, task_end_time=time.time(), step_end_time=time.time())

    # ==========================
    # Cleanup Logging
    # ==========================
    if log_path:
        print(f"\n[Log End] Task finished at {datetime.datetime.now()}")
        # Allow a small delay for flush if needed, but not strictly required
        log_capture.set_log_file(None)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = safe_filename(file.filename)
        # Ensure filename is not empty (e.g. if filename was just special chars)
        if not filename:
            filename = "upload.tif"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        return jsonify({'filename': filename})
    return jsonify({'error': 'Upload failed'}), 500


@app.route('/uploads', methods=['GET'])
def list_uploads():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify([])
    try:
        # Filter for likely image extensions
        exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.img')
        files = []
        for f in os.listdir(UPLOAD_FOLDER):
            if f.lower().endswith(exts):
                path = os.path.join(UPLOAD_FOLDER, f)
                mtime = os.path.getmtime(path)
                files.append({'name': f, 'mtime': mtime})
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['mtime'], reverse=True)
        return jsonify(files)
    except Exception as e:
        print(f"Error listing uploads: {e}")
        return jsonify([])


@app.route('/uploads/exists', methods=['GET'])
def check_upload_exists():
    filename = request.args.get('filename', '').strip()
    if not filename:
        return jsonify({'exists': False})
    
    path = os.path.join(UPLOAD_FOLDER, filename)
    exists = os.path.exists(path) and os.path.isfile(path)
    return jsonify({'exists': exists})


@app.route('/uploads/clear', methods=['POST'])
def clear_uploads():
    removed = 0
    if os.path.isdir(UPLOAD_FOLDER):
        for name in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, name)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                removed += 1
            except Exception:
                pass
    return jsonify({'ok': True, 'removed': removed})


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/start", methods=["POST"])
def start():
    with state_lock:
        if task_state.get("running"):
            return jsonify({"ok": False, "error": "已有任务在运行"}), 400

    uploaded_filename = request.form.get("uploaded_filename", "").strip()
    
    if not uploaded_filename:
        return jsonify({"ok": False, "error": "请先上传影像文件"}), 400
        
    dom_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
    if not os.path.exists(dom_path):
         return jsonify({"ok": False, "error": "找不到上传的文件"}), 400

    # Auto-generate output directory based on task ID
    task_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use user provided task name or default to filename
    task_name = request.form.get("task_name", "").strip()
    if not task_name:
        task_name = os.path.splitext(os.path.basename(uploaded_filename))[0]
        
    output_dir = os.path.join(OUTPUT_FOLDER, task_id)
    os.makedirs(output_dir, exist_ok=True)

    # Collect configuration from form
    config_data = {}
    # List of fields to capture
    fields = [
        "BLOCK_SIZE", "BLOCK_OVERLAP", "AI_IMG_SIZE", "AI_DEVICE", "AI_MODEL_PATH",
        "ENABLE_GLOBAL_FILL", "ENABLE_FILTER_AREA", "MIN_AREA_M2",
        "ENABLE_SMOOTHING", "BUFFER_EXPAND_M", "BUFFER_SHRINK_M",
        "ENABLE_SIMPLIFY", "SIMPLIFY_TOL", "RECT_IOU_THRESH"
    ]
    for field in fields:
        val = request.form.get(field)
        if val is not None:
            # Store checkboxes as boolean
            if field.startswith("ENABLE_"):
                config_data[field] = (val == "on")
            else:
                config_data[field] = val
    # Handle checkboxes that are unchecked (not sent in form)
    for field in fields:
        if field.startswith("ENABLE_") and field not in config_data:
            config_data[field] = False

    meta = {
        "id": task_id,
        "name": task_name,
        "uploaded_filename": uploaded_filename,
        "created_at": time.time(),
        "config": config_data,
        "status": "pending" # Initial status
    }
    try:
        with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    def to_bool(field, default):
        v = request.form.get(field)
        if v is None:
            return default
        return v.lower() in ("true", "1", "on", "yes")

    def to_float(field, default):
        v = request.form.get(field, "").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            return default
    def to_int(field, default):
        v = request.form.get(field, "").strip()
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            try:
                return int(float(v))
            except ValueError:
                return default

    overrides = {}

    overrides["BLOCK_SIZE"] = to_int("BLOCK_SIZE", Config.BLOCK_SIZE)
    overrides["BLOCK_OVERLAP"] = to_int("BLOCK_OVERLAP", Config.BLOCK_OVERLAP)
    overrides["AI_MODEL_PATH"] = request.form.get("AI_MODEL_PATH", Config.AI_MODEL_PATH)
    overrides["AI_DEVICE"] = request.form.get("AI_DEVICE", Config.AI_DEVICE)
    overrides["AI_IMG_SIZE"] = to_int("AI_IMG_SIZE", getattr(Config, "AI_IMG_SIZE", 2048))

    overrides["ENABLE_GLOBAL_FILL"] = to_bool("ENABLE_GLOBAL_FILL", False)

    overrides["ENABLE_FILTER_AREA"] = to_bool("ENABLE_FILTER_AREA", False)
    overrides["MIN_AREA_M2"] = to_float("MIN_AREA_M2", Config.MIN_AREA_M2)
    
    overrides["ENABLE_SMOOTHING"] = to_bool("ENABLE_SMOOTHING", False)
    overrides["BUFFER_EXPAND_M"] = to_float("BUFFER_EXPAND_M", Config.BUFFER_EXPAND_M)
    overrides["BUFFER_SHRINK_M"] = to_float("BUFFER_SHRINK_M", Config.BUFFER_SHRINK_M)
    overrides["ENABLE_SIMPLIFY"] = to_bool("ENABLE_SIMPLIFY", False)
    overrides["SIMPLIFY_TOL"] = to_float("SIMPLIFY_TOL", Config.SIMPLIFY_TOL)
    overrides["RECT_IOU_THRESH"] = to_float("RECT_IOU_THRESH", Config.RECT_IOU_THRESH)

    overrides["DEBUG_MODE"] = to_bool("DEBUG_MODE", Config.DEBUG_MODE)
    overrides["DEBUG_DIR"] = request.form.get("DEBUG_DIR", Config.DEBUG_DIR)

    t = threading.Thread(target=run_extraction, args=(dom_path, output_dir, overrides, task_id, task_name), daemon=True)
    t.start()

    return jsonify({"ok": True, "message": "任务已启动"})


@app.route("/status")
def status():
    with state_lock:
        data = dict(task_state)
    task_start = data.get("task_start_time")
    if task_start:
        task_end = data.get("task_end_time") or time.time()
        data["task_elapsed"] = max(0.0, task_end - task_start)
    step_start = data.get("step_start_time")
    if step_start:
        step_end = data.get("step_end_time") or time.time()
        data["step_elapsed"] = max(0.0, step_end - step_start)
    return jsonify(data)


@app.route("/tasks", methods=["GET"])
def tasks_list():
    return jsonify(list_tasks())


@app.route("/tasks/clear", methods=["POST"])
def tasks_clear():
    with state_lock:
        current_id = task_state.get("task_id")

    removed = 0
    if os.path.isdir(OUTPUT_FOLDER):
        for name in os.listdir(OUTPUT_FOLDER):
            path = os.path.join(OUTPUT_FOLDER, name)
            if os.path.isdir(path) and name != current_id:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
                except Exception:
                    pass

        for name in os.listdir(OUTPUT_FOLDER):
            if not name.lower().endswith(".zip"):
                continue
            base = os.path.splitext(name)[0]
            if base == current_id:
                continue
            zip_path = os.path.join(OUTPUT_FOLDER, name)
            try:
                os.remove(zip_path)
            except OSError:
                pass

    return jsonify({"ok": True, "removed": removed})


@app.route("/control", methods=["POST"])
def control():
    payload = request.get_json(silent=True) or {}
    action = (payload.get("action") or "").lower()
    with state_lock:
        running = task_state.get("running")
        if action == "pause":
            if running:
                task_state["pause_requested"] = True
        elif action == "resume":
            task_state["pause_requested"] = False
        elif action == "stop":
            if running:
                task_state["stop_requested"] = True
    return jsonify({"ok": True})


@app.route("/debug_image")
def debug_image():
    ts = request.args.get("ts", "")
    with state_lock:
        path = task_state.get("debug_image")
    
    if path and os.path.exists(path):
        return send_from_directory(os.path.dirname(path), os.path.basename(path))
    
    return "No debug image", 404


@app.route("/debug_gallery")
def debug_gallery():
    tid = request.args.get("task_id")
    if not tid:
        with state_lock:
            tid = task_state.get("task_id")
    
    if not tid:
        return jsonify([])

    debug_dir = os.path.join(OUTPUT_FOLDER, tid, "debug")
    if not os.path.isdir(debug_dir):
        return jsonify([])

    # Define description map
    desc_map = {
        "00_Input_Preview": "输入数据 (Input Data)",
        "01_Global_AI_Raw": "AI预测 (AI Prediction)",
        "02_Area_Filtered": "面积过滤 (Area Filter)",
        "03_Holes_Filled": "孔洞填充 (Hole Filling)",
        "04_Smoothed": "平滑处理 (Smoothing)",
        "05_Simplified": "简化处理 (Simplification)"
    }

    files = []
    try:
        for f in os.listdir(debug_dir):
            if f.endswith(".png") and not f.startswith("block_"):
                # Check if it matches our interested steps
                if f[0].isdigit():
                    name_no_ext = os.path.splitext(f)[0]
                    desc = desc_map.get(name_no_ext, name_no_ext)
                    
                    # Add timestamp to URL to prevent browser caching of partial/old images
                    file_path = os.path.join(debug_dir, f)
                    mtime = int(os.path.getmtime(file_path))
                    
                    files.append({
                        "name": f,
                        "url": f"/debug_file/{tid}/{f}?v={mtime}",
                        "desc": desc
                    })
    except Exception:
        pass

    files.sort(key=lambda x: x["name"])
    return jsonify(files)


@app.route("/debug_file/<task_id>/<filename>")
def serve_debug_file(task_id, filename):
    # Security check: ensure task_id is valid (alphanumeric or underscores)
    if not task_id.replace('_', '').isalnum():
         return "Invalid task id", 400
    
    debug_dir = os.path.join(OUTPUT_FOLDER, task_id, "debug")
    return send_from_directory(debug_dir, filename)


@app.route("/task_info")
def get_task_info():
    task_id = request.args.get("task_id")
    if not task_id:
        return jsonify({"error": "Missing task_id"}), 400
        
    # Security check
    if not task_id.replace('_', '').isalnum():
         return jsonify({"error": "Invalid task id"}), 400
         
    task_dir = os.path.join(OUTPUT_FOLDER, task_id)
    if not os.path.isdir(task_dir):
        return jsonify({"error": "Task not found"}), 404
        
    info = {
        "meta": {},
        "logs": ""
    }
    
    # Read meta.json
    meta_path = os.path.join(task_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                info["meta"] = json.load(f)
        except Exception:
            pass
            
    # Read console_log.txt
    log_path = os.path.join(task_dir, "console_log.txt")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                info["logs"] = f.read()
        except Exception:
            pass
            
    return jsonify(info)


@app.route("/logs")
def get_logs():
    return jsonify({"logs": log_capture.get_content()})


if __name__ == "__main__":
    print("Starting Web UI at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
