from time import time

from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from io import BytesIO
import mimetypes
import os

from new_pipeline import process_and_stabilize

# Register HEIF/HEIC support — must happen before any Image.open() calls
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("pillow-heif loaded: HEIC/HEIF supported")
except ImportError:
    print("WARNING: pillow-heif not installed. HEIC files will fail. Run: pip install pillow-heif")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
# output_folder = 'output
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'heic', 'heif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

# MIME type → extension fallback for when iOS sends no filename/extension
MIME_TO_EXT = {
    'image/heic': '.heic',
    'video/quicktime': '.mov',
    'image/heif': '.heif',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
}

def allowed_file(filename):
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS or extension in ALLOWED_VIDEO_EXTENSIONS

def get_save_filename(filename, mimetype):
    """Return a safe filename, inferring extension from MIME type if missing."""
    if filename and filename != '' and '.' in filename:
        return secure_filename(filename)
    # No extension — guess from MIME type
    ext = MIME_TO_EXT.get(mimetype) or mimetypes.guess_extension(mimetype) or '.bin'
    return f"upload{ext}"

def log_image_info_and_save(image, source=""):
    """Log image details without calling image.show() (which needs a GUI)."""
    print(f"  Source:  {source}")
    print(f"  Format:  {image.format}")
    print(f"  Mode:    {image.mode}")
    print(f"  Size:    {image.size}")
    image.show()


def build_output_csv_path(filename):
    base_name, _ = os.path.splitext(filename)
    return os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_tracking.csv")


def is_video_upload(filename, mimetype):
    if mimetype and mimetype.startswith('video/'):
        return True
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_VIDEO_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "POST an image/video to /upload"})

@app.route('/upload', methods=['POST'])
def upload_file():
    print("--- /upload request received ---")
    print(f"  Content-Type:   {request.content_type}")
    print(f"  Content-Length: {request.content_length}")
    print(f"  Form keys:      {list(request.form.keys())}")
    print(f"  File keys:      {list(request.files.keys())}")
    filename = request.headers.get('filename')
    print(f"  Filename header: {filename}")
    try:
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename:
            raw_data = uploaded_file.read()
            mimetype = uploaded_file.mimetype or ''
            # filename = get_save_filename(uploaded_file.filename, mimetype)
        else:
            # Raw body upload where Content-Type is the MIME type directly.
            raw_data = request.get_data(cache=True)
            if not raw_data:
                print("  No file field and no raw body.")
                return jsonify({"error": "No data provided"}), 400

            mimetype = request.content_type.split(';')[0].strip() if request.content_type else ''
            # filename = get_save_filename('', mimetype)

        print(f"  Upload: {len(raw_data)} bytes, MIME: {mimetype}, filename: {filename}")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(raw_data)
        print(f"  Saved to: {filepath}")

        if is_video_upload(filename, mimetype):
            output_csv = build_output_csv_path(filename)
            pipeline_result = process_and_stabilize(filepath, output_dir=os.path.join(app.config['OUTPUT_FOLDER'], "results_" + time.strftime("%Y%m%d-%H%M%S"))
, smooth_radius=30)
            return 
        jsonify({
                "message": "Video received and processed successfully",
                # "filename": filename,
                # "size_bytes": len(raw_data),
                # "output_csv": pipeline_result["csv_path"],
                # "output_video": pipeline_result["video_path"],
                # "frames_visualized": pipeline_result["frames_visualized"],
                # "tracking_points": pipeline_result["tracking_points"],
            }), 200

        try:
            with Image.open(BytesIO(raw_data)) as image:
                log_image_info_and_save(image, source="raw body")
        except Exception as img_err:
            print(f"  (Not an image or unsupported format: {img_err})")

        return jsonify({
            "message": "File received successfully",
            "filename": filename,
            "size_bytes": len(raw_data),
        }), 200

    except Exception as e:
        print(f"  ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
