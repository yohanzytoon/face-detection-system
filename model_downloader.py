import os
import sys
import urllib.request

def download_file(url, file_path):
    """Download a file from URL to specified path"""
    print(f"Downloading {url} to {file_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Download with progress indicator
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            s = f"\rDownloading: {percent:.1f}% ({read_so_far} / {total_size} bytes)"
            sys.stdout.write(s)
            sys.stdout.flush()
        
    try:
        urllib.request.urlretrieve(url, file_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def download_models():
    """Download necessary model files for face detection"""
    
    # DNN face detection models
    models = [
        {
            "name": "DNN Prototxt",
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "path": "models/deploy.prototxt"
        },
        {
            "name": "DNN Caffemodel",
            "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "path": "models/res10_300x300_ssd_iter_140000.caffemodel"
        },
        {
            "name": "Haar Cascade",
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            "path": "models/haarcascade_frontalface_default.xml"
        }
    ]
    
    # Download each model
    success = True
    for model in models:
        if not os.path.exists(model["path"]):
            print(f"Downloading {model['name']}...")
            if not download_file(model["url"], model["path"]):
                success = False
                print(f"Failed to download {model['name']}. Face detection may not work correctly.")
        else:
            print(f"{model['name']} already exists at {model['path']}")
    
    if success:
        print("\nAll model files downloaded successfully!")
    
    return success

if __name__ == "__main__":
    print("=== Model Download Helper ===")
    download_models()