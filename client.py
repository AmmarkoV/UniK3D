import os
import sys
import time
import json
import shutil
import tempfile
import argparse
import cv2
from gradio_client import Client, handle_file

# Private per-process download dir. gradio_client otherwise pools downloads in the SHARED
# /tmp/gradio used by every other gradio client AND server on the machine — cleaning that
# would race with their in-flight files. We isolate ours and only clean our own dir.
GRADIO_DL_DIR = tempfile.mkdtemp(prefix="unik3d_client_")

def wipe_dir(d):
    """Remove everything inside our private download dir (this process owns it exclusively)."""
    for name in os.listdir(d):
        p = os.path.join(d, name)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

def load_image_files(input_path):
    files = []
    if os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if is_image_file(f):
                files.append(os.path.join(input_path, f))
    elif os.path.isfile(input_path) and is_image_file(input_path):
        files = [input_path]
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    return files

def main():
    parser = argparse.ArgumentParser(description="Gradio depth and distance map client.")
    parser.add_argument("input", help="Path to image file or directory of images.")
    parser.add_argument("--ip", default="127.0.0.1", help="IP address of the Gradio server.")
    parser.add_argument("--port", default="7860", help="Port of the Gradio server.")
    parser.add_argument("--output-dir", default="output_images", help="Directory to save output images.")
    parser.add_argument("--output-json", default="output.json", help="Path to save JSON results.")
    parser.add_argument("--start", type=int, default=0, help="Index to start processing from (for resume).")
    parser.add_argument("--dist-min", type=float, default=0.0, help="Minimum distance value for normalization.")
    parser.add_argument("--dist-max", type=float, default=20.0, help="Maximum distance value for normalization.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect images
    files = load_image_files(args.input)
    if not files:
        print("No valid images found.")
        return

    print(f"Connecting to http://{args.ip}:{args.port}")
    client = Client(f"http://{args.ip}:{args.port}", download_files=GRADIO_DL_DIR)

    results = {}
    for i, image_path in enumerate(files[args.start:], start=args.start):
        print(f"\n[{i+1}/{len(files)}] Processing: {image_path}")
        try:
            start_time = time.time()

            # Call Gradio endpoint (assumes default api_name if not custom)
            depth_img, distance_img = client.predict(
                handle_file(image_path),
                args.dist_min,
                args.dist_max,
                api_name="/predict"  # default Gradio endpoint
            )

            # Save images
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            depth_path = os.path.join(args.output_dir, f"{base_name}_depth.png")
            distance_path = os.path.join(args.output_dir, f"{base_name}_distance.png")

            cv2.imwrite(depth_path, cv2.imread(depth_img))
            cv2.imwrite(distance_path, cv2.imread(distance_img))
            wipe_dir(GRADIO_DL_DIR)   # free our private gradio download cache

            # Store results
            results[image_path] = {
                "depth_map": depth_path,
                "distance_map": distance_path
            }

            elapsed = time.time() - start_time
            print(f"✓ Done in {elapsed:.2f}s | Saved depth & distance maps.")

        except Exception as e:
            print(f"✗ Failed on {image_path}: {e}")
            continue

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll results saved to {args.output_json}")

if __name__ == "__main__":
    main()

