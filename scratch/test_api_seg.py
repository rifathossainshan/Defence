import urllib.request
import json
import base64
import os

def main():
    # Calling the new overlay endpoint for BraTS-GLI-02073-101
    url = "http://127.0.0.1:5000/api/slice?patient_id=BraTS-GLI-02073-101&modality=seg&plane=coronal&slice_pct=0.5"
    print(f"Requesting: {url}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            print(f"Status Code: {status}")
            if status == 200:
                data = json.loads(response.read().decode('utf-8'))
                img_data = data["image"]
                header, encoded = img_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                
                output_dir = "scratch"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "overlay_test.png")
                with open(output_path, "wb") as f:
                    f.write(img_bytes)
                print(f"SUCCESS: Saved slice image to {output_path} ({len(img_bytes)} bytes)")
            else:
                print(f"FAILED: Status {status}")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
