"""
Hugging Face Spaces API wrapper
Provides direct JSON responses without job queues
"""
from flask import Flask, request, jsonify
from inference_core import run_pipeline_for_image, download_image_from_url, upload_to_cloudinary
import os

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """Home page with API documentation"""
    return jsonify({
        "service": "Anomaly Detection API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/infer": "POST - Run inference on image URL"
        },
        "example_request": {
            "method": "POST",
            "url": "/infer",
            "body": {
                "image_url": "https://example.com/image.jpg"
            }
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/infer", methods=["POST"])
def infer():
    """
    Inference endpoint - returns direct JSON response
    Request JSON: {"image_url": "https://..."}
    """
    try:
        data = request.get_json()
        if not data or "image_url" not in data:
            return jsonify({"error": "Missing image_url"}), 400
        
        image_url = data["image_url"]
        
        # Download image
        local_path = download_image_from_url(image_url)
        
        # Run pipeline
        results = run_pipeline_for_image(local_path)
        
        # Upload outputs
        boxed_url = upload_to_cloudinary(results["boxed_path"], folder="pipeline_outputs") if results["boxed_path"] else None
        mask_url = upload_to_cloudinary(results["mask_path"], folder="pipeline_outputs") if results["mask_path"] else None
        filtered_url = upload_to_cloudinary(results["filtered_path"], folder="pipeline_outputs") if results["filtered_path"] else None
        
        # Clean up
        if os.path.exists(local_path):
            os.remove(local_path)
        
        # Direct JSON response (no job queue wrapper)
        return jsonify({
            "label": results["label"],
            "boxed_url": boxed_url,
            "mask_url": mask_url,
            "filtered_url": filtered_url,
            "boxes": results.get("boxes", [])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For Hugging Face Spaces, use port 7860
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
