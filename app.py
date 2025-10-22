"""
Hugging Face Spaces API wrapper
Provides direct JSON responses without job queues
Integrated with feedback learning pipeline for continuous model improvement
"""
from flask import Flask, request, jsonify
from inference_core import run_pipeline_for_image, download_image_from_url, upload_to_cloudinary, model, device
from scripts.feedback_learning_pipeline import initialize_feedback_pipeline, run_feedback_training
from scripts.model_versioning import initialize_model_tracker
import os

app = Flask(__name__)

# Initialize feedback learning pipeline
feedback_pipeline = initialize_feedback_pipeline(model, device)

# Initialize model versioning tracker
model_tracker = initialize_model_tracker()


@app.route("/", methods=["GET"])
def home():
    """Home page with API documentation"""
    return jsonify({
        "service": "Anomaly Detection API with Feedback Learning",
        "version": "2.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/infer": "POST - Run inference on image URL",
            "/feedback/stats": "GET - Get feedback statistics and training status",
            "/feedback/train": "POST - Manually trigger feedback training cycle",
            "/model/current": "GET - Get current model version and parameters",
            "/model/versions": "GET - Get model version history",
            "/model/training-history": "GET - Get training cycle history",
            "/model/compare": "POST - Compare model versions"
        },
        "example_request": {
            "method": "POST",
            "url": "/infer",
            "body": {
                "image_url": "https://example.com/image.jpg"
            }
        },
        "feedback_info": {
            "description": "User corrections are automatically fetched from Supabase",
            "training_trigger": "Automatic when 10+ new feedback samples available",
            "manual_training": "POST /feedback/train to trigger immediately"
        },
        "versioning_info": {
            "description": "Model versions and training history tracked automatically",
            "view_current": "GET /model/current to see active model parameters",
            "view_history": "GET /model/versions to see all versions"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """
    Get feedback statistics and training status
    """
    try:
        stats = feedback_pipeline.get_feedback_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/train", methods=["POST"])
def trigger_training():
    """
    Manually trigger a feedback training cycle
    Fetches user corrections from Supabase and improves model
    """
    try:
        results = run_feedback_training(feedback_pipeline)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/current", methods=["GET"])
def get_current_model():
    """
    Get current active model version and parameters
    """
    try:
        current_state = model_tracker.get_current_model_state()
        return jsonify(current_state), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/versions", methods=["GET"])
def get_model_versions():
    """
    Get model version history
    Query params: limit (default: 20)
    """
    try:
        limit = int(request.args.get('limit', 20))
        versions = model_tracker.get_version_history(limit=limit)
        return jsonify({
            "total": len(versions),
            "versions": versions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/training-history", methods=["GET"])
def get_training_history():
    """
    Get training cycle history
    Query params: limit (default: 20)
    """
    try:
        limit = int(request.args.get('limit', 20))
        history = model_tracker.get_training_history(limit=limit)
        return jsonify({
            "total": len(history),
            "training_cycles": history
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/compare", methods=["POST"])
def compare_versions():
    """
    Compare multiple model versions
    Request JSON: {"version_ids": ["id1", "id2", ...]}
    """
    try:
        data = request.get_json()
        if not data or "version_ids" not in data:
            return jsonify({"error": "Missing version_ids"}), 400
        
        version_ids = data["version_ids"]
        comparison = model_tracker.generate_comparison_table(version_ids)
        
        return jsonify({
            "comparison": comparison,
            "version_count": len(version_ids)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
