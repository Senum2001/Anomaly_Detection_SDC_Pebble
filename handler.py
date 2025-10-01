"""
RunPod serverless handler wrapper for the anomaly detection pipeline.
This file acts as the entry point for RunPod serverless execution.
"""
import runpod
from pipeline import run_pipeline_for_image, download_image_from_url, upload_to_cloudinary
import os


def handler(job):
    """
    RunPod handler function that processes inference requests.
    
    Args:
        job: Dictionary containing job_input with image_url
        
    Returns:
        Dictionary with inference results
    """
    try:
        job_input = job["input"]
        image_url = job_input.get("image_url")
        
        if not image_url:
            return {"error": "Missing image_url in input"}
        
        # Download image
        local_path = download_image_from_url(image_url)
        
        # Run inference pipeline
        results = run_pipeline_for_image(local_path)
        
        # Upload outputs to Cloudinary
        boxed_url = upload_to_cloudinary(results["boxed_path"], folder="pipeline_outputs") if results["boxed_path"] else None
        mask_url = upload_to_cloudinary(results["mask_path"], folder="pipeline_outputs") if results["mask_path"] else None
        filtered_url = upload_to_cloudinary(results["filtered_path"], folder="pipeline_outputs") if results["filtered_path"] else None
        
        # Clean up local file
        if os.path.exists(local_path):
            os.remove(local_path)
        
        # Return results
        return {
            "label": results["label"],
            "boxed_url": boxed_url,
            "mask_url": mask_url,
            "filtered_url": filtered_url,
            "boxes": results.get("boxes", [])
        }
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
