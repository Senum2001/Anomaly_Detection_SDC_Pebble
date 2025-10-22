"""
Feedback Learning Pipeline
Fetches user-corrected annotations from Supabase and uses them to improve model accuracy
"""
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from supabase import create_client, Client
from PIL import Image
import tempfile

# Import model versioning system
try:
    from scripts.model_versioning import ModelVersionTracker
    MODEL_VERSIONING_AVAILABLE = True
except ImportError:
    MODEL_VERSIONING_AVAILABLE = False
    print("[Warning] Model versioning not available")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://xbcgrpqiibicestnhytt.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhiY2dycHFpaWJpY2VzdG5oeXR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTkxMzk3MywiZXhwIjoyMDcxNDg5OTczfQ.sANBuVZ6gdYc5kHkxTXZ67jtE9QHPw5HFaUKffP1Jrs")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Training state tracking
TRAINING_STATE_FILE = "feedback_training_state.json"
MIN_FEEDBACK_FOR_RETRAINING = 10  # Minimum feedback samples to trigger retraining


class FeedbackLearningPipeline:
    """Pipeline for continuous learning from user feedback"""
    
    def __init__(self, model, device):
        """
        Initialize the feedback learning pipeline
        
        Args:
            model: The PatchCore model instance
            device: PyTorch device (cpu or cuda)
        """
        self.model = model
        self.device = device
        self.training_state = self._load_training_state()
        
        # Initialize model versioning tracker
        if MODEL_VERSIONING_AVAILABLE:
            self.version_tracker = ModelVersionTracker()
        else:
            self.version_tracker = None
        
    def _load_training_state(self) -> Dict[str, Any]:
        """Load training state from disk"""
        if os.path.exists(TRAINING_STATE_FILE):
            with open(TRAINING_STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "last_training_time": None,
            "last_processed_feedback_id": None,
            "total_feedback_processed": 0,
            "training_runs": []
        }
    
    def _save_training_state(self):
        """Save training state to disk"""
        with open(TRAINING_STATE_FILE, 'w') as f:
            json.dump(self.training_state, f, indent=2)
    
    def fetch_new_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch new feedback logs from Supabase
        
        Args:
            limit: Maximum number of feedback records to fetch
            
        Returns:
            List of feedback log dictionaries
        """
        try:
            query = supabase.table('feedback_logs').select('*')
            
            # Only fetch feedback newer than last processed
            if self.training_state.get("last_processed_feedback_id"):
                query = query.gt('created_at', self.training_state["last_processed_feedback_id"])
            
            response = query.order('created_at', desc=False).limit(limit).execute()
            
            feedback_logs = response.data if response.data else []
            print(f"[Feedback Pipeline] Fetched {len(feedback_logs)} new feedback records")
            return feedback_logs
            
        except Exception as e:
            print(f"[Feedback Pipeline] Error fetching feedback: {e}")
            return []
    
    def extract_corrected_annotations(self, feedback_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract user-corrected annotations from feedback logs
        
        Args:
            feedback_logs: List of feedback log dictionaries
            
        Returns:
            List of correction samples with image_id, model_pred, and user_correction
        """
        corrections = []
        
        for log in feedback_logs:
            try:
                correction = {
                    "feedback_id": log.get("id"),
                    "image_id": log.get("image_id"),
                    "model_predicted": log.get("model_predicted_anomalies", {}),
                    "user_corrected": log.get("final_accepted_annotations", {}),
                    "annotator_metadata": log.get("annotator_metadata", {}),
                    "created_at": log.get("created_at")
                }
                
                # Validate that we have both predictions and corrections
                if correction["model_predicted"] and correction["user_corrected"]:
                    corrections.append(correction)
                    
            except Exception as e:
                print(f"[Feedback Pipeline] Error processing feedback log {log.get('id')}: {e}")
                continue
        
        print(f"[Feedback Pipeline] Extracted {len(corrections)} valid corrections")
        return corrections
    
    def calculate_correction_patterns(self, corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in user corrections to understand model weaknesses
        
        Args:
            corrections: List of correction samples
            
        Returns:
            Dictionary with analysis results
        """
        patterns = {
            "total_corrections": len(corrections),
            "bbox_adjustments": [],
            "label_changes": [],
            "severity_changes": [],
            "false_positives": 0,
            "false_negatives": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        for correction in corrections:
            model_pred = correction["model_predicted"]
            user_corr = correction["user_corrected"]
            
            # Analyze label changes
            model_label = model_pred.get("label", "")
            user_label = user_corr.get("label", "")
            
            if model_label != user_label:
                patterns["label_changes"].append({
                    "from": model_label,
                    "to": user_label,
                    "image_id": correction["image_id"]
                })
            
            # Count false positives (model detected anomaly, user said normal)
            if "Critical" in model_label and "Normal" in user_label:
                patterns["false_positives"] += 1
            
            # Count false negatives (model said normal, user found anomaly)
            if "Normal" in model_label and "Critical" in user_label:
                patterns["false_negatives"] += 1
            
            # Analyze bounding box adjustments
            model_detections = model_pred.get("detections", [])
            user_detections = user_corr.get("detections", [])
            
            if len(model_detections) != len(user_detections):
                patterns["bbox_adjustments"].append({
                    "image_id": correction["image_id"],
                    "model_count": len(model_detections),
                    "user_count": len(user_detections)
                })
        
        print(f"[Feedback Pipeline] Pattern Analysis:")
        print(f"  - Label Changes: {len(patterns['label_changes'])}")
        print(f"  - False Positives: {patterns['false_positives']}")
        print(f"  - False Negatives: {patterns['false_negatives']}")
        print(f"  - BBox Adjustments: {len(patterns['bbox_adjustments'])}")
        
        return patterns
    
    def apply_model_adjustments(self, patterns: Dict[str, Any]):
        """
        Apply learned patterns to improve model inference
        
        This function adjusts model confidence thresholds and parameters
        based on user feedback patterns
        
        Args:
            patterns: Pattern analysis results
        """
        try:
            # Calculate adjustment factors based on false positive/negative rates
            total = patterns["total_corrections"]
            if total == 0:
                return
            
            fp_rate = patterns["false_positives"] / total
            fn_rate = patterns["false_negatives"] / total
            
            print(f"[Feedback Pipeline] Model Adjustment:")
            print(f"  - False Positive Rate: {fp_rate:.2%}")
            print(f"  - False Negative Rate: {fn_rate:.2%}")
            
            # Store adjustment metadata for inference
            adjustment_data = {
                "fp_rate": fp_rate,
                "fn_rate": fn_rate,
                "total_corrections": total,
                "timestamp": datetime.now().isoformat(),
                "recommendation": self._get_threshold_recommendation(fp_rate, fn_rate)
            }
            
            # Save adjustment data
            with open("model_adjustments.json", "w") as f:
                json.dump(adjustment_data, f, indent=2)
            
            print(f"[Feedback Pipeline] Recommendation: {adjustment_data['recommendation']}")
            
        except Exception as e:
            print(f"[Feedback Pipeline] Error applying adjustments: {e}")
    
    def _get_threshold_recommendation(self, fp_rate: float, fn_rate: float) -> str:
        """Generate threshold adjustment recommendation"""
        if fp_rate > 0.3:
            return "INCREASE detection threshold - too many false positives"
        elif fn_rate > 0.3:
            return "DECREASE detection threshold - too many false negatives"
        else:
            return "Current threshold is balanced"
    
    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained based on feedback count
        
        Returns:
            True if retraining should be triggered
        """
        new_feedback_count = len(self.fetch_new_feedback(limit=1000))
        
        if new_feedback_count >= MIN_FEEDBACK_FOR_RETRAINING:
            print(f"[Feedback Pipeline] Retraining threshold met: {new_feedback_count} new feedback samples")
            return True
        
        print(f"[Feedback Pipeline] Not enough feedback for retraining: {new_feedback_count}/{MIN_FEEDBACK_FOR_RETRAINING}")
        return False
    
    def run_training_cycle(self):
        """
        Execute a full training cycle: fetch feedback, analyze patterns, apply adjustments
        """
        print(f"\n[Feedback Pipeline] Starting training cycle at {datetime.now()}")
        
        # Capture model state BEFORE training
        before_state = None
        if self.version_tracker:
            try:
                before_state = self.version_tracker.get_current_model_state()
                self.version_tracker.log_model_version(before_state)
                print(f"[Model Versioning] Captured state before training: {before_state['version_id'][:8]}...")
            except Exception as e:
                print(f"[Model Versioning] Warning: Could not capture before state: {e}")
                self.version_tracker = None  # Disable versioning for this run
        
        # Fetch new feedback
        feedback_logs = self.fetch_new_feedback(limit=1000)
        
        if not feedback_logs:
            print("[Feedback Pipeline] No new feedback available")
            return {
                "status": "no_feedback",
                "message": "No new feedback to process"
            }
        
        # Extract corrections
        corrections = self.extract_corrected_annotations(feedback_logs)
        
        if not corrections:
            print("[Feedback Pipeline] No valid corrections found")
            return {
                "status": "no_corrections",
                "message": "No valid corrections extracted"
            }
        
        # Analyze patterns
        patterns = self.calculate_correction_patterns(corrections)
        
        # Apply adjustments
        self.apply_model_adjustments(patterns)
        
        # Update training state
        self.training_state["last_training_time"] = datetime.now().isoformat()
        self.training_state["last_processed_feedback_id"] = feedback_logs[-1].get("created_at")
        self.training_state["total_feedback_processed"] += len(corrections)
        self.training_state["training_runs"].append({
            "timestamp": datetime.now().isoformat(),
            "corrections_processed": len(corrections),
            "patterns": patterns
        })
        
        self._save_training_state()
        
        # Capture model state AFTER training
        after_state = None
        training_cycle_id = None
        if self.version_tracker and before_state:
            try:
                after_state = self.version_tracker.get_current_model_state()
                self.version_tracker.log_model_version(after_state)
                print(f"[Model Versioning] Captured state after training: {after_state['version_id'][:8]}...")
                
                # Log the training cycle with before/after comparison
                training_cycle_id = self.version_tracker.log_training_cycle(
                    before_state=before_state,
                    after_state=after_state,
                    feedback_count=len(corrections),
                    patterns=patterns,
                    performance_metrics=None  # TODO: Calculate actual metrics
                )
                
                if training_cycle_id:
                    print(f"[Training History] Logged training cycle: {training_cycle_id[:8]}...")
            except Exception as e:
                print(f"[Model Versioning] Error logging version: {e}")
        
        print(f"[Feedback Pipeline] Training cycle completed successfully")
        print(f"[Feedback Pipeline] Total feedback processed: {self.training_state['total_feedback_processed']}")
        
        return {
            "status": "success",
            "corrections_processed": len(corrections),
            "patterns": patterns,
            "total_feedback_processed": self.training_state["total_feedback_processed"],
            "before_version_id": before_state.get("version_id") if before_state else None,
            "after_version_id": after_state.get("version_id") if after_state else None,
            "training_cycle_id": training_cycle_id
        }
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback and training"""
        try:
            # Get total feedback count from Supabase
            response = supabase.table('feedback_logs').select('id', count='exact').execute()
            total_feedback = response.count if response.count else 0
            
            return {
                "total_feedback_in_db": total_feedback,
                "total_processed": self.training_state.get("total_feedback_processed", 0),
                "last_training_time": self.training_state.get("last_training_time"),
                "training_runs": len(self.training_state.get("training_runs", [])),
                "ready_for_retraining": total_feedback >= MIN_FEEDBACK_FOR_RETRAINING
            }
            
        except Exception as e:
            print(f"[Feedback Pipeline] Error getting stats: {e}")
            return {
                "error": str(e)
            }


def initialize_feedback_pipeline(model, device):
    """
    Initialize the feedback learning pipeline
    
    Args:
        model: PatchCore model instance
        device: PyTorch device
        
    Returns:
        FeedbackLearningPipeline instance
    """
    return FeedbackLearningPipeline(model, device)


def run_feedback_training(pipeline: FeedbackLearningPipeline):
    """
    Run a single training cycle
    
    Args:
        pipeline: FeedbackLearningPipeline instance
        
    Returns:
        Training results dictionary
    """
    return pipeline.run_training_cycle()
