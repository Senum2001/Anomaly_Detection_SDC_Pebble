"""
Model Versioning & Training History System
Tracks model parameters, thresholds, and training evolution in Supabase
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from supabase import create_client, Client
import uuid

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://xbcgrpqiibicestnhytt.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class ModelVersionTracker:
    """
    Tracks model versions, parameters, and training history
    """
    
    def __init__(self):
        """Initialize the model version tracker"""
        self.supabase = supabase
        
    def get_current_model_state(self) -> Dict[str, Any]:
        """
        Get current model parameters and thresholds
        
        Returns:
            Dictionary containing current model state
        """
        # Read current model adjustments if they exist
        adjustments = {}
        if os.path.exists("model_adjustments.json"):
            with open("model_adjustments.json", "r") as f:
                adjustments = json.load(f)
        
        # Read training state
        training_state = {}
        if os.path.exists("feedback_training_state.json"):
            with open("feedback_training_state.json", "r") as f:
                training_state = json.load(f)
        
        # Define current model parameters
        model_state = {
            "version_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            
            # Model Configuration
            "model_architecture": "PatchCore",
            "backbone": "Wide ResNet-50",
            "layers": ["layer2", "layer3"],
            "input_size": [256, 256],
            
            # Detection Thresholds
            "anomaly_threshold": 128,  # Binary mask threshold
            "confidence_range": [0.3, 0.99],
            "min_detection_size": 100,  # Minimum pixels for detection
            
            # Classification Thresholds
            "red_color_threshold": {
                "hue_range": [0, 10, 170, 180],
                "saturation_min": 100,
                "value_min": 100
            },
            "yellow_color_threshold": {
                "hue_range": [20, 30],
                "saturation_min": 100,
                "value_min": 100
            },
            "orange_color_threshold": {
                "hue_range": [10, 20],
                "saturation_min": 100,
                "value_min": 100
            },
            
            # Post-processing Parameters
            "merge_distance_threshold": 20,
            "iou_threshold": 0.4,
            "min_contour_area": 100,
            
            # Learned Adjustments (from feedback)
            "false_positive_rate": adjustments.get("fp_rate", 0.0),
            "false_negative_rate": adjustments.get("fn_rate", 0.0),
            "threshold_recommendation": adjustments.get("recommendation", "Not yet calculated"),
            
            # Training Metadata
            "total_feedback_processed": training_state.get("total_feedback_processed", 0),
            "last_training_time": training_state.get("last_training_time"),
            "training_runs_count": len(training_state.get("training_runs", []))
        }
        
        return model_state
    
    def log_model_version(self, model_state: Dict[str, Any]) -> Optional[str]:
        """
        Log current model version to Supabase
        
        Args:
            model_state: Dictionary containing model parameters
            
        Returns:
            Version ID if successful, None otherwise
        """
        try:
            # Prepare record for database
            record = {
                "version_id": model_state["version_id"],
                "timestamp": model_state["timestamp"],
                "model_architecture": model_state["model_architecture"],
                "backbone": model_state["backbone"],
                "parameters": {
                    "layers": model_state["layers"],
                    "input_size": model_state["input_size"],
                    "anomaly_threshold": model_state["anomaly_threshold"],
                    "confidence_range": model_state["confidence_range"],
                    "min_detection_size": model_state["min_detection_size"]
                },
                "thresholds": {
                    "red_color": model_state["red_color_threshold"],
                    "yellow_color": model_state["yellow_color_threshold"],
                    "orange_color": model_state["orange_color_threshold"],
                    "merge_distance": model_state["merge_distance_threshold"],
                    "iou": model_state["iou_threshold"],
                    "min_contour_area": model_state["min_contour_area"]
                },
                "learned_adjustments": {
                    "false_positive_rate": model_state["false_positive_rate"],
                    "false_negative_rate": model_state["false_negative_rate"],
                    "recommendation": model_state["threshold_recommendation"]
                },
                "training_metadata": {
                    "total_feedback_processed": model_state["total_feedback_processed"],
                    "last_training_time": model_state["last_training_time"],
                    "training_runs_count": model_state["training_runs_count"]
                },
                "is_active": True
            }
            
            # Insert into database
            response = self.supabase.table('model_versions').insert(record).execute()
            
            if response.data:
                print(f"[Model Versioning] Logged version {model_state['version_id']}")
                return model_state["version_id"]
            else:
                print("[Model Versioning] Failed to log version")
                return None
                
        except Exception as e:
            print(f"[Model Versioning] Error logging version: {e}")
            return None
    
    def log_training_cycle(self, 
                          before_state: Dict[str, Any],
                          after_state: Dict[str, Any],
                          feedback_count: int,
                          patterns: Dict[str, Any],
                          performance_metrics: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Log a training cycle with before/after comparison
        
        Args:
            before_state: Model state before training
            after_state: Model state after training
            feedback_count: Number of feedback samples processed
            patterns: Pattern analysis from feedback
            performance_metrics: Optional performance metrics
            
        Returns:
            Training cycle ID if successful
        """
        try:
            cycle_id = str(uuid.uuid4())
            
            # Calculate parameter changes
            parameter_changes = self._calculate_parameter_changes(before_state, after_state)
            
            record = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now().isoformat(),
                "before_version_id": before_state["version_id"],
                "after_version_id": after_state["version_id"],
                "feedback_samples_processed": feedback_count,
                
                # Pattern Analysis
                "feedback_patterns": {
                    "label_changes": patterns.get("label_changes", []),
                    "bbox_adjustments": patterns.get("bbox_adjustments", []),
                    "false_positives": patterns.get("false_positives", 0),
                    "false_negatives": patterns.get("false_negatives", 0)
                },
                
                # Parameter Changes
                "parameter_changes": parameter_changes,
                
                # Performance Metrics (if available)
                "performance_metrics": performance_metrics or {
                    "accuracy_improvement": "Not yet calculated",
                    "precision_improvement": "Not yet calculated",
                    "recall_improvement": "Not yet calculated"
                },
                
                # Recommendations
                "threshold_recommendation": after_state.get("threshold_recommendation", ""),
                
                # Status
                "status": "completed",
                "notes": f"Processed {feedback_count} feedback samples"
            }
            
            # Insert into database
            response = self.supabase.table('training_history').insert(record).execute()
            
            if response.data:
                print(f"[Training History] Logged cycle {cycle_id}")
                return cycle_id
            else:
                print("[Training History] Failed to log cycle")
                return None
                
        except Exception as e:
            print(f"[Training History] Error logging cycle: {e}")
            return None
    
    def _calculate_parameter_changes(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate what changed between before and after states"""
        changes = {}
        
        # Compare false positive/negative rates
        if before["false_positive_rate"] != after["false_positive_rate"]:
            changes["false_positive_rate"] = {
                "before": before["false_positive_rate"],
                "after": after["false_positive_rate"],
                "delta": after["false_positive_rate"] - before["false_positive_rate"]
            }
        
        if before["false_negative_rate"] != after["false_negative_rate"]:
            changes["false_negative_rate"] = {
                "before": before["false_negative_rate"],
                "after": after["false_negative_rate"],
                "delta": after["false_negative_rate"] - before["false_negative_rate"]
            }
        
        # Compare training metadata
        if before["total_feedback_processed"] != after["total_feedback_processed"]:
            changes["total_feedback_processed"] = {
                "before": before["total_feedback_processed"],
                "after": after["total_feedback_processed"],
                "delta": after["total_feedback_processed"] - before["total_feedback_processed"]
            }
        
        if before["threshold_recommendation"] != after["threshold_recommendation"]:
            changes["threshold_recommendation"] = {
                "before": before["threshold_recommendation"],
                "after": after["threshold_recommendation"]
            }
        
        return changes
    
    def get_version_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent model version history
        
        Args:
            limit: Maximum number of versions to retrieve
            
        Returns:
            List of model versions
        """
        try:
            response = self.supabase.table('model_versions')\
                .select('*')\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"[Model Versioning] Error fetching history: {e}")
            return []
    
    def get_training_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent training cycles
        
        Args:
            limit: Maximum number of cycles to retrieve
            
        Returns:
            List of training cycles
        """
        try:
            response = self.supabase.table('training_history')\
                .select('*')\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"[Training History] Error fetching history: {e}")
            return []
    
    def get_active_version(self) -> Optional[Dict[str, Any]]:
        """
        Get currently active model version
        
        Returns:
            Active model version or None
        """
        try:
            response = self.supabase.table('model_versions')\
                .select('*')\
                .eq('is_active', True)\
                .order('timestamp', desc=True)\
                .limit(1)\
                .execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"[Model Versioning] Error fetching active version: {e}")
            return None
    
    def generate_comparison_table(self, version_ids: List[str]) -> str:
        """
        Generate a comparison table between model versions
        
        Args:
            version_ids: List of version IDs to compare
            
        Returns:
            Formatted comparison table string
        """
        try:
            versions = []
            for vid in version_ids:
                response = self.supabase.table('model_versions')\
                    .select('*')\
                    .eq('version_id', vid)\
                    .execute()
                if response.data:
                    versions.append(response.data[0])
            
            if not versions:
                return "No versions found"
            
            # Generate comparison table
            table = "\n" + "=" * 100 + "\n"
            table += "MODEL VERSION COMPARISON\n"
            table += "=" * 100 + "\n\n"
            
            for i, v in enumerate(versions):
                table += f"Version {i+1}: {v['version_id'][:8]}...\n"
                table += f"Timestamp: {v['timestamp']}\n"
                table += f"Architecture: {v['model_architecture']} ({v['backbone']})\n"
                table += f"False Positive Rate: {v['learned_adjustments']['false_positive_rate']:.2%}\n"
                table += f"False Negative Rate: {v['learned_adjustments']['false_negative_rate']:.2%}\n"
                table += f"Feedback Processed: {v['training_metadata']['total_feedback_processed']}\n"
                table += f"Recommendation: {v['learned_adjustments']['recommendation']}\n"
                table += "-" * 100 + "\n"
            
            return table
            
        except Exception as e:
            print(f"[Model Versioning] Error generating comparison: {e}")
            return f"Error: {e}"


def initialize_model_tracker():
    """Initialize the model version tracker"""
    return ModelVersionTracker()


# SQL for creating the required tables (run in Supabase Dashboard)
CREATE_TABLES_SQL = """
-- Table: model_versions
-- Stores each model version with parameters and thresholds
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_architecture VARCHAR(100) NOT NULL,
    backbone VARCHAR(100),
    parameters JSONB,
    thresholds JSONB,
    learned_adjustments JSONB,
    training_metadata JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_versions_timestamp ON model_versions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active) WHERE is_active = TRUE;

-- Table: training_history
-- Stores training cycle information with before/after comparisons
CREATE TABLE IF NOT EXISTS training_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    before_version_id VARCHAR(255),
    after_version_id VARCHAR(255),
    feedback_samples_processed INTEGER,
    feedback_patterns JSONB,
    parameter_changes JSONB,
    performance_metrics JSONB,
    threshold_recommendation TEXT,
    status VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_history_timestamp ON training_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_training_history_status ON training_history(status);

-- Foreign key constraints
ALTER TABLE training_history 
    ADD CONSTRAINT fk_before_version 
    FOREIGN KEY (before_version_id) 
    REFERENCES model_versions(version_id);

ALTER TABLE training_history 
    ADD CONSTRAINT fk_after_version 
    FOREIGN KEY (after_version_id) 
    REFERENCES model_versions(version_id);
"""
