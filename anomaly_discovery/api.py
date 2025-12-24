#!/usr/bin/env python3
"""
Discovery API - FastAPI Router
Endpoints for anomaly detection and correlation discovery.
"""

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

discovery_router = APIRouter(prefix="/api/discovery", tags=["Anomaly Discovery"])

# Global engine instance (lazy loaded)
_engine = None


def get_engine():
    """Get or create discovery engine instance."""
    global _engine
    if _engine is None:
        from .discovery_engine import DiscoveryEngine
        _engine = DiscoveryEngine()
    return _engine


def get_db():
    return psycopg2.connect(DATABASE_URL)


class TrainRequest(BaseModel):
    days_of_data: int = 30
    min_samples: int = 1000


class DetectRequest(BaseModel):
    hours_back: float = 1.0
    persist: bool = True


class AnalyzeRequest(BaseModel):
    days_back: int = 7
    min_correlation: float = 0.3


# =============================================================================
# TRAINING
# =============================================================================
@discovery_router.post("/train")
async def train_models(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train anomaly detection models on historical data.
    This is a long-running operation - returns immediately and trains in background.
    """
    def train_task():
        engine = get_engine()
        result = engine.train(
            days_of_data=request.days_of_data,
            min_samples=request.min_samples
        )
        print(f"[Discovery API] Training complete: {result}")
    
    background_tasks.add_task(train_task)
    
    return {
        "status": "training_started",
        "message": f"Training on {request.days_of_data} days of data",
        "training_id": f"TRAIN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    }


@discovery_router.get("/status")
async def get_training_status():
    """Get current training/model status."""
    engine = get_engine()
    
    # Try to load models
    model_path = engine.model_dir
    model_exists = os.path.exists(os.path.join(model_path, 'ensemble_metadata.json'))
    
    return {
        "is_trained": engine.is_trained or model_exists,
        "model_path": model_path,
        "model_exists_on_disk": model_exists
    }


# =============================================================================
# ANOMALY DETECTION
# =============================================================================
@discovery_router.post("/detect")
async def detect_anomalies(request: DetectRequest):
    """
    Run anomaly detection on recent data.
    """
    engine = get_engine()
    
    # Load models if not trained
    if not engine.is_trained:
        try:
            engine.ensemble.load(engine.model_dir)
            engine.is_trained = True
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Models not trained. Call /train first. Error: {e}"
            )
    
    result = engine.detect_anomalies(
        hours_back=request.hours_back,
        persist=request.persist
    )
    
    return result


@discovery_router.get("/anomalies")
async def get_anomalies(
    machine_id: str = Query(None),
    severity: str = Query(None),
    hours_back: float = Query(24),
    reviewed: bool = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Get detected anomalies from database."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT * FROM anomaly_detections
            WHERE created_at > NOW() - INTERVAL '%s hours'
        """
        params = [hours_back]
        
        if machine_id:
            query += " AND machine_id = %s"
            params.append(machine_id)
        
        if severity:
            query += " AND severity = %s"
            params.append(severity)
        
        if reviewed is not None:
            query += " AND reviewed = %s"
            params.append(reviewed)
        
        query += " ORDER BY ensemble_score DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        anomalies = cursor.fetchall()
        
        # Convert timestamps
        for a in anomalies:
            for field in ['timestamp', 'created_at', 'reviewed_at']:
                if a.get(field):
                    a[field] = a[field].isoformat()
        
        cursor.close()
        conn.close()
        
        return {
            "count": len(anomalies),
            "data": anomalies
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/anomalies/{detection_id}/review")
async def review_anomaly(
    detection_id: str,
    is_true_positive: bool,
    reviewed_by: str = "System",
    notes: str = None
):
    """Mark an anomaly as reviewed and provide feedback."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            UPDATE anomaly_detections
            SET reviewed = TRUE,
                reviewed_by = %s,
                reviewed_at = NOW(),
                is_true_positive = %s,
                notes = %s
            WHERE detection_id = %s
            RETURNING *
        """, (reviewed_by, is_true_positive, notes, detection_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
@discovery_router.post("/correlations/analyze")
async def analyze_correlations(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Run correlation analysis on historical data.
    """
    def analyze_task():
        engine = get_engine()
        result = engine.analyze_correlations(days_back=request.days_back)
        print(f"[Discovery API] Correlation analysis complete: {result.get('correlation_count', 0)} found")
    
    background_tasks.add_task(analyze_task)
    
    return {
        "status": "analysis_started",
        "message": f"Analyzing {request.days_back} days of data"
    }


@discovery_router.get("/correlations")
async def get_correlations(
    source_machine: str = Query(None),
    target_machine: str = Query(None),
    min_strength: str = Query(None),  # 'weak', 'moderate', 'strong', 'very_strong'
    granger_causal: bool = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Get discovered correlations from database."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM correlation_discoveries WHERE 1=1"
        params = []
        
        if source_machine:
            query += " AND source_machine_id = %s"
            params.append(source_machine)
        
        if target_machine:
            query += " AND target_machine_id = %s"
            params.append(target_machine)
        
        if min_strength:
            strength_order = {'weak': 0, 'moderate': 1, 'strong': 2, 'very_strong': 3}
            min_val = strength_order.get(min_strength, 0)
            if min_val > 0:
                query += " AND strength IN %s"
                params.append(tuple([k for k, v in strength_order.items() if v >= min_val]))
        
        if granger_causal is not None:
            query += " AND granger_causal = %s"
            params.append(granger_causal)
        
        query += " ORDER BY ABS(correlation_coefficient) DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        correlations = cursor.fetchall()
        
        # Convert timestamps
        for c in correlations:
            for field in ['created_at', 'updated_at', 'analysis_start', 'analysis_end']:
                if c.get(field):
                    c[field] = c[field].isoformat()
        
        cursor.close()
        conn.close()
        
        return {
            "count": len(correlations),
            "data": correlations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INSIGHTS
# =============================================================================
@discovery_router.post("/insights/generate")
async def generate_insights():
    """Generate insights from recent discoveries."""
    engine = get_engine()
    return engine.generate_insight()


@discovery_router.get("/insights")
async def get_insights(
    insight_type: str = Query(None),
    priority: str = Query(None),
    acknowledged: bool = Query(None),
    limit: int = Query(20, ge=1, le=100)
):
    """Get generated insights."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM discovery_insights WHERE 1=1"
        params = []
        
        if insight_type:
            query += " AND insight_type = %s"
            params.append(insight_type)
        
        if priority:
            query += " AND priority = %s"
            params.append(priority)
        
        if acknowledged is not None:
            query += " AND acknowledged = %s"
            params.append(acknowledged)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        insights = cursor.fetchall()
        
        for i in insights:
            if i.get('created_at'):
                i['created_at'] = i['created_at'].isoformat()
        
        cursor.close()
        conn.close()
        
        return {
            "count": len(insights),
            "data": insights
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/insights/{insight_id}/acknowledge")
async def acknowledge_insight(
    insight_id: str,
    acknowledged_by: str = "System",
    action_taken: str = None
):
    """Acknowledge an insight."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            UPDATE discovery_insights
            SET acknowledged = TRUE,
                acknowledged_by = %s,
                acknowledged_at = NOW(),
                action_taken = %s
            WHERE insight_id = %s
            RETURNING *
        """, (acknowledged_by, action_taken, insight_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ['discovery_router']
