# src/components/ai_vision/face_recognition.py

import sqlite3
import os
import cv2
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import asyncio
from pathlib import Path
import hashlib
from scipy.spatial.distance import cosine

from .core import VisionTask, VisionResult, TaskStatus, AIVisionAgent


@dataclass 
class BoundingBox:
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: BoundingBox
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None


@dataclass
class FaceRecognitionResult:
    """Face recognition result"""
    face_id: Optional[str]
    name: Optional[str]
    confidence: float
    embedding: np.ndarray
    detection: FaceDetection
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Identity:
    """Identity storage structure"""
    id: str
    name: str
    embeddings: List[np.ndarray]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class FaceDatabase:
    """SQLite database for face identity persistence"""
    
    def __init__(self, db_path: str = "faces.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.FaceDatabase")
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embeddings BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id TEXT,
                    confidence REAL,
                    detection_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (identity_id) REFERENCES identities (id)
                )
            """)
            
            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")
    
    def add_identity(self, identity: Identity) -> bool:
        """Add a new identity to the database"""
        try:
            embeddings_blob = pickle.dumps(identity.embeddings)
            metadata_str = str(identity.metadata) if identity.metadata else "{}"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO identities 
                    (id, name, embeddings, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    identity.id,
                    identity.name, 
                    embeddings_blob,
                    metadata_str,
                    identity.created_at.isoformat(),
                    identity.updated_at.isoformat()
                ))
                conn.commit()
            
            self.logger.info(f"Added identity: {identity.name} ({identity.id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add identity {identity.name}: {e}")
            return False
    
    def get_identity(self, identity_id: str) -> Optional[Identity]:
        """Get identity by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, embeddings, metadata, created_at, updated_at
                    FROM identities WHERE id = ?
                """, (identity_id,))
                
                row = cursor.fetchone()
                if row:
                    embeddings = pickle.loads(row[2])
                    metadata = eval(row[3]) if row[3] else {}
                    return Identity(
                        id=row[0],
                        name=row[1],
                        embeddings=embeddings,
                        metadata=metadata,
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5])
                    )
        except Exception as e:
            self.logger.error(f"Failed to get identity {identity_id}: {e}")
        
        return None
    
    def get_all_identities(self) -> List[Identity]:
        """Get all identities from database"""
        identities = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, embeddings, metadata, created_at, updated_at
                    FROM identities
                """)
                
                for row in cursor.fetchall():
                    embeddings = pickle.loads(row[2])
                    metadata = eval(row[3]) if row[3] else {}
                    identities.append(Identity(
                        id=row[0],
                        name=row[1], 
                        embeddings=embeddings,
                        metadata=metadata,
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5])
                    ))
                    
        except Exception as e:
            self.logger.error(f"Failed to get all identities: {e}")
        
        return identities
    
    def delete_identity(self, identity_id: str) -> bool:
        """Delete identity from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
                cursor.execute("DELETE FROM face_logs WHERE identity_id = ?", (identity_id,))
                conn.commit()
                
                self.logger.info(f"Deleted identity: {identity_id}")
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Failed to delete identity {identity_id}: {e}")
            return False
    
    def log_detection(self, identity_id: Optional[str], confidence: float, detection_data: Dict[str, Any]) -> bool:
        """Log a face detection event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO face_logs (identity_id, confidence, detection_data)
                    VALUES (?, ?, ?)
                """, (identity_id, confidence, str(detection_data)))
                conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to log detection: {e}")
            return False


class FaceDetector:
    """OpenCV DNN-based face detector"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize face detector
        
        Args:
            model_path: Path to DNN model file (.pb or .caffemodel)
            config_path: Path to model config file (.pbtxt or .prototxt)
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(f"{__name__}.FaceDetector")
        
        # Use default OpenCV DNN face detection model
        if model_path is None or config_path is None:
            self._use_default_model()
        else:
            self._load_custom_model(model_path, config_path)
    
    def _use_default_model(self):
        """Use OpenCV's built-in cascade classifier as fallback"""
        self.logger.info("Using Haar cascade face detector as default")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.use_dnn = False
    
    def _load_custom_model(self, model_path: str, config_path: str):
        """Load custom DNN model"""
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            self.use_dnn = True
            self.logger.info(f"Loaded DNN model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load DNN model: {e}")
            self._use_default_model()
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of FaceDetection objects
        """
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_cascade(image)
    
    def _detect_faces_dnn(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using DNN"""
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                bbox = BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1)
                faces.append(FaceDetection(bbox=bbox, confidence=confidence))
        
        return faces
    
    def _detect_faces_cascade(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        faces = []
        for (x, y, w, h) in faces_rect:
            bbox = BoundingBox(x=x, y=y, width=w, height=h)
            faces.append(FaceDetection(bbox=bbox, confidence=0.8))  # Cascade doesn't provide confidence
        
        return faces


class FaceEmbedder:
    """Face embedding extractor using FaceNet-like model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face embedder
        
        Args:
            model_path: Path to face embedding model
        """
        self.logger = logging.getLogger(f"{__name__}.FaceEmbedder")
        self.embedding_size = 128  # Standard FaceNet embedding size
        
        # For demo purposes, we'll use a simple feature extractor
        # In production, load a real FaceNet or InsightFace model
        self._initialize_embedder(model_path)
    
    def _initialize_embedder(self, model_path: Optional[str]):
        """Initialize embedding model"""
        if model_path and os.path.exists(model_path):
            try:
                # Load custom model (TensorFlow, ONNX, etc.)
                self.logger.info(f"Loading embedding model from {model_path}")
                # self.model = cv2.dnn.readNet(model_path)
                self.use_simple_features = False
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.use_simple_features = True
        else:
            self.logger.info("Using simple feature extractor (for demo)")
            self.use_simple_features = True
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from face image
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Face embedding as numpy array
        """
        if self.use_simple_features:
            return self._extract_simple_features(face_image)
        else:
            return self._extract_deep_features(face_image)
    
    def _extract_simple_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract simple features (demo implementation)"""
        # Resize to standard size
        face_resized = cv2.resize(face_image, (112, 112))
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract histogram and texture features
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        
        # LBP-like texture features
        lbp_features = self._compute_lbp_features(gray)
        
        # Combine features
        features = np.concatenate([hist, lbp_features])
        
        # Pad or truncate to embedding_size
        if len(features) > self.embedding_size:
            features = features[:self.embedding_size]
        elif len(features) < self.embedding_size:
            features = np.pad(features, (0, self.embedding_size - len(features)), 'constant')
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features.astype(np.float32)
    
    def _compute_lbp_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern features"""
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                code = 0
                
                # 8 neighbors
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # Compute histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=32, range=[0, 256])
        return hist.astype(np.float32)
    
    def _extract_deep_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract features using deep learning model (placeholder)"""
        # This would use a real FaceNet/InsightFace model
        # For now, fall back to simple features
        return self._extract_simple_features(face_image)


class FaceRecognizer:
    """Face recognition system combining detection and embedding"""
    
    def __init__(self, 
                 db_path: str = "faces.db",
                 detection_threshold: float = 0.5,
                 recognition_threshold: float = 0.7):
        """
        Initialize face recognizer
        
        Args:
            db_path: Path to faces database
            detection_threshold: Minimum confidence for face detection
            recognition_threshold: Minimum similarity for face recognition
        """
        self.recognition_threshold = recognition_threshold
        self.logger = logging.getLogger(f"{__name__}.FaceRecognizer")
        
        # Initialize components
        self.detector = FaceDetector(confidence_threshold=detection_threshold)
        self.embedder = FaceEmbedder()
        self.database = FaceDatabase(db_path)
        
        # Cache for known identities
        self._identity_cache = {}
        self._load_identities_cache()
    
    def _load_identities_cache(self):
        """Load known identities into memory cache"""
        identities = self.database.get_all_identities()
        self._identity_cache = {}
        
        for identity in identities:
            # Average embeddings for each identity
            if identity.embeddings:
                avg_embedding = np.mean(identity.embeddings, axis=0)
                self._identity_cache[identity.id] = {
                    'name': identity.name,
                    'embedding': avg_embedding,
                    'metadata': identity.metadata
                }
        
        self.logger.info(f"Loaded {len(self._identity_cache)} identities into cache")
    
    def enroll_identity(self, name: str, face_images: List[np.ndarray], 
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Enroll a new identity with multiple face images
        
        Args:
            name: Name of the person
            face_images: List of face images for the person
            metadata: Optional metadata dictionary
            
        Returns:
            Identity ID if successful, None otherwise
        """
        try:
            # Generate unique ID
            identity_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Extract embeddings from all face images
            embeddings = []
            for face_image in face_images:
                # Detect faces in image
                faces = self.detector.detect_faces(face_image)
                
                if not faces:
                    self.logger.warning(f"No face detected in enrollment image for {name}")
                    continue
                
                # Use the largest face
                largest_face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
                
                # Crop face region
                face_crop = self._crop_face(face_image, largest_face.bbox)
                
                # Extract embedding
                embedding = self.embedder.extract_embedding(face_crop)
                embeddings.append(embedding)
            
            if not embeddings:
                self.logger.error(f"No valid face embeddings extracted for {name}")
                return None
            
            # Create identity object
            identity = Identity(
                id=identity_id,
                name=name,
                embeddings=embeddings,
                metadata=metadata or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Save to database
            if self.database.add_identity(identity):
                # Update cache
                avg_embedding = np.mean(embeddings, axis=0)
                self._identity_cache[identity_id] = {
                    'name': name,
                    'embedding': avg_embedding,
                    'metadata': metadata or {}
                }
                
                self.logger.info(f"Successfully enrolled {name} with ID: {identity_id}")
                return identity_id
            
        except Exception as e:
            self.logger.error(f"Failed to enroll {name}: {e}")
        
        return None
    
    def recognize_faces(self, image: np.ndarray) -> List[FaceRecognitionResult]:
        """
        Recognize faces in image
        
        Args:
            image: Input image
            
        Returns:
            List of recognition results
        """
        results = []
        
        try:
            # Detect faces
            faces = self.detector.detect_faces(image)
            
            for face in faces:
                # Crop face region
                face_crop = self._crop_face(image, face.bbox)
                
                # Extract embedding
                embedding = self.embedder.extract_embedding(face_crop)
                
                # Find best match
                best_match_id, best_match_name, confidence = self._find_best_match(embedding)
                
                # Create result
                result = FaceRecognitionResult(
                    face_id=best_match_id,
                    name=best_match_name,
                    confidence=confidence,
                    embedding=embedding,
                    detection=face
                )
                
                results.append(result)
                
                # Log detection
                self.database.log_detection(
                    identity_id=best_match_id,
                    confidence=confidence,
                    detection_data={
                        'bbox': [face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height],
                        'detection_confidence': face.confidence
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
        
        return results
    
    def _crop_face(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Crop face region from image"""
        return image[bbox.y:bbox.y2, bbox.x:bbox.x2]
    
    def _find_best_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Find best matching identity for query embedding
        
        Args:
            query_embedding: Query face embedding
            
        Returns:
            Tuple of (identity_id, name, confidence)
        """
        if not self._identity_cache:
            return None, None, 0.0
        
        best_similarity = 0.0
        best_id = None
        best_name = None
        
        for identity_id, identity_info in self._identity_cache.items():
            # Compute cosine similarity
            similarity = 1 - cosine(query_embedding, identity_info['embedding'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = identity_id
                best_name = identity_info['name']
        
        # Check if similarity meets threshold
        if best_similarity >= self.recognition_threshold:
            return best_id, best_name, best_similarity
        else:
            return None, "Unknown", best_similarity
    
    def delete_identity(self, identity_id: str) -> bool:
        """Delete identity from database and cache"""
        success = self.database.delete_identity(identity_id)
        if success and identity_id in self._identity_cache:
            del self._identity_cache[identity_id]
        return success
    
    def get_identity_info(self, identity_id: str) -> Optional[Dict[str, Any]]:
        """Get identity information"""
        identity = self.database.get_identity(identity_id)
        if identity:
            return {
                'id': identity.id,
                'name': identity.name,
                'num_embeddings': len(identity.embeddings),
                'metadata': identity.metadata,
                'created_at': identity.created_at.isoformat(),
                'updated_at': identity.updated_at.isoformat()
            }
        return None
    
    def list_identities(self) -> List[Dict[str, Any]]:
        """List all enrolled identities"""
        identities = self.database.get_all_identities()
        return [
            {
                'id': identity.id,
                'name': identity.name,
                'num_embeddings': len(identity.embeddings),
                'created_at': identity.created_at.isoformat(),
                'updated_at': identity.updated_at.isoformat()
            }
            for identity in identities
        ]


class FaceRecognitionTask(VisionTask):
    """Face recognition task implementation"""
    
    def __init__(self, task_id: str, recognizer: FaceRecognizer, **kwargs):
        super().__init__(task_id, "face_recognition", **kwargs)
        self.recognizer = recognizer
    
    async def execute(self, input_data: Any) -> VisionResult:
        """Execute face recognition task"""
        try:
            if isinstance(input_data, str):
                # Load image from path
                image = cv2.imread(input_data)
            elif isinstance(input_data, np.ndarray):
                image = input_data
            else:
                raise ValueError("Input must be image path (str) or image array (np.ndarray)")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Perform face recognition
            results = self.recognizer.recognize_faces(image)
            
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_results.append({
                    'face_id': result.face_id,
                    'name': result.name,
                    'confidence': float(result.confidence),
                    'bbox': {
                        'x': result.detection.bbox.x,
                        'y': result.detection.bbox.y,
                        'width': result.detection.bbox.width,
                        'height': result.detection.bbox.height
                    },
                    'detection_confidence': float(result.detection.confidence)
                })
            
            return VisionResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=serializable_results,
                confidence=max([r['confidence'] for r in serializable_results], default=0.0),
                metadata={'num_faces': len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"Face recognition task failed: {e}")
            return VisionResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                data=None,
                confidence=0.0,
                error_message=str(e)
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if isinstance(input_data, str):
            return os.path.exists(input_data)
        elif isinstance(input_data, np.ndarray):
            return len(input_data.shape) == 3 and input_data.shape[2] == 3
        return False


# Convenience functions
def create_face_recognizer(db_path: str = "faces.db", 
                          detection_threshold: float = 0.5,
                          recognition_threshold: float = 0.7) -> FaceRecognizer:
    """Create a face recognizer instance"""
    return FaceRecognizer(
        db_path=db_path,
        detection_threshold=detection_threshold, 
        recognition_threshold=recognition_threshold
    )


def recognize_faces(image: Union[str, np.ndarray], 
                   recognizer: Optional[FaceRecognizer] = None) -> List[FaceRecognitionResult]:
    """
    Recognize faces in image
    
    Args:
        image: Image path or image array
        recognizer: Face recognizer instance (creates default if None)
        
    Returns:
        List of recognition results
    """
    if recognizer is None:
        recognizer = create_face_recognizer()
    
    if isinstance(image, str):
        image = cv2.imread(image)
    
    return recognizer.recognize_faces(image)


def enroll_person(name: str, 
                 face_images: List[Union[str, np.ndarray]],
                 recognizer: Optional[FaceRecognizer] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Enroll a person with multiple face images
    
    Args:
        name: Person's name
        face_images: List of face images (paths or arrays)
        recognizer: Face recognizer instance (creates default if None)
        metadata: Optional metadata
        
    Returns:
        Identity ID if successful, None otherwise
    """
    if recognizer is None:
        recognizer = create_face_recognizer()
    
    # Convert image paths to arrays
    image_arrays = []
    for img in face_images:
        if isinstance(img, str):
            img_array = cv2.imread(img)
            if img_array is not None:
                image_arrays.append(img_array)
        elif isinstance(img, np.ndarray):
            image_arrays.append(img)
    
    return recognizer.enroll_identity(name, image_arrays, metadata)
