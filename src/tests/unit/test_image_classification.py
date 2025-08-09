"""
Unit tests for Image Classification module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from PIL import Image

from components.ai_vision.image_classification import (
    ImageClassifier,
    ImageClassificationTask,
    ClassificationResult,
    TopKResult,
    BatchClassificationResult,
    LoggingHook,
    CheckpointHook,
    classify_image,
    classify_images_batch,
    create_classifier
)
from components.ai_vision.core import TaskStatus


@pytest.mark.unit
@pytest.mark.classification
class TestClassificationResult:
    """Test ClassificationResult dataclass"""
    
    def test_classification_result_creation(self):
        """Test creating a ClassificationResult"""
        result = ClassificationResult(
            class_id=1,
            class_name="cat",
            confidence=0.95
        )
        
        assert result.class_id == 1
        assert result.class_name == "cat"
        assert result.confidence == 0.95
    
    def test_classification_result_to_dict(self):
        """Test converting ClassificationResult to dictionary"""
        result = ClassificationResult(
            class_id=2,
            class_name="dog",
            confidence=0.87
        )
        
        result_dict = result.to_dict()
        expected = {
            'class_id': 2,
            'class_name': 'dog',
            'confidence': 0.87
        }
        
        assert result_dict == expected


@pytest.mark.unit
@pytest.mark.classification
class TestTopKResult:
    """Test TopKResult dataclass"""
    
    def test_top_k_result_creation(self):
        """Test creating a TopKResult"""
        predictions = [
            ClassificationResult(0, "cat", 0.95),
            ClassificationResult(1, "dog", 0.87),
            ClassificationResult(2, "bird", 0.65)
        ]
        
        result = TopKResult(
            predictions=predictions,
            processing_time=0.1,
            k=3
        )
        
        assert len(result.predictions) == 3
        assert result.processing_time == 0.1
        assert result.k == 3
        assert result.predictions[0].class_name == "cat"
    
    def test_top_k_result_to_dict(self):
        """Test converting TopKResult to dictionary"""
        predictions = [
            ClassificationResult(0, "cat", 0.95),
            ClassificationResult(1, "dog", 0.87)
        ]
        
        result = TopKResult(
            predictions=predictions,
            processing_time=0.15,
            k=2
        )
        
        result_dict = result.to_dict()
        
        assert "predictions" in result_dict
        assert "processing_time" in result_dict
        assert "k" in result_dict
        assert len(result_dict["predictions"]) == 2
        assert result_dict["predictions"][0]["class_name"] == "cat"


@pytest.mark.unit
@pytest.mark.classification
class TestBatchClassificationResult:
    """Test BatchClassificationResult dataclass"""
    
    def test_batch_result_creation(self):
        """Test creating a BatchClassificationResult"""
        predictions1 = [ClassificationResult(0, "cat", 0.95)]
        predictions2 = [ClassificationResult(1, "dog", 0.87)]
        
        results = [
            TopKResult(predictions1, 0.1, 1),
            TopKResult(predictions2, 0.12, 1)
        ]
        
        batch_result = BatchClassificationResult(
            results=results,
            processing_time=0.22,
            batch_size=2
        )
        
        assert len(batch_result.results) == 2
        assert batch_result.processing_time == 0.22
        assert batch_result.batch_size == 2


@pytest.mark.unit
@pytest.mark.classification
class TestImageClassificationTask:
    """Test ImageClassificationTask"""
    
    def test_task_initialization(self, mock_model_hub):
        """Test task initialization"""
        task = ImageClassificationTask("task_001", model_name="resnet50")
        
        assert task.task_id == "task_001"
        assert task.task_type == "image_classification"
        assert task.model_name == "resnet50"
        assert task.model is None
    
    def test_input_validation_success(self, mock_model_hub):
        """Test successful input validation"""
        task = ImageClassificationTask("task_001")
        
        # Test valid inputs
        assert task.validate_input({"image": "test_image"}) is True
        assert task.validate_input({"batch_images": ["img1", "img2"]}) is True
        assert task.validate_input("direct_image") is True
    
    def test_input_validation_failure(self, mock_model_hub):
        """Test input validation failure"""
        task = ImageClassificationTask("task_001")
        
        # Test invalid inputs
        assert task.validate_input(None) is False
        assert task.validate_input({}) is False  # Neither image nor batch_images
    
    @pytest.mark.asyncio
    async def test_execute_single_image(self, mock_model_hub, sample_image_pil):
        """Test executing classification on single image"""
        with patch('components.ai_vision.image_classification.classify_image') as mock_classify:
            # Setup mock
            mock_result = TopKResult(
                predictions=[ClassificationResult(0, "cat", 0.95)],
                processing_time=0.1,
                k=5
            )
            mock_classify.return_value = mock_result
            
            task = ImageClassificationTask("task_001")
            task.model = Mock()  # Mock loaded model
            
            input_data = {"image": sample_image_pil, "k": 5}
            result = await task.execute(input_data)
            
            assert result.task_id == "task_001"
            assert result.status == TaskStatus.COMPLETED
            assert result.confidence == 0.95
            assert "predictions" in result.data
    
    @pytest.mark.asyncio
    async def test_execute_batch_images(self, mock_model_hub, sample_image_pil):
        """Test executing classification on batch of images"""
        with patch('components.ai_vision.image_classification.classify_images_batch') as mock_batch:
            # Setup mock
            mock_result = BatchClassificationResult(
                results=[TopKResult([ClassificationResult(0, "cat", 0.95)], 0.1, 5)],
                processing_time=0.1,
                batch_size=1
            )
            mock_batch.return_value = mock_result
            
            task = ImageClassificationTask("task_001")
            task.model = Mock()  # Mock loaded model
            
            input_data = {"batch_images": [sample_image_pil], "k": 5}
            result = await task.execute(input_data)
            
            assert result.task_id == "task_001"
            assert result.status == TaskStatus.COMPLETED
            assert result.metadata["batch_size"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_model_loading_error(self, mock_model_hub):
        """Test execution when model loading fails"""
        mock_model_hub.ensure_model_ready.side_effect = Exception("Model load failed")
        
        task = ImageClassificationTask("task_001")
        task.model_hub = mock_model_hub
        
        input_data = {"image": "test_image"}
        result = await task.execute(input_data)
        
        assert result.status == TaskStatus.FAILED
        assert "Model load failed" in result.error_message


@pytest.mark.unit
@pytest.mark.classification
class TestImageClassifier:
    """Test ImageClassifier class"""
    
    def test_classifier_initialization(self, mock_model_hub):
        """Test classifier initialization"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            classifier = ImageClassifier(
                model_name="resnet50",
                device="cpu",
                num_classes=1000
            )
            
            assert classifier.model_name == "resnet50"
            assert classifier.num_classes == 1000
            assert classifier.model is None
    
    def test_classifier_class_names_loading(self, mock_model_hub):
        """Test class names loading"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            classifier = ImageClassifier(num_classes=5)
            
            assert len(classifier.class_names) == 5
            assert all(isinstance(name, str) for name in classifier.class_names)
    
    def test_classifier_custom_num_classes(self, mock_model_hub):
        """Test classifier with custom number of classes"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            classifier = ImageClassifier(num_classes=100)
            
            assert len(classifier.class_names) == 100
    
    @pytest.mark.asyncio
    async def test_ensure_model_ready_success(self, mock_model_hub, mock_torch):
        """Test successful model loading"""
        mock_model = Mock()
        mock_model_hub.ensure_model_ready.return_value = mock_model
        
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.image_classification._lazy_import_torch', return_value=mock_torch):
                with patch('components.ai_vision.image_classification._lazy_import_torchvision') as mock_tv:
                    mock_torchvision, mock_transforms = Mock(), Mock()
                    mock_tv.return_value = (mock_torchvision, mock_transforms)
                    mock_torchvision.models.resnet50.return_value = mock_model
                    
                    classifier = ImageClassifier(model_name="resnet50")
                    result = await classifier.ensure_model_ready()
                    
                    assert result is True
                    assert classifier.model is not None
    
    @pytest.mark.asyncio
    async def test_ensure_model_ready_failure(self, mock_model_hub):
        """Test model loading failure"""
        mock_model_hub.ensure_model_ready.side_effect = Exception("Load failed")
        
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            classifier = ImageClassifier()
            result = await classifier.ensure_model_ready()
            
            assert result is False
    
    def test_preprocess_image_pil(self, mock_model_hub, sample_image_pil):
        """Test image preprocessing with PIL image"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.image_classification._lazy_import_torchvision') as mock_tv:
                with patch('components.ai_vision.image_classification._lazy_import_pil') as mock_pil:
                    # Setup mocks
                    mock_torchvision, mock_transforms = Mock(), Mock()
                    mock_tv.return_value = (mock_torchvision, mock_transforms)
                    mock_pil.return_value = Image
                    
                    # Create mock transform
                    mock_transform = Mock()
                    mock_transform.return_value = Mock()  # Mock tensor
                    mock_transforms.Compose.return_value = mock_transform
                    
                    classifier = ImageClassifier()
                    classifier._init_transforms()
                    
                    result = classifier._preprocess_image(sample_image_pil)
                    mock_transform.assert_called_once_with(sample_image_pil)
    
    def test_preprocess_image_numpy(self, mock_model_hub, sample_image_array):
        """Test image preprocessing with numpy array"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.image_classification._lazy_import_torchvision') as mock_tv:
                with patch('components.ai_vision.image_classification._lazy_import_pil') as mock_pil:
                    # Setup mocks
                    mock_torchvision, mock_transforms = Mock(), Mock()
                    mock_tv.return_value = (mock_torchvision, mock_transforms)
                    mock_PIL = Mock()
                    mock_PIL.fromarray.return_value = Mock()
                    mock_pil.return_value = mock_PIL
                    
                    mock_transform = Mock()
                    mock_transform.return_value = Mock()
                    mock_transforms.Compose.return_value = mock_transform
                    
                    classifier = ImageClassifier()
                    classifier._init_transforms()
                    
                    result = classifier._preprocess_image(sample_image_array)
                    mock_PIL.fromarray.assert_called_once_with(sample_image_array)
    
    def test_classify_image_without_model(self, mock_model_hub):
        """Test classification without loaded model"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            classifier = ImageClassifier()
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                classifier.classify_image("test_image")
    
    def test_classify_images_batch_without_model(self, mock_model_hub):
        """Test batch classification without loaded model"""
        with patch('components.ai_vision.image_classification.get_model_hub', return_value=mock_model_hub):
            classifier = ImageClassifier()
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                classifier.classify_images_batch(["test_image"])


@pytest.mark.unit
@pytest.mark.classification
class TestFineTuningHooks:
    """Test fine-tuning hooks"""
    
    def test_logging_hook_initialization(self):
        """Test LoggingHook initialization"""
        hook = LoggingHook(log_every_n_batches=50)
        assert hook.log_every_n_batches == 50
    
    def test_logging_hook_training_start(self):
        """Test LoggingHook training start callback"""
        hook = LoggingHook()
        
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_train_loader = Mock()
        mock_train_loader.__len__ = Mock(return_value=100)
        
        # Should not raise exception
        hook.on_training_start(mock_model, mock_optimizer, mock_train_loader)
    
    def test_logging_hook_training_start_with_invalid_loader(self):
        """Test LoggingHook with invalid loader"""
        hook = LoggingHook()
        
        mock_model = Mock()
        mock_optimizer = Mock()
        invalid_loader = "not_a_loader"  # Invalid type
        
        # Should handle gracefully
        hook.on_training_start(mock_model, mock_optimizer, invalid_loader)
    
    def test_logging_hook_epoch_start(self):
        """Test LoggingHook epoch start callback"""
        hook = LoggingHook()
        
        mock_model = Mock()
        mock_optimizer = Mock()
        
        # Should not raise exception
        hook.on_epoch_start(1, mock_model, mock_optimizer)
    
    def test_logging_hook_batch_end(self):
        """Test LoggingHook batch end callback"""
        hook = LoggingHook(log_every_n_batches=2)
        
        mock_model = Mock()
        
        # Should log on batch 0
        hook.on_batch_end(0, Mock(), Mock(), 0.5, mock_model)
        
        # Should not log on batch 1
        hook.on_batch_end(1, Mock(), Mock(), 0.4, mock_model)
        
        # Should log on batch 2
        hook.on_batch_end(2, Mock(), Mock(), 0.3, mock_model)
    
    def test_logging_hook_epoch_end(self):
        """Test LoggingHook epoch end callback"""
        hook = LoggingHook()
        
        mock_model = Mock()
        
        # Test with validation loss
        hook.on_epoch_end(1, 0.5, 0.4, mock_model)
        
        # Test without validation loss
        hook.on_epoch_end(2, 0.3, None, mock_model)
    
    def test_checkpoint_hook_initialization(self, temp_test_dir):
        """Test CheckpointHook initialization"""
        hook = CheckpointHook(temp_test_dir, save_every_n_epochs=3)
        
        assert hook.checkpoint_dir == temp_test_dir
        assert hook.save_every_n_epochs == 3
        assert temp_test_dir.exists()
    
    def test_checkpoint_hook_epoch_end(self, temp_test_dir, mock_torch):
        """Test CheckpointHook epoch end callback"""
        with patch('components.ai_vision.image_classification._lazy_import_torch', return_value=mock_torch):
            hook = CheckpointHook(temp_test_dir, save_every_n_epochs=2)
            
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            # Should save on epoch 0
            hook.on_epoch_end(0, 0.5, 0.4, mock_model)
            mock_torch.save.assert_called()
            
            # Should not save on epoch 1
            mock_torch.save.reset_mock()
            hook.on_epoch_end(1, 0.4, 0.3, mock_model)
            mock_torch.save.assert_not_called()
            
            # Should save on epoch 2
            hook.on_epoch_end(2, 0.3, 0.2, mock_model)
            mock_torch.save.assert_called()
    
    def test_checkpoint_hook_training_end(self, temp_test_dir, mock_torch):
        """Test CheckpointHook training end callback"""
        with patch('components.ai_vision.image_classification._lazy_import_torch', return_value=mock_torch):
            hook = CheckpointHook(temp_test_dir)
            
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            final_metrics = {"final_train_loss": 0.1, "final_val_loss": 0.2}
            hook.on_training_end(mock_model, final_metrics)
            
            mock_torch.save.assert_called()
            # Check that final_model.pth was saved
            expected_path = temp_test_dir / "final_model.pth"
            mock_torch.save.assert_called_with({"param": "value"}, expected_path)


@pytest.mark.unit
@pytest.mark.classification
class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_classify_image_without_model(self):
        """Test classify_image function without model"""
        with pytest.raises(ValueError, match="Model must be provided"):
            classify_image("test_image", model=None)
    
    def test_classify_image_with_model(self, sample_image_pil):
        """Test classify_image function with model"""
        mock_model = Mock()
        
        with patch('components.ai_vision.image_classification.ImageClassifier') as mock_classifier_class:
            mock_classifier = Mock()
            mock_result = TopKResult([ClassificationResult(0, "cat", 0.95)], 0.1, 5)
            mock_classifier.classify_image.return_value = mock_result
            mock_classifier_class.return_value = mock_classifier
            
            result = classify_image(sample_image_pil, mock_model, k=5)
            
            assert result == mock_result
            mock_classifier.classify_image.assert_called_once_with(sample_image_pil, 5)
    
    def test_classify_images_batch_without_model(self):
        """Test classify_images_batch function without model"""
        with pytest.raises(ValueError, match="Model must be provided"):
            classify_images_batch(["test_image"], model=None)
    
    def test_classify_images_batch_with_model(self, sample_image_pil):
        """Test classify_images_batch function with model"""
        mock_model = Mock()
        images = [sample_image_pil]
        
        with patch('components.ai_vision.image_classification.ImageClassifier') as mock_classifier_class:
            mock_classifier = Mock()
            mock_result = BatchClassificationResult(
                [TopKResult([ClassificationResult(0, "cat", 0.95)], 0.1, 5)],
                0.1, 1
            )
            mock_classifier.classify_images_batch.return_value = mock_result
            mock_classifier_class.return_value = mock_classifier
            
            result = classify_images_batch(images, mock_model, k=5)
            
            assert result == mock_result
            mock_classifier.classify_images_batch.assert_called_once_with(images, 5)
    
    @pytest.mark.asyncio
    async def test_create_classifier_success(self, mock_model_hub):
        """Test create_classifier function success"""
        with patch('components.ai_vision.image_classification.ImageClassifier') as mock_classifier_class:
            mock_classifier = Mock()
            mock_classifier.ensure_model_ready = AsyncMock(return_value=True)
            mock_classifier_class.return_value = mock_classifier
            
            result = await create_classifier("resnet50", "cpu")
            
            assert result == mock_classifier
            mock_classifier.ensure_model_ready.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_classifier_failure(self, mock_model_hub):
        """Test create_classifier function failure"""
        with patch('components.ai_vision.image_classification.ImageClassifier') as mock_classifier_class:
            mock_classifier = Mock()
            mock_classifier.ensure_model_ready = AsyncMock(return_value=False)
            mock_classifier_class.return_value = mock_classifier
            
            with pytest.raises(RuntimeError, match="Failed to initialize classifier"):
                await create_classifier("resnet50", "cpu")
