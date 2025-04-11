import whisper
import tempfile
import os
from typing import Optional
import torch

class AudioProcessor:
    """Класс для обработки аудио с помощью локальной модели Whisper."""
    
    def __init__(self, model_name: str = "tiny", device: str = "auto"):
        """
        Инициализация модели Whisper.
        
        :param model_name: Модель Whisper (tiny, base, small, medium, large).
        :param device: "cpu", "cuda" или "auto" для автоматического выбора.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None  # Модель загружается при первом использовании

    def _resolve_device(self, device: str) -> str:
        """Определяет доступное устройство (CPU/GPU)."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Ленивая загрузка модели Whisper."""
        if self.model is None:
            print(f"Загрузка модели Whisper ({self.model_name}) для {self.device}...")
            self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        """
        Преобразует аудио в текст.
        
        :param audio_bytes: Аудио в виде байтов.
        :return: Распознанный текст или None при ошибке.
        """
        try:
            self.load_model()  # Убедимся, что модель загружена
            
            # Создаем временный файл для аудио
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # Транскрибируем аудио
            result = self.model.transcribe(tmp_file_path, language="ru")
            
            # Удаляем временный файл
            os.unlink(tmp_file_path)
            
            return result["text"]
        except Exception as e:
            print(f"Ошибка при транскрибации аудио: {str(e)}")
            return None
