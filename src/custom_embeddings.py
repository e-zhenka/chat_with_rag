from typing import List
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import os


class ONNXEmbedder:
    """
    Класс, реализующий работу Chroma DB с моделью векторизации models/multilingual-e5-small
    """

    def __init__(self):
        self.model_dir = "models/multilingual-e5-small"

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

            # Скачиваем файлы модели
            self._download_model_files()

        # Инициализируем токенизатор и модель
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.session = ort.InferenceSession(f"{self.model_dir}/onnx/model.onnx")

    def _download_model_files(self):
        """Скачивает все необходимые файлы модели"""
        files_to_download = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "onnx/model.onnx"
        ]

        for file in files_to_download:
            local_filename = file.split('/')[-1]  # Извлекаем имя файла
            if not os.path.exists(f"{self.model_dir}/{local_filename}"):
                hf_hub_download(
                    repo_id="intfloat/multilingual-e5-small",
                    filename=file,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )

    # Сохраняем оригинальные методы без изменений
    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_token_type_ids=True
        )

        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        # Преобразуем тип данных в int64
        inputs["input_ids"] = inputs["input_ids"].astype(np.int64)
        inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs["token_type_ids"]
            }
        )
        return self._mean_pooling(outputs[0], inputs["attention_mask"]).tolist()

    def _mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return np.sum(model_output * input_mask_expanded, axis=1) / sum_mask
