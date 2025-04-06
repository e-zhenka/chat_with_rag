from typing import List
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

class ONNXEmbedder:
    def __init__(self, model_path: str = "models/multilingual-e5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.session = ort.InferenceSession(f"{model_path}/model.onnx")
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_token_type_ids=True
        )
        
        # Гарантируем наличие всех требуемых входов
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            
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
