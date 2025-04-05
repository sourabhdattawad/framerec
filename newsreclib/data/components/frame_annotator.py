from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class XLMRobertFrameAnnotator(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer_max_len: int,
    ) -> None:
        super().__init__()

        file = "/mount/studenten-temp1/users/randomuser/newsreclib_new/newsreclib/local_models/mfc-xlm-roberta"

        self.tokenizer = AutoTokenizer.from_pretrained(
            file, model_max_length=tokenizer_max_len,  local_files_only=True
        )
        self.config = AutoConfig.from_pretrained(file, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(file, local_files_only=True)

    def forward(self, text: str) -> Tuple[str, float]:
        """Computes the sentiment orientation of a text.

        Args:
            text:
                A piece of text.

        Returns:
            A tuple containing the frame of the text.
        """
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        logits = self.model(**encoded_input).logits
        scores = F.softmax(logits[0], dim=0).detach().numpy()

        frame_class = self.config.id2label[scores.argmax()]
    
        return frame_class