"""
No Retrieval Baseline
- LLM만으로 답변 (parametric knowledge only)
- Retrieval의 필요성 증명 (ablation)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from typing import List, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import compute_em, compute_f1, aggregate_metrics

logger = logging.getLogger(__name__)


class NoRetrievalBaseline:
    """No Retrieval: LLM direct answering without any context"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        quantization: str = "4bit",
        device: str = "cuda",
    ):
        self.device = device

        # Load model
        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

    def build_prompt(self, question: str) -> str:
        """No-context prompt"""
        return (
            f"Answer the following question concisely.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    @torch.no_grad()
    def generate(self, question: str, max_new_tokens: int = 64) -> str:
        """단일 질문에 대한 답변 생성"""
        prompt = self.build_prompt(question)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

    @torch.no_grad()
    def evaluate(
        self,
        qa_pairs: list,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 64,
    ) -> dict:
        """전체 QA pairs 평가"""
        if max_samples:
            qa_pairs = qa_pairs[:max_samples]

        all_metrics = []

        for item in tqdm(qa_pairs, desc="No Retrieval Baseline"):
            question = item["question"]
            answer = item["answer"]

            prediction = self.generate(question, max_new_tokens)

            em = compute_em(prediction, answer)
            f1 = compute_f1(prediction, answer)

            all_metrics.append({"em": em, "f1": f1})

        result = aggregate_metrics(all_metrics)
        result["num_samples"] = len(all_metrics)
        result["method"] = "no_retrieval"

        logger.info(f"No Retrieval: EM={result['em']:.4f}, F1={result['f1']:.4f}")
        return result
