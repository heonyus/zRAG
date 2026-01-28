"""
Fixed Reader - Stage 2 Answer 생성을 위한 완전 고정 모델

Track A의 공정 비교를 위해:
- 동일 모델 (Qwen3-8B, 4bit)
- 동일 프롬프트 템플릿
- 동일 decoding (greedy, deterministic)
- NO LoRA, NO fine-tuning

모든 방식 (zRAG, BM25, Dense)이 동일한 Reader로 Answer를 생성하여
Stage 1 (Evidence) 품질만 비교할 수 있도록 함.
"""

import torch
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ReaderConfig:
    """Fixed Reader 설정"""
    model_name: str = "Qwen/Qwen3-8B"
    quantization: str = "4bit"
    max_new_tokens: int = 64
    do_sample: bool = False  # greedy (deterministic)
    temperature: float = 1.0
    top_p: float = 1.0
    device: str = "cuda"

    # 고정 프롬프트 템플릿
    prompt_template: str = """Answer the question based on the given evidence.

Evidence: {evidence}

Question: {query}
Answer:"""


class FixedReader:
    """
    Stage 2 Answer 생성을 위한 완전 고정 Reader

    공정 비교 보장:
    - NO LoRA, NO fine-tuning
    - Greedy decoding (deterministic)
    - 동일 프롬프트 템플릿
    """

    def __init__(
        self,
        config: Optional[ReaderConfig] = None,
        model_name: str = None,
        quantization: str = None,
        device: str = None,
    ):
        """
        Args:
            config: ReaderConfig 객체
            model_name, quantization, device: config 없이 개별 지정 가능
        """
        if config is None:
            config = ReaderConfig()
        if model_name:
            config.model_name = model_name
        if quantization:
            config.quantization = quantization
        if device:
            config.device = device

        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing FixedReader: {config.model_name}")
        logger.info(f"  Quantization: {config.quantization}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Decoding: greedy (do_sample=False)")

        # Quantization config
        quant_config = None
        if config.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model (완전 frozen - NO LoRA)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("FixedReader initialized (all parameters frozen)")

    def build_prompt(self, query: str, evidence: str) -> str:
        """
        고정 프롬프트 템플릿으로 입력 구성

        Args:
            query: 질문
            evidence: Stage 1에서 생성된 Evidence 텍스트

        Returns:
            프롬프트 문자열
        """
        return self.config.prompt_template.format(
            query=query,
            evidence=evidence,
        )

    @torch.no_grad()
    def generate_answer(
        self,
        query: str,
        evidence: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        단일 QA에 대해 Answer 생성

        Args:
            query: 질문
            evidence: Evidence 텍스트
            max_new_tokens: 최대 생성 토큰 수 (기본값: config에서 지정)

        Returns:
            생성된 Answer 문자열
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        prompt = self.build_prompt(query, evidence)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 새로 생성된 토큰만 추출
        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return answer

    @torch.no_grad()
    def generate_answers(
        self,
        qa_pairs: List[Dict],
        evidences: List[str],
        query_key: str = "question",
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        여러 QA에 대해 Answer 생성

        Args:
            qa_pairs: [{question, answer, ...}, ...] 리스트
            evidences: 각 QA에 대응하는 Evidence 텍스트 리스트
            query_key: 질문이 저장된 키 이름
            max_new_tokens: 최대 생성 토큰 수
            show_progress: tqdm 진행률 표시

        Returns:
            생성된 Answer 문자열 리스트
        """
        assert len(qa_pairs) == len(evidences), \
            f"qa_pairs({len(qa_pairs)}) and evidences({len(evidences)}) must have same length"

        answers = []
        iterator = zip(qa_pairs, evidences)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Generating answers")

        for qa, evidence in iterator:
            query = qa[query_key]
            answer = self.generate_answer(query, evidence, max_new_tokens)
            answers.append(answer)

        return answers

    def verify_determinism(self, query: str, evidence: str, n_runs: int = 3) -> bool:
        """
        동일 입력에 대해 동일 출력이 나오는지 검증 (deterministic 확인)

        Args:
            query: 테스트 질문
            evidence: 테스트 Evidence
            n_runs: 반복 횟수

        Returns:
            True if deterministic, False otherwise
        """
        answers = []
        for _ in range(n_runs):
            answer = self.generate_answer(query, evidence)
            answers.append(answer)

        is_deterministic = len(set(answers)) == 1

        if is_deterministic:
            logger.info(f"Determinism verified: all {n_runs} runs produced identical output")
        else:
            logger.warning(f"NOT deterministic! Got {len(set(answers))} different outputs:")
            for i, ans in enumerate(answers):
                logger.warning(f"  Run {i+1}: {ans[:50]}...")

        return is_deterministic

    def get_info(self) -> Dict:
        """Reader 설정 정보 반환"""
        return {
            "model_name": self.config.model_name,
            "quantization": self.config.quantization,
            "device": self.device,
            "do_sample": self.config.do_sample,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "prompt_template": self.config.prompt_template,
            "lora": "None (fully frozen)",
        }


def create_reader_from_config(config_dict: dict) -> FixedReader:
    """
    Config dictionary에서 FixedReader 생성

    Args:
        config_dict: reader 설정 dict (from YAML)

    Returns:
        FixedReader 인스턴스
    """
    reader_config = ReaderConfig(
        model_name=config_dict.get("model", "Qwen/Qwen3-8B"),
        quantization=config_dict.get("quantization", "4bit"),
        max_new_tokens=config_dict.get("decoding", {}).get("max_new_tokens", 64),
        do_sample=config_dict.get("decoding", {}).get("do_sample", False),
        temperature=config_dict.get("decoding", {}).get("temperature", 1.0),
        top_p=config_dict.get("decoding", {}).get("top_p", 1.0),
    )

    # 커스텀 프롬프트 템플릿
    if "prompt_template" in config_dict:
        reader_config.prompt_template = config_dict["prompt_template"]

    return FixedReader(config=reader_config)


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    print("Creating FixedReader...")
    reader = FixedReader()

    print("\nReader info:")
    for k, v in reader.get_info().items():
        print(f"  {k}: {v}")

    # Determinism 테스트
    print("\nTesting determinism...")
    test_query = "What is the capital of France?"
    test_evidence = "France is a country in Western Europe. Its capital city is Paris, which is also the largest city in the country."

    is_det = reader.verify_determinism(test_query, test_evidence)
    print(f"Deterministic: {is_det}")

    # 단일 생성 테스트
    print("\nGenerating answer...")
    answer = reader.generate_answer(test_query, test_evidence)
    print(f"Query: {test_query}")
    print(f"Evidence: {test_evidence[:100]}...")
    print(f"Answer: {answer}")
