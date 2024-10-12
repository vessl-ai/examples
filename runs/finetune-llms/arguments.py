from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    model_name_or_path: str

    # quantization config
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "Enables 8bit quantization"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Enables 4bit quantization"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    bnb_4bit_use_double_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables double quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8", metadata={"help": "Store dtype for 4bit base models"}
    )

    # use flash attention
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training"},
    )

    # use unsloth
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training"},
    )


@dataclass
class PeftArguments:
    peft_type: Optional[str] = field(default=None)
    task_type: Optional[str] = field(default="CAUSAL_LM")
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]
    )
    lora_dropout: Optional[float] = field(default=0.0)
    lora_bias: Optional[str] = field(default="none")


@dataclass
class DatasetArguments:
    dataset_name: str
    packing: Optional[bool] = field(default=False)
    dataset_text_field: Optional[str] = field(
        default=None, metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=512)


@dataclass
class VesslArguments:
    upload_model: Optional[bool] = field(default=False)
    repository_name: Optional[str] = field(default=None)
    save_merged : Optional[bool] = field(default=False)
