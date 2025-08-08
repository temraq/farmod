from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

def load_model():
    # 4-битная квантизация
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_peft():
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def format_prompt(example):
    return f"### Вопрос: {example['question']}\n### Контекст: {' '.join(example['context'])}\n### Ответ: {example['long_answer']}"

def train():
    model, tokenizer = load_model()
    peft_config = setup_peft()
    
    # Подготовка модели
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Загрузка данных
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    train_data = dataset["train"].shuffle().select(range(1000))  # Пример для быстрого запуска
    train_data = train_data.map(lambda x: {"text": format_prompt(x)})
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_32bit",
        report_to="none"
    )
    
    # Тренировщик
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=1024
    )
    
    # Обучение
    trainer.train()
    
    # Сохранение
    model.save_pretrained("models/adapters/pubmed_lora")
    print("Adapter saved")

if __name__ == "__main__":
    train()