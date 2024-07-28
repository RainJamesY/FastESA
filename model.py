# coding=UTF8

from transformers import (AutoTokenizer,
                          BartForConditionalGeneration)

def get_prediction(test_example):
    """define parameters"""
    batch_size = 32
    epochs = 5
    max_input_length = 512  # 最大输入长度
    max_target_length = 128  # 最大输出长度
    learning_rate = 1e-04
    print("ok1")
    # 加载tokenizer,中文bart使用bert的tokenizer
    tokenizer =

    # 加载训练好的模型
    model =
    test_examples = test_example
    inputs = tokenizer(
        test_examples,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    # 生成
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
    # 将token转换为文字
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output_str = [s.replace(" ", "") for s in output_str]
    print(type(output_str))
    print(output_str[0])

    return output_str[0]