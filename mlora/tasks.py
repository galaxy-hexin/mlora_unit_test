from mlora.modelargs import LoraBatchDataConfig, MultiLoraBatchData
from mlora.tokenizer import Tokenizer
from mlora.prompter import Prompter
from mlora.model import LLMModel

from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import datasets as hf_datasets
import evaluate as hf_evaluate
import logging
import torch


class BasicTask():
    def __init__(self) -> None:
        pass

    def dataload_function(self, data_point) -> Tuple:
        return None, None, {"bos": True, "eos": True}

    def init_kwargs(self) -> Dict:
        return {}


# Casual Fine-tuning Tasks
class CasualTask(BasicTask):
    def __init__(self, prompter: Prompter = None) -> None:
        super().__init__()
        self.prompter_ = prompter

    def dataload_function(self, data_point) -> Tuple:
        return (self.prompter_.generate_prompt(
            data_point["instruction"],
            data_point.get("input", None),
            data_point.get("output", None)),
            None, {"bos": True, "eos": True})

# 这里有一个函数名参数，意味着这个函数可以使用不同的数据加载方式；callable是表示这个变量可调用
# Sequence Classification
class SequenceClassification(BasicTask):
    def __init__(self,
                 task_type: str,
                 label_dtype: torch.dtype,
                 num_labels: int,
                 dataload_function: Callable) -> None:
        super().__init__()
        self.dataload_function_ = dataload_function
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type

    def dataload_function(self, data_point) -> Tuple:
        return self.dataload_function_(data_point)

    def init_kwargs(self) -> Dict:
        return {
            "task_type": self.task_type_,
            "num_labels": self.num_labels_,
            "label_dtype": self.label_dtype_,
        }


#一个任务分类字典
#glue是一个自然语言理解基准和分析平台
#https://zhuanlan.zhihu.com/p/135283598
#cola，mnli等均为glue中的测试任务，比如cola是单句子分类任务
    

classification_tasks = {
    "glue:cola": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True}
        ),
    ),

    "glue:mnli": SequenceClassification(
        task_type="single_label_classification",
        num_labels=3,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["premise"], data_point["hypothesis"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),

    "glue:mrpc": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence1"], data_point["sentence2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),

    "glue:qnli": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["question"], data_point["sentence"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),

    "glue:qqp": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["question1"], data_point["question2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),

    "glue:rte": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence1"], data_point["sentence2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),

    "glue:sst2": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),

    "glue:wnli": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence1"] + " </s> " + data_point["sentence2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
}


#评估方案参数配置
'''
适配器名；任务类型；批处理大小；
任务类型；数据集；评估集；批处理起始id，批处理结束id；

task_type是一个任务名，范围对应classification类中的关键字
self.data_ = hf_datasets.load_dataset(result[0], result[1])["validation"]
上述语句是加载[0]数据集中的[1]子集，比如加载glue合集中的子集cola；

主要是记录训练数据和评估方案
'''
@dataclass
class EvaluateConfig:
    adapter_name_: str = None
    task_type_: str = None
    batch_size_: int = 16
    # Do not set these manually
    task_: SequenceClassification = None
    data_: hf_datasets.Dataset = None
    metric_: hf_evaluate.EvaluationModule = None
    batch_start_idx_: int = 0
    batch_end_idx_: int = 0

    def init_task(self):
        self.task_ = classification_tasks[self.task_type_]
        if ':' in self.task_type_:
            result = self.task_type_.split(':')
            self.data_ = hf_datasets.load_dataset(
                result[0], result[1])["validation"]
            self.metric_ = hf_evaluate.load(result[0], result[1])
        else:
            self.data_ = hf_datasets.load_dataset(
                self.task_type_)["validation"]
            self.metric_ = hf_evaluate.load(self.task_type_)

    def dataload(self, data_point):
        return self.task_.dataload_function_(data_point)


'''
二次补充：
我个人认为这个就是用来统一接口的（）

bacth 是一次性处理的信息集
batch size是这一批的数据的大小
batch_start_id &batch_end_id是在数据集里的位置，
数据集里每一个index都是一个数据
因此一个config会先把自己start-end的数据都拿出来用Tokenizer进行embeddeding编码
label 是数据集的标签，不同的数据集有不同的标签，比如cola的标签为0，1，表示假，真
kwargs 目前猜测是一个标志位

这个函数的目的，是为了将系列evaluateConfig拆分成连续的config（config的start和end相等），
和连续并且对应的label
还有对应的multilorabatchdata（这个里面要包含完整的token序列，以及对应的batchdata序列，以及对应的mask配置）

'''
def _dispatch_task_in(tokenizer: Tokenizer, configs: List[EvaluateConfig], max_seq_len: int):
    batch_data_config = []
    current_configs = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []
    for config in configs:
        if config.batch_start_idx_ >= len(config.data_):
            continue
        config.batch_end_idx_ = min(
            config.batch_start_idx_ + config.batch_size_, len(config.data_))
        batch_start_idx = len(batch_tokens)

        for idx in range(config.batch_start_idx_, config.batch_end_idx_):
            if idx >= len(config.data_):
                break
            texts, labels, kwargs = config.dataload(config.data_[idx])

            if "eos" in kwargs:
                kwargs["eos"] = False
            tokens = tokenizer.encode(texts, **kwargs) #**dict字典一次性传参

            #词条太长插入最大词条值，太短则用填充符补充
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            while len(tokens) < max_seq_len:
                tokens.append(tokenizer.pad_id_)

            batch_tokens.append(tokens)
            atten_masks.append(tokenizer.mask_from(tokens))
            batch_labels.append(labels.copy())

        config.batch_start_idx_ = config.batch_end_idx_
        current_configs.append(config)
        batch_data_config.append(LoraBatchDataConfig(adapter_name_=config.adapter_name_,
                                                     batch_start_idx_=batch_start_idx, batch_end_idx_=len(batch_tokens)))

    return (current_configs,
            batch_labels,
            MultiLoraBatchData(
                lora_batch_data_config_=batch_data_config,
                batch_tokens_=batch_tokens,
                attention_masks_=atten_masks,
                gradient_checkpoint_=False))



'''
@torch.inference_mode()模型评估用的，关闭版本计数器和视图追踪，能获得更好的性能
（这段暂时不太用的到，gpt给的比较清晰，看它的吧就）
这段代码是一个用于评估模型性能的函数：
1. 函数签名表明，这个函数名为 `evaluate`，接受四个参数：`model`（LLMModel 类型的模型）、`tokenizer`（Tokenizer 类型的分词器）、`configs`（EvaluateConfig 类型的配置列表）、`max_seq_len`（一个可选的整数参数，默认为 512，用于指定最大序列长度）。
2. 函数中的 `for` 循环遍历了配置列表 `configs` 中的每个配置项，通过调用 `init_task` 方法初始化任务，并检查每个配置项中的数据长度，以确定最大迭代次数 `max_iterations`。
3. 接着是一个无限循环 `while True`，其中调用 `_dispatch_task_in` 函数来处理任务，获取当前的配置、批次标签和输入参数。
4. 如果当前配置为空，则退出循环。
5. 对模型 `model` 调用 `forward` 方法，传入输入参数 `input_args`，得到模型的输出 `outputs`。
6. 将输入的 batch tokens 转换为 PyTorch 的张量 `input_ids`。
7. 循环遍历模型的输出 `outputs`，并处理每个输出。对于每个输出，获取相关的配置、任务、度量指标、起始索引、结束索引和逻辑回归结果等信息。
8. 根据任务类型，进行相应的处理。如果是单标签分类任务，则将逻辑回归结果转换为预测标签。如果任务类型不是单标签或多标签分类，则抛出 ValueError 异常。
9. 将预测值和标签传递给度量指标对象，以便更新度量指标。
10. 记录评估信息，包括适配器名称和当前评估数据的步骤信息。
11. 循环结束后，再次遍历配置列表 `configs`，记录每个配置的评估结果，并调用 `compute` 方法计算度量指标的结果。
12. 最终，将每个度量指标的名称和值记录到日志中。
这个函数主要用于评估模型在给定配置下的性能，并记录评估结果。
'''
@torch.inference_mode()
def evaluate(model: LLMModel,
             tokenizer: Tokenizer,
             configs: List[EvaluateConfig],
             max_seq_len: int = 512):
    max_iterations = 0
    for config in configs:
        config.init_task()
        if len(config.data_) > max_iterations:
            max_iterations = len(config.data_)

    while True:
        current_configs, batch_labels, input_args = _dispatch_task_in(
            tokenizer, configs, max_seq_len)

        if len(current_configs) == 0:
            break

        outputs = model.forward(input_args)

        input_ids = torch.tensor(input_args.batch_tokens_, dtype=torch.long)

        for idx, output in enumerate(outputs):
            config: EvaluateConfig = current_configs[idx]
            task: SequenceClassification = config.task_
            metric = config.metric_
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_
            logits = output.logits

            batch_size = logits.shape[0]
            sequence_lengths = (torch.eq(input_ids[start_idx:end_idx],
                                         tokenizer.pad_id_).int().argmax(-1) - 1).to(logits.device)
            pooled_logits = logits[torch.arange(batch_size,
                                                device=logits.device), sequence_lengths]
            labels = torch.tensor(batch_labels[start_idx:end_idx],
                                  dtype=task.label_dtype_, device=logits.device)
            if task.task_type_ == "single_label_classification":
                pooled_logits = torch.argmax(
                    pooled_logits, dim=-1).to(task.label_dtype_)
            elif task.task_type_ != "multi_label_classification":
                raise ValueError(f"unknown task type {task.task_type_}")

            metric.add_batch(predictions=pooled_logits.detach().cpu(),
                             references=labels.detach().cpu())
            logging.info(f"{config.adapter_name_} evaluate data:")
            logging.info(
                f"    step: {config.batch_start_idx_}/{len(config.data_)}")

    for config in configs:
        logging.info(f"{config.adapter_name_} evaluate result:")
        result = config.metric_.compute()
        for name, value in result.items():
            logging.info(f"    {name} = {value}")
