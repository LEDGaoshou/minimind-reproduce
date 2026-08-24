# DeepSpeed 版 GRPO 训练（基于 train_grpo.py 改造）
# 与原版的差异：
#   1. 分布式/优化器/调度器/混合精度/梯度累积/梯度裁剪全部由 DeepSpeed 接管（config: ds_config/grpo.json）
#   2. rollout 推理用 model_engine.module（DeepSpeed engine 无 generate），与 engine 共享参数
#   3. 断点续训用 DeepSpeed 原生 save_checkpoint/load_checkpoint（含 optimizer/scheduler 状态）
#   4. swanlab 带未安装/未登录降级保护
# 启动方式（deepspeed launcher，需先 pip install deepspeed）：
#   deepspeed --include localhost:0,1 code/trainer/train_grpo_ds.py --ds_config ds_config/grpo.json
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import json
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, model_engine, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # policy 前向：不传 attention_mask，让 FlashAttention(is_causal) 生效，
        # 避免显式 scores 矩阵 O(seq^2) 爆显存；pad 位置在 loss 处由 completion_mask 屏蔽
        res = model_engine(outputs)
        aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

        # DeepSpeed：loss 除以 accumulation_steps 保持与原版相同的数值语义；
        # backward/step 每个 micro-batch 都调用，梯度累积/裁剪由 DeepSpeed 在内部处理
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        model_engine.backward(loss)
        model_engine.step()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]["lr"]

            if torch.cuda.is_available():
                mem_info = (f", GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f}/"
                            f"{torch.cuda.memory_reserved() / 1e9:.2f} GB")
            else:
                mem_info = ""

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, '
                   f'Learning Rate: {current_lr:.8f}{mem_info}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # 保存：① torch 格式权重（供 eval_llm.py 直接加载）② DeepSpeed 原生 checkpoint（含优化器状态，供续训）
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model_engine.module.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = getattr(model_engine.module, '_orig_mod', model_engine.module)
            torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
            Logger(f'[Save] 权重已保存: {ckp}')
            model_engine.module.train()

        if step % args.save_interval == 0 or step == iters:
            ds_ckpt_dir = os.path.join(args.save_dir, "ds_checkpoints")
            model_engine.save_checkpoint(ds_ckpt_dir, tag=f"epoch{epoch}_step{step}")
            if is_main_process():
                torch.save({"epoch": epoch, "step": step}, os.path.join(ds_ckpt_dir, "meta.pt"))

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model_engine.module)

        # 周期性释放缓存碎片（缓解渐进式显存增长）；reserved-allocated 差值大说明碎片多
        if args.empty_cache_interval > 0 and step % args.empty_cache_interval == 0:
            torch.cuda.empty_cache()

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (DeepSpeed 版)")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="micro batch size（每个 GPU 每个累积步的 batch）")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备（deepspeed 下由 launcher 指定，此参数仅兜底）")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（rollout/ref 推理用）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数（写入 ds_config）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（写入 ds_config）")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="./data/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="./code/model/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是，走DeepSpeed checkpoint）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用swanlab实验跟踪")
    parser.add_argument("--wandb_mode", type=str, default="local", choices=["local", "online", "offline"],
                        help="swanlab运行模式：local=数据存本地(默认，无需登录)；online=上传swanlab.cn(需先swanlab login)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="swanlab项目名")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--empty_cache_interval", type=int, default=50,
                        help="每隔多少step调用一次torch.cuda.empty_cache()释放碎片（0=禁用，缓解渐进式显存增长）")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    parser.add_argument("--tokenizer_path", type=str, default="./model_learn_tokenizer", help="自定义Tokenizer路径")
    parser.add_argument("--is_mini", type=bool, default=True, help="是否为mini数据集训练的模型")
    parser.add_argument("--ds_config", type=str, default="./ds_config/grpo.json", help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="deepspeed launcher 传入的本地 rank")
    args = parser.parse_args()

    # ========== 1. 初始化分布式（DeepSpeed 接管） ==========
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"
    setup_seed(42 + dist.get_rank())

    # ========== 2. 配置目录、模型参数 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))

    # ========== 3. rollout/ref 推理的混合精度上下文（训练精度由 DeepSpeed 管理） ==========
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配 swanlab（带降级保护） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
        except ImportError as e:
            Logger(f"swanlab 未安装：{e}。执行 `pip install swanlab` 后再使用 --use_wandb")
            wandb = None
        else:
            wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
            try:
                wandb.init(project=args.wandb_project, name=wandb_run_name, mode=args.wandb_mode)
                Logger(f"swanlab 初始化成功（mode={args.wandb_mode}，project={args.wandb_project}）")
                if args.wandb_mode == "local":
                    Logger("本地查看：运行 `swanlab watch` 或访问 http://127.0.0.1:5092 （数据在 ./swanlog）")
            except Exception as e:
                Logger(f"swanlab 初始化失败（mode={args.wandb_mode}）：{e}")
                wandb = None

    # ========== 5. 初始化模型 ==========
    base_weight = args.from_weight
    # Policy 模型（将交给 DeepSpeed 管理）
    model, tokenizer = init_model(lm_config, base_weight, device=args.device, is_mini=args.is_mini, tokenizer_path=args.tokenizer_path)
    # Reference 模型（纯推理，不参与训练）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device, is_mini=args.is_mini, tokenizer_path=args.tokenizer_path)
    ref_model = ref_model.to(dtype).eval().requires_grad_(False)
    # Reward 模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # ========== 6. 数据与 DeepSpeed 初始化 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs

    # 读取 ds_config 并按命令行参数覆盖关键字段
    ds_config = json.load(open(args.ds_config, encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.accumulation_steps
    ds_config["gradient_clipping"] = args.grad_clip
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["total_num_steps"] = total_optimizer_steps

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, config=ds_config)
    Logger(f'DeepSpeed 初始化完成: ZeRO stage={ds_config["zero_optimization"]["stage"]}, '
           f'micro_batch={args.batch_size}, accumulation={args.accumulation_steps}, '
           f'total_optimizer_steps={total_optimizer_steps}')

    # ========== 7. 从 DeepSpeed checkpoint 恢复 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume == 1:
        ds_ckpt_dir = os.path.join(args.save_dir, "ds_checkpoints")
        if os.path.exists(ds_ckpt_dir) and model_engine.load_checkpoint(ds_ckpt_dir):
            meta_path = os.path.join(ds_ckpt_dir, "meta.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location="cpu")
                start_epoch, start_step = meta.get("epoch", 0), meta.get("step", 0)
            Logger(f'[Resume] 从 DeepSpeed checkpoint 恢复: epoch {start_epoch}, step {start_step}')
        else:
            Logger("[Resume] 未找到 DeepSpeed checkpoint，从头开始训练")

    # ========== 8. Rollout 引擎（policy 用 engine.module，与 engine 共享参数） ==========
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model_engine.module,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    rollout_engine.update_policy(model_engine.module)

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, model_engine, start_step, wandb, use_sglang=(args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, model_engine, 0, wandb, use_sglang=(args.rollout_engine == "sglang"))

    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
