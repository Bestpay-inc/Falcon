"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time

import torch

torch.set_printoptions(linewidth=500)

import sys
sys.path.append('file_path')

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from models.falcon_model import FaModel
from models.kv_cache import initialize_past_key_values
from models.utils import *
from models.choices import *



from collections import defaultdict
def get_num_accept_length(accept_length_stat):
    stat = defaultdict(int)
    for k in accept_length_stat.keys():
        for length in accept_length_stat[k]:
            stat[length]+=1
    print("Statistic Result: ")
    print("<accept length: number of choices>")
    print(stat)
    s = 0
    num = 0
    for k in stat.keys():
        s+=k*stat[k]
        num+=stat[k]
    print("average accept length:", s/num)

def fa_forward(input_ids, model, tokenizer, tree_choices, logits_processor=None, args=None, max_steps=512, dic={}, dic_length={}):
    
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.fa_layer.reset_kv()
    
    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices, device=model.base_model.model.layers[-1].self_attn.q_proj.weight.device, k_mask=args.k_mask
        )
        
                
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.base_model.lm_head.weight.device)
        
   # print('tree_buffers:', tree_buffers)
    
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices
    
        
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
        
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data
    
    
    input_len = input_ids.shape[1]
    #print('input_len', input_len)
    
    reset_tree_mode(model)
    
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, tree_buffers["tree_attn_mask"], past_key_values, logits_processor, k_mask=args.k_mask
    )
    
    new_token = 0
    accept_len = 0
    for idx in range(max_steps):
        
        #print("**********")
        #print('idx:', idx)
        #print("**********")

        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits,
            tree_buffers["tree_indices"],
            tree_buffers["retrieve_indices"],
            sample_token,
            logits_processor,
            k_mask=args.k_mask
        )
                
        #print("**********")
        #print("tree_decoding")
        #print("**********")
        #print('input_ids.shape:', input_ids.shape)
        
        logits, hidden_state_new, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_buffers["tree_position_ids"],
            input_ids,
            tree_buffers["retrieve_indices_head"],
            current_length_data,
            idx,
            k_mask=args.k_mask
        )
       
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
            tree_candidates, tree_buffers["b_indices"], k_mask=args.k_mask
        )
        

        dic[best_candidate.item()]+=1  # 统计路径被选择的次数
        dic_length[best_candidate.item()].append(accept_length.item())  # 如果路径被选择，统计它的accept length
        accept_len+=accept_length
        
        #print("**********")
        #print("update_inference")
        #print("**********")
        #print('accept length', accept_length)
        input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            tree_buffers["retrieve_indices"],
            logits_processor,
            logits,
            tree_logits,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state,
            hidden_state_new,
            sample_p,
            past_key_values,
            k_mask=args.k_mask
        )

        
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        if input_ids.shape[1] > 1960:
            break
    
    return input_ids, new_token, idx, accept_len, dic, dic_length


def run_eval(
        base_model_path,
        fa_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        tree_choices,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    
    #shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1
    
    # print('use_ray:', use_ray)

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []

    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                fa_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                tree_choices,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        fa_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        tree_choices,
        args
):
    # temperature = 0.0

    model = FaModel.from_pretrained(
        base_model_path=base_model_path,
        fa_model_path=fa_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        tree_choice=tree_choices
    )
    
    print('model:', model)

    tokenizer = model.get_tokenizer()
    
    #print('tokenizer:', tokenizer)
    
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None
    
    model.eval()
    #print('model.eval():', model.eval())
    
    print('Check model training state:', model.training)

    question = questions[0]
    from collections import defaultdict
    dic=defaultdict(int)
    dic_length = defaultdict(list)
    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        conv = get_conversation_template("vicuna")
        # print('conv:', conv)
        
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        #print('len(question["turns"]):', len(question["turns"]))
        
        for j in range(1):
            qs = question["prompt"]
            
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            
            prompt = conv.get_prompt()
            
            input_ids = tokenizer([prompt]).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, accept_len, dic, dic_length = fa_forward(
                torch.as_tensor(input_ids).cuda(),
                model,
                tokenizer,
                tree_choices,
                logits_processor,
                args=args,
                dic=dic,
                dic_length=dic_length
            )
            
            #print('input_ids.shape:', torch.as_tensor(input_ids).shape)
            #print('output_ids.shape:', output_ids.shape)
            #print('new_token:', new_token)
            #print('idx:', idx)
            #print('accept_len:', accept_len)
            # print('dic:', dic)
            # print('dic_length:', dic_length)
            
            
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # print('output_ids', output_ids)
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            #print('output:', output)
            
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')
    #sorted_dic_path = sorted(dic.items(), key=lambda x:x[1], reverse=True)
    #print('accept path stat:', sorted_dic_path)
    #sorted_dic_length = {key: dic_length[key] for key in sorted(dic)}
    #print('accept length stat:', sorted_dic_length)
    # questions=questions[6:]
    qs_id = 0
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            accept_len_list = []
            for j in range(1):
                qs = question["prompt"]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    
                    output_ids, new_token, idx, accept_len, dic, dic_length = fa_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        tree_choices,
                        logits_processor,
                        args=args,
                        dic=dic,
                        dic_length=dic_length
                    )
                    
                    # print(dic)
                    # print(dic_length)
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                accept_len_list.append(accept_len.item())
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "accept tokens": accept_len_list,"wall_time": wall_time})

        
        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": qs_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
        qs_id+=1
    sorted_dic_path = sorted(dic.items(), key=lambda x:x[1], reverse=True)
    print('accept path stat:', sorted_dic_path)
    sorted_dic_length = {key: dic_length[key] for key in sorted(dic)}
    print('accept length stat:', sorted_dic_length)
    get_num_accept_length(sorted_dic_length)

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fa-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/70B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="fa-vicuna-7b-0502")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--k_mask",
        type=int,
        default=2,
    )
    

    args = parser.parse_args()
    
    # print('args:', args)

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)
    
    # print('args.tree_choices:', args.tree_choices)
    
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()
    
    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    
    
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    print(f"Tree paras:\nk_mask: {args.k_mask}\ntree choice: {args.tree_choices}")
    run_eval(
        args.base_model_path,
        args.fa_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,

        args.temperature,
        args.tree_choices,
        args
    )

    reorg_answer_file(answer_file)
