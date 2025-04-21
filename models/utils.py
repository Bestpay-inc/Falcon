import secrets  # 添加secrets模块导入

# typing 
from typing import List, Tuple
import time
import torch
from collections import defaultdict
# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")
#torch.set_printoptions(profile="full")

torch.set_printoptions(linewidth=500)
TOPK = 20  # topk for sparse tree
# print('TOPK', TOPK)

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        return result

    return wrapper


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list

def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def get_node_with_child(sorted_tree_choices, k_mask=1):
    nodes_with_child = []
    for n in sorted_tree_choices:
        if len(n)<=k_mask:
            continue
        parent = n[:-k_mask]
        if parent not in nodes_with_child:
            nodes_with_child.append(parent)
    return nodes_with_child

def get_parent_idx(nodes_into_layer):
    num_layer = len(nodes_into_layer)
    ret = [[] for i in range(num_layer)]
    for i in range(1,num_layer):
        for idx in range(len(nodes_into_layer[i])):
            parent = nodes_into_layer[i][idx][:-1]
            ret[i].append(nodes_into_layer[i-1].index(parent))
    return ret

def process_node_into_layer(nodes_with_child):
    # ret = []
    max_len = len(nodes_with_child[-1])
    ret = [[] for i in range(max_len)]
    for i in range(len(nodes_with_child)):
        ret[len(nodes_with_child[i])-1].append(nodes_with_child[i])
    return ret

def get_kparent_info(node_wc_layer, k_mask):
    n = len(node_wc_layer)
    parent_info = [[] for i in range(n)]
    for i in range(n):
        if i%k_mask!=0 or i==0:
            continue
        children = node_wc_layer[i]
        n = len(children)

        grand_parents = node_wc_layer[i-k_mask]
        for k in range(n): 
            query_parent = children[k][:-k_mask]
            index = grand_parents.index(query_parent)
            parent_info[i].append(index)

def generate_tree_buffers(tree_choices, device="cuda", k_mask=2):
    
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))

    tree_len = len(sorted_tree_choices) + k_mask 

    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    height = len(sorted_tree_choices[-1])
    detailed_depth_counts = [[] for i in range(height)]
    detailed_depth_counts[0].append(depth_counts[0])

    cur_detailed = defaultdict(int)
    for node in sorted_tree_choices:
        if len(node)==1:
            continue
        cur_detailed[tuple(node[:-1])]+=1

    sorted_keys = sorted(cur_detailed.keys(), key=lambda x:(len(x), x))
    for key in sorted_keys:
        detailed_depth_counts[len(key)].append(cur_detailed[key])

    node_wc_layer = process_node_into_layer(sorted_tree_choices)

    total_parent_info = [[] for i in range(len(node_wc_layer))]
    for i in range(len(total_parent_info)):
        if (i+1)==k_mask:
            children = node_wc_layer[i]
            n = len(children)
            parent_info = [0 for _ in range(n)]
            grand_parents = node_wc_layer[i] 
            for k in range(n):
                query_parent = children[k][:-k_mask]
                index = grand_parents[k][0]
                parent_info[k] = index
            total_parent_info[i]=parent_info
            continue
        if (i+1)%k_mask!=0: 
            continue
        children = node_wc_layer[i]
        n = len(children)
        offset = 0

        parent_info = [0 for _ in range(n)]
        grand_parents = node_wc_layer[i-k_mask]
        for k in range(n): 
            query_parent = children[k][:-k_mask]
            index = grand_parents.index(query_parent)
            parent_info[k] = index
        total_parent_info[i]=parent_info

    closest_parent_info = [[] for i in range(len(node_wc_layer))] 
    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    for i in range(k_mask):
        tree_indices[i] = i
    p_indices = [0 for _ in range(tree_len - k_mask)]
    b_indices = [[] for _ in range(tree_len - k_mask)]
    offset = k_mask
    start_idx = k_mask
    bias = 0
    for i in range(len(detailed_depth_counts)):
        if i!=0:
            offset += TOPK
        for j in range(len(detailed_depth_counts[i])):
            num_top = detailed_depth_counts[i][j]
            for k in range(num_top):
                tree_indices[start_idx] = offset + k
                start_idx+=1
            
    p_indices = [-1]*k_mask + p_indices

    tree_indices = [_ for _ in range(k_mask)]

    skip_offset = k_mask*TOPK
    init_offset = k_mask + k_mask*TOPK
    for i in range(len(node_wc_layer)):
        children = node_wc_layer[i]
        n = len(children)
        parent_info = [0 for _ in range(n)]
        if len(children[0])==1:  
            first_layer_offset = k_mask
            for j in range(n):
                tree_indices.append(first_layer_offset)
                first_layer_offset+=1
            continue
        if (i+1)<=k_mask:
            offset = k_mask + i*TOPK
            cur_parent = None
            for j in range(n):
                parent = children[j][:-1]
                if not cur_parent or parent==cur_parent:
                    cur_parent = parent
                    tree_indices.append(offset)
                    offset+=1
                else:
                    cur_parent = parent
                    offset = k_mask+i*TOPK
                    tree_indices.append(offset)
                    offset+=1
        else:
            if i == k_mask:
                last_offset = init_offset
                small_offset = 0
                cur_parent = None
                for j in range(n):
                    parent = children[j][:-1]
                    if not cur_parent or parent==cur_parent:
                        cur_parent = parent
                        tree_indices.append(last_offset+small_offset)
                        small_offset+=1
                    else:
                        cur_parent = parent
                        small_offset = 1
                        last_offset+=skip_offset
                        tree_indices.append(last_offset)

                continue
            if i%k_mask!=0:
                init_offset+=TOPK
                last_offset = init_offset
                small_offset = 0
                cur_parent = None
                cur_grand_parent = None
                for j in range(n):
                    grand_parent = children[j][:-(i%k_mask+1)]
                    parent = children[j][:-1]
                    if not cur_grand_parent or grand_parent==cur_grand_parent:
                        cur_grand_parent = grand_parent
                        if not cur_parent or cur_parent!=parent:
                            cur_parent = parent
                            small_offset = 0
                        elif cur_parent == parent:
                            small_offset+=1
                        tree_indices.append(last_offset+small_offset)
                    else:
                        cur_grand_parent = grand_parent
                        cur_parent = parent
                        small_offset = 0
                        last_offset+=skip_offset
                        tree_indices.append(last_offset)
                
                
            else:
                last_offset = TOPK * (last_offset//TOPK) + k_mask + TOPK
                init_offset = last_offset
                small_offset = 0
                cur_parent = None
                for j in range(n):
                    parent = children[j][:-1]
                    if not cur_parent or parent==cur_parent:
                        cur_parent = parent
                        tree_indices.append(last_offset+small_offset)
                        small_offset+=1
                    else:
                        cur_parent = parent
                        small_offset = 1
                        last_offset+=skip_offset
                        tree_indices.append(last_offset)
    tree_indices = torch.tensor(tree_indices)   

    tree_attn_mask = torch.eye(tree_len, tree_len)

    tree_attn_mask[:, 0:k_mask] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]

            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + k_mask)
            tree_attn_mask[j + start + k_mask, ancestor_idx] = 1

        start += depth_counts[i]
    for i in range(tree_len):
        for j in range(i,tree_len):
            if j>i:
                tree_attn_mask[i,j]=0

    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    for i in range(k_mask):
        tree_position_ids[i]=i
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + k_mask: start + depth_counts[i] + k_mask] = i + k_mask
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])

    retrieve_indices = [pad_path(path, max_length, -k_mask-1) for path in retrieve_indices_nest]

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + k_mask
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], k_mask), dtype=torch.long), retrieve_indices],
                                    dim=1)
    for i in range(retrieve_indices.shape[0]):
        for j in range(k_mask):
            retrieve_indices[i,j]=j

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):

        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()

    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]

    p_indices_new = p_indices_new.tolist()

    b_indices = [[] for i in range(k_mask)] + b_indices
    b_indices_new = []

    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if False:
                #if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }
    
    for k in tree_buffers.keys():
        print(k)
        print(tree_buffers[k])

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    
        
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new

    return tree_buffers


def initialize_tree(input_ids, model, tree_attn_mask, past_key_values, logits_processor, k_mask=2):

    
    tree_logits, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor, k_mask=k_mask
    )
    
    model.base_model.model.tree_mask = tree_attn_mask
    
    return tree_logits, logits, hidden_state, sample_token


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor, k_mask=2):
    
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token.view(-1)
    
    candidates_tree_logits = tree_logits[0]
    
    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)
    
    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = tree_logits[1]
        
        candidates_prob = torch.cat(
            [torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32), candidates_tree_prob.view(-1)],
            dim=-1)

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [tree_candidates_prob, torch.ones((1), dtype=torch.float32, device=tree_candidates_prob.device)], dim=0)
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None
    # Unsqueeze the tree candidates for dimension consistency.
    
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        current_length_data,
        idx,
        k_mask=2,
):

   
    current_length_data.fill_(input_ids.shape[-1]-k_mask+1)
    
    position_ids = tree_position_ids + input_ids.shape[1]-k_mask+1
    
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        init=False,
        k_mask=k_mask
    )
    
    logits = tree_logits[0, retrieve_indices]
    
    return logits, hidden_state, outputs


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        cart_candidates_prob,
        op,
        p_indices,
        tree_candidates,
        b_indices,
        k_mask=2
) -> Tuple[torch.Tensor, int]:
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    
    
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        
        posterior_mask = (
                #candidates[:, k_mask:].to(logits.device) == torch.argmax(logits[:,(k_mask-1):-(k_mask-1)], dim=-1)
                candidates[:, k_mask:].to(logits.device) == torch.argmax(logits[:,(k_mask-1):-1], dim=-1)
        ).int()
        print("torch.argmax(logits[:,(k_mask-1):-1], dim=-1)",torch.argmax(logits[:,(k_mask-1):-1], dim=-1))
        
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        
        accept_length = candidates_accept_length.max()

        # Choose the best candidate

        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

        return best_candidate, accept_length, logits[best_candidate, accept_length+k_mask-1]

    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = secrets.randbelow(10000) / 10000  # 使用secrets模块替代random.random()
                    px = gtp[xi]
                    qx = cart_candidates_prob[j, i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        q = op[i - 1][p_indices[j][i]].clone()
                        b = b_indices[j][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            q[mask] = 0
                            q = q / q.sum()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length-1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state,
        hidden_state_new,
        sample_p,
        past_key_values,
        k_mask=2
):

    prev_input_len = input_ids.shape[1]-(k_mask-1)
    
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + k_mask] + prev_input_len)

    input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, k_mask-1: accept_length + k_mask].to(input_ids.device)], dim=-1)
    
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    
    if accept_length+1 < k_mask:
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, :accept_length+k_mask]
    else:
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, k_mask-1:accept_length+k_mask]
    
    
    for past_key_values_data in past_key_values_data_list:
                
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]

        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    current_length_data.fill_(prev_input_len + tgt.shape[-2])
    
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    
    input_ids_new = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    
    token_return = input_ids_new[:, -k_mask:]
     
    tree_logits = model.fa_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=input_ids_new,
                                              head=model.base_model.lm_head, logits_processor=logits_processor, accept_length=accept_length, k_mask=k_mask)
  
    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, accept_hidden_state_new, token_return
        

if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
