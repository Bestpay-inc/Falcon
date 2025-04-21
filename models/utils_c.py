import torch

# typing
from typing import List

TOPK = 20  # topk for sparse tree

def get_node_has_child(tree_choices):
    res = []
    for node in tree_choices:
        if len(node)==1:
            continue
        n = len(node)
        parent = node[:n-1]
        if parent not in res:
            res.append(parent)
    return res

def get_tree_attn_mask(tree_choices, k_mask):
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        
    tree_len = len(sorted_tree_choices*k_mask)  # [debug]

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    tree_attn_mask = torch.eye(tree_len, tree_len)

    # 初始化：主对角线置为1
    for i in range(0,tree_len,k_mask):
        tree_attn_mask[i:i+k_mask,i:i+k_mask]=1

    # 处理下三角部分
    start=0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):

            cur_tree_choice = sorted_tree_choices[start + j]
            # 处理根结点的子节点
            if len(cur_tree_choice) == 1:
                for k in range(k_mask):
                    tree_attn_mask[k_mask*(0+j)+k, k_mask*(0+j):k_mask*(0+j)+k_mask]=1
                continue

            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                for k in range(k_mask):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1])*k_mask + k)  

            for k in range(k_mask):
                for kk in range(len(ancestor_idx)//k_mask):
                    tree_attn_mask[j*k_mask + start*k_mask + k, ancestor_idx[kk*k_mask]:ancestor_idx[kk*k_mask+k_mask-1]+1] = 1
                    
        start += depth_counts[i]

    return tree_attn_mask


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

class node:
    def __init__(self,parent=None,value=None,dict_key=None):
        self.parent=parent
        self.value=value
        if parent:
            self.depth=parent.depth+1
            parent.children.append(self)
        else:
            self.depth=0
        self.children=[]
        self.dict_key=dict_key
    def is_leaf(self):
        return len(self.children)==0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index()+[self.index]



class Tree:
    def __init__(self,tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root=node()
        self.node_dic={}
        for tree_node in sorted_tree_list:
            cur_value=tree_node[-1]
            if len(tree_node)==1:
                cur_node=node(parent=self.root,value=cur_value,dict_key=tuple(tree_node))
            else:
                cur_parent=self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value,dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c=0
        
        #print('self.node_dic:', self.node_dic)
        
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c+=1
        return num_c

    def get_node_wchild(self):
        ns=[]
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index=0
        for key in self.node_dic:
            cur_node=self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index=cur_index
                cur_index+=1

def get_node_with_child(sorted_tree_choices, k_mask=1):
    nodes_with_child = []
    for n in sorted_tree_choices:
        if len(n)<=k_mask:
            continue
        parent = n[:-k_mask]
        if parent not in nodes_with_child:
            nodes_with_child.append(parent)
    #print('nodes_with_child', nodes_with_child)
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

def generate_tree_buffers(tree_choices, device="cuda", k_mask=2):
    
    tree=Tree(tree_choices)
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = tree.num_node_wchild()  # 10
    #print('utils_c tree_len', tree_len)
    max_depth=tree.max_depth()
    nodes_wc=tree.get_node_wchild()
    nodes_wc_list = get_node_with_child(sorted_tree_choices)
    nodes_wgc_list = get_node_with_child(sorted_tree_choices, k_mask=k_mask) 
    node_wc_layer = process_node_into_layer(nodes_wc_list)

    depth_counts=[0 for _ in range(max_depth-1)] 
    for x in nodes_wc:
        depth_counts[x.depth-1]+=1
    depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]  
    tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
    start = 0
    bias = 1
    offset = 0
    for i in range(len(depth_counts)):
        bias = i
        repeat_j = 0
        for j in range(depth_counts[i]):
            cur_node = nodes_wc_list[start + j]

            if len(cur_node)==1:
                tree_indices_list[i][j] = cur_node[0]
                continue
            cur_parent = cur_node[:-1]
            if j != 0:
                if cur_parent != parent:
                    offset = 0
                    parent = cur_parent
                    if (i+1)%k_mask==0:

                        repeat_j=j
            else:
                parent = cur_parent
            tree_indices_list[i][j] = cur_node[-1]
            offset += 1

        offset = 0
        start += depth_counts[i]

    token_repeat_indices_list = [[] for i in range(len(node_wc_layer))]
    total_parent_info = [[] for i in range(len(node_wc_layer))]
    for i in range(len(token_repeat_indices_list)):
        if (i+1)%k_mask!=0:
            continue
        
        for j in range(len(node_wc_layer[i])):
            for k in range(k_mask):
                token_repeat_indices_list[i].append(k*TOPK+node_wc_layer[i][j][-(k_mask-k)])

    for i in range(len(token_repeat_indices_list)):
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
        for j in range(n):
            for _ in range(k_mask):
                token_repeat_indices_list[i][k_mask*j+_]+=parent_info[j]*k_mask*TOPK

    hs_repeat_indice_list = [[] for i in range(len(node_wc_layer))]
    for i in range(len(hs_repeat_indice_list)):
        if (i+1)%k_mask!=0:
            continue
        for j in range(len(node_wc_layer[i])):
            offset = 0
            for k in range(k_mask-1):
                parent = node_wc_layer[i][j][:-(k_mask-k-1)]


                parent_idx = node_wc_layer[i-(k_mask-k-1)].index(parent)
                hs_repeat_indice_list[i].append(offset+parent_idx)
                offset+=len(node_wc_layer[i-(k_mask-k-1)])
            hs_repeat_indice_list[i].append(offset+j)

    tree_attn_mask = torch.eye(tree_len, tree_len)

    for id,x in enumerate(nodes_wc):
        tree_attn_mask[id,x.all_index()]=1

    tree_attn_mask_list=[torch.zeros([len(ml),len(ml)]) for ml in token_repeat_indices_list]

    for i in range(len(tree_attn_mask_list)):

        W,H = tree_attn_mask_list[i].shape
        assert W%k_mask==0 and W==H
        for j in range(W//k_mask):
            tree_attn_mask_list[i][j*k_mask:j*k_mask+k_mask, j*k_mask:j*k_mask+k_mask]=1

    for i in range(len(tree_attn_mask_list)):
        if (i+1)%k_mask==0 and (i+1)!=k_mask:
            mask_cache = []
            idx = total_parent_info[i]
            for index,j in enumerate(idx):
                mask_cache.append(tree_attn_mask_list[i-k_mask][j*k_mask:j*k_mask+k_mask, :])
            tensor = torch.cat(mask_cache, dim=0)
            tree_attn_mask_list[i] = torch.cat([tensor,tree_attn_mask_list[i]], dim=1)
     
    position_ids = [torch.zeros([len(ml)]) for ml in token_repeat_indices_list]
    pos_tensor = torch.tensor([i for i in range(k_mask)])
    for i in range(len(position_ids)):
        if (i+1)%k_mask==0:
            assert position_ids[i].shape[0]%k_mask==0
            for j in range(position_ids[i].shape[0]//k_mask):
                position_ids[i][j*k_mask:j*k_mask+k_mask] = pos_tensor
  
    for i in range(len(token_repeat_indices_list)):
        token_repeat_indices_list[i] = torch.tensor(token_repeat_indices_list[i])
        hs_repeat_indice_list[i] = torch.tensor(hs_repeat_indice_list[i])
    tree_buffers = {
            "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
            "tree_indices": tree_indices_list,
            "position_ids":position_ids,
            "token_repeat_indices":token_repeat_indices_list,
            "hs_repeat_indices":hs_repeat_indice_list,
            "total_parent_info":total_parent_info
        }
    print('small model tree buffer')
    for k in tree_buffers.keys():
        print(k)
        print(tree_buffers[k])

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: [i.clone().to(device) for i in v]
        if isinstance(v[0], torch.Tensor)
        else (
            torch.tensor(v, device=device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in tree_buffers.items()
    }
    return tree_buffers


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



if __name__=="__main__":
    from choices import mc_sim_7b_63
    a=generate_tree_buffers(mc_sim_7b_63)
    print(a)