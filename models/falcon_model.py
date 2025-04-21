"""
Copyright 2025 Bestpay

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json

import torch
import torch.nn as nn
from transformers import AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import FConfig

class FaModel(nn.Module):
    """
    Falcon Acceleration Model (FaModel) implements the semi-autoregressive 
    speculative decoding framework for faster LLM inference.
    
    This model wraps base LLMs (like LLaMA, Qwen2) and enhances them with
    the Coupled Sequential Glancing Distillation technique and custom-designed
    decoding tree for faster inference while maintaining quality.
    """

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            fa_model_path,
            tree_choice=None
    ):
        """
        Initialize the Falcon Acceleration Model.
        
        Args:
            base_model: The underlying language model (LLaMA, Qwen2, etc.)
            base_model_name_or_path: Path or name of the base model
            fa_model_path: Path to the Falcon acceleration model weights
            tree_choice: Optional custom tree structure for speculative decoding
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = FConfig.from_pretrained(fa_model_path)
        self.k_mask = config.k_mask
        self.tree_choice = tree_choice
        with open(fa_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.fa_layer = Model(config,tree_choice=tree_choice,bias=bias)

        low_memory=False

        # Handle multi-device setup (e.g., for large models)
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.fa_layer.diff_device = True
            if not low_memory:
                self.fa_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.fa_layer.layer_device = device

        else:
            self.fa_layer.diff_device = False
        self.fa_layer.to(self.base_model.dtype).to(device)
        self.fa_layer.init_tree()

    def get_tokenizer(self):
        """
        Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            fa_model_path=None,
            tree_choice=None,
            **kwargs,
    ):
        """
        Load a Falcon model from pretrained weights.
        
        Args:
            Type: Model type identifier (auto-detected from config)
            base_model_path: Path to the base LLM
            fa_model_path: Path to the Falcon acceleration model
            tree_choice: Optional custom tree for decoding
            **kwargs: Additional arguments passed to the base model loader
            
        Returns:
            FaModel: Initialized Falcon acceleration model
        """
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=='Qwen2ForCausalLM':
            base_model=KVQwen2ForCausalLM.from_pretrained(
                base_model_path,**kwargs
            ).to(torch.bfloat16)
            print('Using Qwen2 model, dtype:', base_model.dtype)

        # Load configuration for the Falcon acceleration layer
        configpath=os.path.join(fa_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(fa_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath,
            tree_choice=tree_choice
        )
        
        # Load Falcon acceleration weights
        load_model_path=os.path.join(fa_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(fa_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
    
        model.fa_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None,
            k_mask=2
    ):
        """
        Forward pass through the model.
        
        This implementation enables the semi-autoregressive generation by processing
        the output from the base model through the Falcon acceleration layer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for training
            past_key_values: Cached KV pairs for faster generation
            output_orig: Whether to return original outputs from base model
            position_ids: Position IDs
            init: Whether to initialize for generation
            logits_processor: Optional processor for logits
            k_mask: K-mask value for generation
            
        Returns:
            Tuple containing various outputs depending on parameters
        """
        with torch.inference_mode():
            # Pass input through the base model
            
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids
            )

            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
                
            hidden_states = outputs[0].clone()
            
        if init:
            # Generate the next token based on the base model's prediction
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1:])
                token = token[None, None]

            # Prepare token sequence for the Falcon layer
            llm_tokens = []
            for i in range(k_mask-1, 0, -1):
                llm_tokens.append(input_ids[0][-i][None,None])
            llm_tokens.append(token)
            token_return = torch.cat(llm_tokens)
            
            # Add new token to input_ids and process through Falcon layer
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            ea_logits = self.fa_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, logits_processor, k_mask=k_mask)
            
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token_return
            return ea_logits, hidden_states, token_return
        else:
            if output_orig:
                return outputs, orig, hidden_states


    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,

    ):
        """
        Enhanced accelerated generation method using custom tree-based speculative decoding.
        
        This method accelerates inference by speculatively generating multiple tokens
        at once and verifying them against the base model.
        
        Args:
            input_ids: Input token IDs
            temperature: Sampling temperature (0 for greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter  
            max_new_tokens: Maximum number of new tokens to generate
            max_length: Maximum length of the sequence including input
            tree_choices: Tree structure for speculative decoding
            
        Returns:
            Tensor: Generated token IDs
        """
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
            
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.fa_layer.reset_kv()

        # Setup tree buffers for decoding
        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Initialize tree-based decoding
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        
        new_token = 0

        # Main generation loop
        for idx in range(max_length):
            # Generate candidate tokens
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            
            # Decode tree-based candidates
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
            
            # Evaluate candidate tokens against base model
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            
            # Update generation state
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
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            # Check stopping conditions
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids
            if new_token > max_new_tokens:
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        """
        Streaming version of enhanced accelerated generation that yields results incrementally.
        
        This method yields partial results after each step, useful for streaming generation
        where tokens need to be displayed as they are generated.
        
        Args:
            input_ids: Input token IDs
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_steps: Maximum number of generation steps
            tree_choices: Tree structure for speculative decoding
            
        Yields:
            Tensor: Partial generated sequences after each step
        """
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.fa_layer.reset_kv()

        # Setup tree buffers for decoding
        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Initialize tree-based decoding
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0

        # Main generation loop with streaming
        for idx in range(max_steps):
            # Generate candidate tokens
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            
            # Decode tree-based candidates
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )

            # Evaluate candidate tokens against base model
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            
            # Update generation state
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
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            # Yield current progress
            yield input_ids

            # Check stopping conditions
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        """
        Baseline naive autoregressive generation for comparison.
        
        This method provides a standard autoregressive generation approach
        without the Falcon acceleration techniques, useful for benchmarking.
        
        Args:
            input_ids: Input token IDs
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_steps: Maximum number of generation steps
            tree_choices: Tree structure (unused in naive generation, kept for API consistency)
            
        Yields:
            Tensor: Partial generated sequences after each step
        """
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.fa_layer.reset_kv()

        # Setup tree buffers for consistency with other methods
        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Standard autoregressive generation with the base model
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        # Main generation loop
        for idx in range(max_steps):
            # Get most likely token
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            # Generate next token
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            # Yield current progress
            yield input_ids

            # Check stopping conditions
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
