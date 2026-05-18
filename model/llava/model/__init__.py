from .language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

try:
    from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
except ImportError:
    LlavaMPTConfig = None
    LlavaMPTForCausalLM = None
