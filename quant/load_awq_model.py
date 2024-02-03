from tqdm import tqdm
import torch
import gc
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
# from awq.utils.utils import simple_dispatch_model
from transformers import default_data_collator, Trainer, AutoModelForCausalLM, LlamaConfig
# from awq.quantize.quantizer import scale_activations, pseudo_quantize_tensor, set_op_by_name
# from qmodule import WQLinearForTrain


@torch.no_grad()
def real_quantize_model_weight(
    model, w_bit, q_config,
    init_only=False
):
    # from awq.quantize.qmodule import WQLinear
    from awq.quantize.pre_quant import get_blocks, get_named_linears
    assert q_config["zero_point"], "We only support zero_point quantization now."
    
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="real weight quantization..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinearForTrain.from_linear(
                    module, w_bit, q_config['q_group_size'], True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config)
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinearForTrain.from_linear(
                    module, w_bit, q_config['q_group_size'], False, scales, zeros)
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()
                
    torch.cuda.empty_cache()
    gc.collect()


def load_saved_awq_model(model, saved_awq_model_path, model_config, dtype, w_bit, q_config):
    print("Loading pre-computed quantized weights...")
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config=model_config,
    #                                                 torch_dtype=dtype, trust_remote_code=True)
    real_quantize_model_weight(
        model, w_bit=w_bit, q_config=q_config, init_only=False)
    
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
    )
    # # Load checkpoint in the model
    load_checkpoint_in_model(
        model,
        checkpoint=saved_awq_model_path,
        device_map=device_map
    )
    # # Dispatch model
    model = simple_dispatch_model(model, device_map)
    return model