import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75,
                     small_lr_keywords=('offset',), small_lr_ratio=0.1):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if hasattr(model, 'blocks'):
        num_layers = len(model.blocks) + 1
    elif hasattr(model, 'dcnv3') and hasattr(model.dcnv3, 'levels'):
        num_layers = len(model.dcnv3.levels) + 1
    else:
        raise NotImplementedError

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        small_lr = False
        for small_lr_keyword in small_lr_keywords:
            if small_lr_keyword in n:
                g_decay = 'decay_small_lr'
                this_decay = weight_decay
                layer_id = get_layer_id_for_vit(n, num_layers)
                group_name = "layer_%d_%s" % (layer_id, g_decay)
                small_lr = True
                this_scale = layer_scales[layer_id] * 0.1

        if not small_lr:
            # no decay: all 1D parameters and model specific ones
            if p.ndim == 1 or n in no_weight_decay_list:
                g_decay = "no_decay"
                this_decay = 0.
            else:
                g_decay = "decay"
                this_decay = weight_decay

            layer_id = get_layer_id_for_vit(n, num_layers)
            group_name = "layer_%d_%s" % (layer_id, g_decay)
            this_scale = layer_scales[layer_id]

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed') or name.startswith('dcnv3.patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('dcnv3.levels'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers