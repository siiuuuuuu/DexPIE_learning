from termcolor import cprint

def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    # Store parameter counts per model part.
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    # Iterate over all parameters.
    for name, param in model.named_parameters():
        # The part name is the prefix before the first dot.
        part_name = name.split('.')[0]
        # Add the part if it is not present.
        if part_name not in params_dict:
            params_dict[part_name] = 0
        # Accumulate this parameter count into the part.
        params_dict[part_name] += param.numel()

    # Print total and per-part parameter counts.
    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        # print num (in M) and percentage
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')
