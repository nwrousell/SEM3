# preprocessing (do BBF / CASL do the same thing that Mnih 2015 does?)
# data-augmentation: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L68



# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L297
def process_inputs(x, data_augmentation=False):
    out = x / 255
    
    if data_augmentation:
        raise Exception("data augmentation not implemented :'(")
    
    return out


# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L146

# and what's up with: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L203
def interpolate_weights():
    pass

