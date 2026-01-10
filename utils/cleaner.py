import awkward as ak

def clean_hit_vars(hit_vars):
    reference_key = next(iter(hit_vars))
    mask = ~ak.is_none(hit_vars[reference_key])
    cleaned_hit_vars = {}
    for key, array in hit_vars.items():
        if isinstance(array, ak.Array):
            cleaned_array = array[mask]
            cleaned_array = ak.drop_none(cleaned_array)
            cleaned_hit_vars[key] = ak.to_list(cleaned_array)
        else:
            cleaned_hit_vars[key] = array
    return cleaned_hit_vars
