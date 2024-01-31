def normaliseList(ll, max_value=10.):
    minv = min(ll)
    maxv = max(ll)
    gap = maxv-minv
    if gap == 0:
        gap = 1

    new_ll = [(x-minv)*max_value/gap for x in ll]

    return new_ll