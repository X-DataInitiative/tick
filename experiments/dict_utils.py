import collections


def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = nested_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
