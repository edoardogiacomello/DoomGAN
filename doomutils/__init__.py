import itertools

def vertices_to_segment_list(vertices):
    if vertices[0] != vertices[-1]:
        vertices += [vertices[0]]
    startiter, enditer = itertools.tee(vertices, 2)
    next(enditer, None)  # shift the end iterator by one
    return zip(startiter, enditer)

def build_batches(n, data, fillvalue=None):
    args = [iter(data)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)