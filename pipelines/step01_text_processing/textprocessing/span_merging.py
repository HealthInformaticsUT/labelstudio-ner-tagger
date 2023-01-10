def mergeSpans(spansA, spansB):
    """
    Merges 2 ordered span lists such that the order is kept. In case a span from the second list (spansB) overlaps an
    existing span from the first list (spansA), it will be skipped.
    :param spansA: List of spans (dicts) that spansB will be merged into. Must have start (int) and end (int) parameters
    :param spansB: List of spans (dicts) that will be merged into spansA. Must have start (int) and end (int) parameters
    :return: Merged spans list without overlapping spans.
    """
    # if one span is empty, then just skip merge
    if len(spansA) == 0:
        return spansB
    elif len(spansB) == 0:
        return spansA

    # Add B if it is between A0 and A1 (consecutive spans)
    # Also adding spans that bound B to cover the edge cases
    spansA.insert(0, {"end": -1})
    spansA.append({"start": spansB[-1]["end"] + 1})

    i = 0
    for B in spansB:
        while True:
            a0 = spansA[i]
            a1 = spansA[i + 1]
            # checking if B is between A0 and A1
            if a0["end"] <= B["start"] and B["end"] <= a1["start"]:
                i += 1
                spansA.insert(i, B)
                break
            # if A is left behind, then take the next ones
            elif B["end"] > a1["start"]:
                i += 1
            else:
                break

    return spansA[1:-1]