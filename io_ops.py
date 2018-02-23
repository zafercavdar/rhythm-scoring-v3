divider = ":"
delimiter = "**"


def save(path, dic):
    text = ""
    for key, value in dic.iteritems():
        text += "{}{}{}{}".format(key, divider, value, delimiter)
    text = text[:-len(delimiter)]
    with open(path, "a") as text_file:
        text_file.write("{}\n".format(text))
    print("Saved dict successfully to path {}".format(path))


def read(path):
    dicts = []
    file = open(path, "r")
    for line in file:
        pairs = line.split(delimiter)
        dic = {}
        for pair in pairs:
            key, value = pair.split(divider)
            dic[str(key)] = value
        dicts.append(dic)
    return dicts
