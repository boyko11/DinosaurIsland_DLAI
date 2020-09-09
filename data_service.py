import pprint


class DataService:

    def __init__(self):
        pass

    @staticmethod
    def get_data():

        data = open('data/dinos.txt', 'r').read()
        data = data.lower()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

        chars = sorted(chars)
        # print(chars)

        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(char_to_ix)
        # print('-------------')
        # pp.pprint(ix_to_char)
        return data, char_to_ix, ix_to_char