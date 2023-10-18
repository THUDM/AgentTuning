# Basic Node to be inherited from.
class Node:
    def __init__(self, name, input_type, output_type):
        self.name = name
        self.input_type = input_type
        self.output_type = output_type

    def run (self, input, log=False):
        raise NotImplementedError

