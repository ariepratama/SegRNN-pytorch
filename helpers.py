import sys
import codecs
from .sentence import Sentence
import torch

class CoNLL09Element(object):
    """
    Representation of a token ni CONLL?
    """

    def __init__(self):
        self.id = None
        self.form = None
        self.nltk_lemma = None
        self.fn_pos = None
        self.nltk_pos = None
        self.sent_num = None
        self.dephead = None
        self.deprel = None
        self.lu = None  # int representation of
        self.lupos = None
        self.frame = None
        self.is_arg = None
        self.argtype = None
        self.role = None
