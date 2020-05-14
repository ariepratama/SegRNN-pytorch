from copy import deepcopy
from sentence import Sentence
import sys
import codecs

# Label settings
UNK = "UNK"
EMPTY_LABEL = "_"
EMPTY_FE = "O"

# BIOS scheme settings
BEGINNING = 0
INSIDE = 1
OUTSIDE = 2
SINGULAR = 3

BIO_INDEX_DICT = {
    "B": BEGINNING,
    "I": INSIDE,
    EMPTY_FE: OUTSIDE,
    "S": SINGULAR
}

INDEX_BIO_DICT = {index: tag for tag, index in BIO_INDEX_DICT.items()}


def extract_spans(indices):
    """
    Handles discontinuous, repeated FEs.
    In PropBank, the equivalent is reference-style arguments, like R-A0
    :param indices: list of array indices with the same FE
    :return: list of tuples containing argument spans
    """
    indices.sort()
    spans = [(indices[0], indices[0])]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            o = spans.pop()
            spans.append((o[0], indices[i]))
        else:
            spans.append((indices[i], indices[i]))
    return spans


class Frame(object):

    def __init__(self, id):
        self.id = id

    def get_str(self, framedict):
        return framedict.getstr(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


class LexicalUnit(object):

    def __init__(self, id, posid):
        self.id = id
        self.posid = posid

    def get_str(self, ludict, luposdict):
        return ludict.getstr(self.id) + "." + luposdict.getstr(self.posid)

    def __hash__(self):
        return hash((self.id, self.posid))

    def __eq__(self, other):
        return self.id == other.id and self.posid == other.posid

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


class FspDict:
    def __init__(self):
        self._strtoint = {}
        self._inttostr = {}
        self._locked = False
        self._posttrainlocked = False
        self._singletons = set([])
        self._unseens = set([])
        self._unks = set([])  # these are vocabulary items which were not in train, so we don't know parameters for.

    def addstr(self, itemstr):
        if self._posttrainlocked and itemstr not in self._strtoint:
            self._unks.add(itemstr)
        if self._locked:
            if itemstr in self._strtoint:
                return self.getid(itemstr)
            self._unseens.add(itemstr)  # rpt handles the repeated, discontinuous FEs(R-FEs, under Propbank style)
            return self._strtoint[UNK]
        else:
            if itemstr not in self._strtoint:
                idforstr = len(self._strtoint)
                self._strtoint[itemstr] = idforstr
                self._inttostr[idforstr] = itemstr
                self._singletons.add(idforstr)
                return idforstr
            else:
                idforstr = self.getid(itemstr)
                if self.is_singleton(idforstr):
                    self._singletons.remove(idforstr)
                return idforstr

    def remove_extras(self, extras):
        for e in extras:
            eid = self._strtoint[e]
            del self._strtoint[e]
            del self._inttostr[eid]
            if eid in self._singletons:
                self._singletons.remove(eid)
                # no need to remove from unks because the repeat extras were never added to it

    def getid(self, itemstr):
        if itemstr in self._strtoint:
            return self._strtoint[itemstr]
        elif self._locked:
            return self._strtoint[UNK]
        else:
            raise Exception("not in dictionary, but can be added", id)

    def getstr(self, itemid):
        if itemid in self._inttostr:
            return self._inttostr[itemid]
        else:
            raise Exception("not in dictionary", itemid)

    def printdict(self):
        print(sorted(self._strtoint.keys()))

    def size(self):
        if not self._locked:
            raise Exception("dictionary still modifiable")
        return len(self._strtoint)

    def lock(self):
        if self._locked:
            raise Exception("dictionary already locked!")
        self.addstr(UNK)
        self._locked = True
        self._unseens = set([])

    def post_train_lock(self):
        if self._posttrainlocked:
            raise Exception("dictionary already post-train-locked!")
        self._posttrainlocked = True
        self._unks = set([])

    def islocked(self):
        return self._locked

    def is_singleton(self, idforstr):
        if idforstr in self._singletons:
            return True
        return False

    def num_unks(self):
        """
        :return: Number of unknowns attempted to be added to dictionary
        """
        # print self._unks
        return len(self._unseens), len(self._unks)

    def getidset(self):
        unkset = {self._strtoint[UNK]}
        fullset = set(self._inttostr.keys())
        return list(fullset - unkset)


VOCDICT = FspDict()
LEMDICT = FspDict()
POSDICT = FspDict()
FRAMEDICT = FspDict()
LUDICT = FspDict()
LUPOSDICT = FspDict()
FEDICT = FspDict()
DEPRELDICT = FspDict()
CLABELDICT = FspDict()


class FrameSemParse(object):
    """frame-semantic parse structure, contain a target LU, frame it evokes, arguments and corresponding frame elements all in the context of a sentence"""

    def __init__(self, sentence):
        self.tokens = sentence.tokens
        self.postags = sentence.postags
        self.lemmas = sentence.lemmas
        # TODO(Swabha): add some inheritance, etc.
        self.sentence = sentence
        self.targetframedict = {}  # map of target position and frame-id
        self.frame = None
        self.lu = None
        # self.fes = {} # map of FE position to a map between FE-type(BIOS) and the label
        self.numargs = 0
        self.modifiable = True  # to differentiate between gold and predicted

    def add_target(self, targetpos, luid, lupos, frameid):
        if not self.modifiable:
            raise Exception(
                'attempt to add target and frame to unmodifiable example')
        if targetpos in self.targetframedict:
            raise Exception('target already in parse', targetpos, frameid)

        if self.frame is not None and frameid != self.frame.id:
            raise Exception(
                "two different frames in a single parse, illegal", frameid, self.frame.id)
        self.frame = Frame(frameid)

        if self.lu is not None and luid != self.lu.id:
            raise Exception("different LU ID than original", self.lu.id, luid)
        self.lu = LexicalUnit(luid, lupos)
        self.targetframedict[targetpos] = (self.lu, self.frame)

    def get_only_targets(self):
        if self.modifiable:
            raise Exception('still modifying the example, incomplete...')
        tdict = {}
        for luidx in self.targetframedict:
            tdict[luidx] = self.targetframedict[luidx][0]
        return tdict


class CoNLL09Example(FrameSemParse):
    """a single example in CoNLL 09 format which corresponds to a single frame-semantic parse structure"""

    def __init__(self, sentence, elements):
        FrameSemParse.__init__(self, sentence)
        # not in parent class
        self._elements = elements
        self.sent_num = elements[0].sent_num

        notfes = []
        self.invertedfes = {}
        for e in elements:
            if e.is_pred:
                self.add_target((e.id - 1), e.lu, e.lupos, e.frame)

            if e.role not in self.invertedfes:
                self.invertedfes[e.role] = []
            if e.argtype == SINGULAR:
                self.invertedfes[e.role].append((e.id - 1, e.id - 1))
                self.numargs += 1
            elif e.argtype == BEGINNING:
                self.invertedfes[e.role].append((e.id - 1, None))
                self.numargs += 1
            elif e.argtype == INSIDE:
                argspan = self.invertedfes[e.role].pop()
                self.invertedfes[e.role].append((argspan[0], e.id - 1))
            else:
                notfes.append(e.id - 1)

        if FEDICT.getid(EMPTY_FE) in self.invertedfes:
            self.invertedfes[FEDICT.getid(EMPTY_FE)] = extract_spans(notfes)

        self.modifiable = False  # true cz generally gold.

    def _get_inverted_femap(self):
        tmp = {}
        for e in self._elements:
            if e.role not in tmp:
                tmp[e.role] = []
            tmp[e.role].append(e.id - 1)

        inverted = {}
        for felabel in tmp:
            argindices = sorted(tmp[felabel])
            argranges = extract_spans(argindices)
            inverted[felabel] = argranges

        return inverted

    def get_str(self, predictedfes=None):
        mystr = ""
        if predictedfes is None:
            for e in self._elements:
                mystr += e.get_str()
        else:
            rolelabels = [EMPTY_FE for _ in self._elements]
            for feid in predictedfes:
                felabel = FEDICT.getstr(feid)
                if felabel == EMPTY_FE:
                    continue
                for argspan in predictedfes[feid]:
                    if argspan[0] == argspan[1]:
                        rolelabels[argspan[0]] = INDEX_BIO_DICT[SINGULAR] + "-" + felabel
                    else:
                        rolelabels[argspan[0]] = INDEX_BIO_DICT[BEGINNING] + "-" + felabel
                    for position in range(argspan[0] + 1, argspan[1] + 1):
                        rolelabels[position] = INDEX_BIO_DICT[INSIDE] + "-" + felabel

            for e, role in zip(self._elements, rolelabels):
                mystr += e.get_str(rolelabel=role)

        return mystr

    def get_predicted_frame_conll(self, predicted_frame):
        """
        Get new CoNLL string, after substituting predicted frame.
        """
        new_conll_str = ""
        for e in range(len(self._elements)):
            field = deepcopy(self._elements[e])
            if (field.id - 1) in predicted_frame:
                field.is_pred = True
                field.lu = predicted_frame[field.id - 1][0].id
                field.lupos = predicted_frame[field.id - 1][0].posid
                field.frame = predicted_frame[field.id - 1][1].id
            else:
                field.is_pred = False
                field.lu = LUDICT.getid(EMPTY_LABEL)
                field.lupos = LUPOSDICT.getid(EMPTY_LABEL)
                field.frame = FRAMEDICT.getid(EMPTY_LABEL)
            new_conll_str += field.get_str()
        return new_conll_str

    def get_predicted_target_conll(self, predicted_target, predicted_lu):
        """
        Get new CoNLL string, after substituting predicted target.
        """
        new_conll_str = ""
        for e in range(len(self._elements)):
            field = deepcopy(self._elements[e])
            if (field.id - 1) == predicted_target:
                field.is_pred = True
                field.lu = predicted_lu.id
                field.lupos = predicted_lu.posid
            else:
                field.is_pred = False
                field.lu = LUDICT.getid(EMPTY_LABEL)
                field.lupos = LUPOSDICT.getid(EMPTY_LABEL)
            field.frame = FRAMEDICT.getid(EMPTY_LABEL)
            new_conll_str += field.get_str(no_args=True)
        return new_conll_str

    def print_internal(self, logger):
        self.print_internal_sent(logger)
        self.print_internal_frame(logger)
        self.print_internal_args(logger)

    def print_internal_sent(self, logger):
        logger.write("tokens and depparse:\n")
        for x in range(len(self.tokens)):
            logger.write(VOCDICT.getstr(self.tokens[x]) + " ")
        logger.write("\n")

    def print_internal_frame(self, logger):
        logger.write("LU and frame: ")
        for tfpos in self.targetframedict:
            t, f = self.targetframedict[tfpos]
            logger.write(VOCDICT.getstr(self.tokens[tfpos]) + ":" + \
                         LUDICT.getstr(t.id) + "." + LUPOSDICT.getstr(t.posid) + \
                         FRAMEDICT.getstr(f.id) + "\n")

    def print_external_frame(self, predtf, logger):
        logger.write("LU and frame: ")
        for tfpos in predtf:
            t, f = predtf[tfpos]
            logger.write(VOCDICT.getstr(self.tokens[tfpos]) + ":" + \
                         LUDICT.getstr(t.id) + "." + LUPOSDICT.getstr(t.posid) + \
                         FRAMEDICT.getstr(f.id) + "\n")

    def print_internal_args(self, logger):
        logger.write("frame:" + FRAMEDICT.getstr(self.frame.id).upper() + "\n")
        for fepos in self.invertedfes:
            if fepos == FEDICT.getid(EMPTY_FE):
                continue
            for span in self.invertedfes[fepos]:
                logger.write(FEDICT.getstr(fepos) + "\t")
                for s in range(span[0], span[1] + 1):
                    logger.write(VOCDICT.getstr(self.tokens[s]) + " ")
                logger.write("\n")
        logger.write("\n")

    def print_external_parse(self, parse, logger):
        for fepos in parse:
            if fepos == FEDICT.getid(EMPTY_FE):
                continue
            for span in parse[fepos]:
                logger.write(FEDICT.getstr(fepos) + "\t")
                for s in range(span[0], span[1] + 1):
                    logger.write(VOCDICT.getstr(self.tokens[s]) + " ")
                logger.write("\n")
        logger.write("\n")


class CoNLL09Element(object):
    """
    All the elements in a single line of a CoNLL 2009-like file.
    """

    def __init__(self, conll_line, read_depsyn=None):
        ele = conll_line.split("\t")
        lufields = ['_', '_']
        self.id = int(ele[0])
        self.form = VOCDICT.addstr(ele[1].lower())
        self.nltk_lemma = LEMDICT.addstr(ele[3])
        self.fn_pos = ele[4]  # Not a gold POS tag, provided by taggers used in FrameNet, ignore.
        self.nltk_pos = POSDICT.addstr(ele[5])
        self.sent_num = int(ele[6])

        self.dephead = EMPTY_LABEL
        self.deprel = EMPTY_LABEL
        if read_depsyn:
            self.dephead = int(ele[9])
            self.deprel = DEPRELDICT.addstr(ele[11])

        self.is_pred = (ele[12] != EMPTY_LABEL)
        if self.is_pred:
            lufields = ele[12].split(".")
        self.lu = LUDICT.addstr(lufields[0])
        self.lupos = LUPOSDICT.addstr(lufields[1])
        self.frame = FRAMEDICT.addstr(ele[13])

        # BIOS scheme
        self.is_arg = (ele[14] != EMPTY_FE)
        self.argtype = BIO_INDEX_DICT[ele[14][0]]
        if self.is_arg:
            self.role = FEDICT.addstr(ele[14][2:])
        else:
            self.role = FEDICT.addstr(ele[14])

    def get_str(self, rolelabel=None, no_args=False):
        idstr = str(self.id)
        form = VOCDICT.getstr(self.form)
        predicted_lemma = LEMDICT.getstr(self.nltk_lemma)
        nltkpos = POSDICT.getstr(self.nltk_pos)

        dephead = "_"
        deprel = "_"
        if self.dephead != EMPTY_LABEL:
            dephead = str(self.dephead)
            deprel = DEPRELDICT.getstr(self.deprel)

        if self.is_pred:
            lu = LUDICT.getstr(self.lu) + "." + LUPOSDICT.getstr(self.lupos)
        else:
            lu = LUDICT.getstr(self.lu)
        frame = FRAMEDICT.getstr(self.frame)

        if rolelabel is None:
            if self.is_arg:
                rolelabel = INDEX_BIO_DICT[self.argtype] + "-" + FEDICT.getstr(self.role)
            else:
                rolelabel = INDEX_BIO_DICT[self.argtype]

        if no_args:  # For Target ID / Frame ID predictions
            rolelabel = "O"

        # if DEBUG_MODE:
        #     return idstr + form + lu + frame + rolelabel
        # else:
        #     # ID    FORM    LEMMA   PLEMMA  POS PPOS    SENT#   PFEAT   HEAD    PHEAD   DEPREL  PDEPREL LU  FRAME ROLE
        #     # 0     1       2       3       4   5       6       7       8       9       10      11      12  13    14
        #     return "{}\t{}\t_\t{}\t{}\t{}\t{}\t_\t_\t{}\t_\t{}\t{}\t{}\t{}\n".format(
        #         self.id, form, predicted_lemma, self.fn_pos, nltkpos, self.sent_num, dephead, deprel, lu, frame, rolelabel).encode('utf-8')


def read_conll(conll_file, syn_type=None):
    sys.stderr.write("\nReading {} ...\n".format(conll_file))

    read_depsyn = read_constits = False

    examples = []
    elements = []
    missingargs = 0.0
    totalexamples = 0.0

    next_ex = 0
    with codecs.open(conll_file, "r", "utf-8") as cf:
        snum = -1
        for l in cf:
            l = l.strip()
            if l == "":
                if elements[0].sent_num != snum:
                    sentence = Sentence(syn_type, elements=elements)
                    next_ex += 1
                    snum = elements[0].sent_num
                e = CoNLL09Example(sentence, elements)
                examples.append(e)

                if e.numargs == 0:
                    missingargs += 1

                totalexamples += 1

                elements = []
                continue
            elements.append(CoNLL09Element(l, read_depsyn))
        cf.close()
    sys.stderr.write("# examples in %s : %d in %d sents\n" %(conll_file, len(examples), next_ex))
    sys.stderr.write("# examples with missing arguments : %d\n" %missingargs)
    return examples, missingargs, totalexamples
