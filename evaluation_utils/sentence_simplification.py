"""Utility functions simplification evaluations"""
from __future__ import print_function

import os
import sys
import torchfile
import subprocess
from readability import Readability

"""
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except NameError:
    # raise EnvironmentError("This file only supports python2")
    pass
"""
reload(sys)
sys.setdefaultencoding('utf8')




def _replace_ner(sentence, ner_dict):
    """Replace the Named Entities in a sentence

    Args:
        sentence: str, sentences to be processed
        ner_dict: dictionary of {NER_tag: word} or an empty list

    Returns:
        processed sentence
    """
    if isinstance(ner_dict, (list, tuple)):
        # the map is empty, no NER in the sentence
        return sentence

    def replace_fn(token):
        # for compatability between python2 and 3
        # upper because the NER are upper-based
        if token.upper().encode() in ner_dict.keys():
            # lower case replaced words
            return ner_dict[token.upper().encode()].decode().lower()
        else:
            return token

    return " ".join(map(replace_fn, sentence.split()))


def _deanonymize_file(file, ner_map_file, mode):
    if not os.path.exists(file):
        raise ValueError("file %s does not exist" % file)
    if not os.path.exists(file):
        raise ValueError("NER_Map %s does not exist" % ner_map_file)

    if mode not in ["train", "valid", "test"]:
        raise ValueError(
            "mode must be in `valid` for `test`, saw ", mode)
    
    # read in unprocessed file
    with open(file) as f:
        raw_outputs = f.readlines()
        raw_outputs = [d.strip() for d in raw_outputs]

    # read in NER_Map
    ner_maps = torchfile.load(ner_map_file)
    # for compatability between python2 and 3
    ner_map = ner_maps[mode.encode(encoding="utf-8")]

    # process sentences
    deanonymized_outputs = []
    if not len(raw_outputs) == len(ner_map):
        raise ValueError("raw_outputs and ner_map shape mismatch")
    for raw_output, ner_dict in zip(raw_outputs, ner_map):
        deanonymized_output = _replace_ner(raw_output, ner_dict)
        deanonymized_outputs.append(deanonymized_output)

    deanonymized_file = file + "_deanonymized"
    with open(deanonymized_file, "w") as f:
        f.write("\n".join(deanonymized_outputs))
    
    return deanonymized_file


def run_BLEU(JOSHUA_dir, output_dir, reference_dir, num_references=8):
    joshua_output = subprocess.check_output("""
        export JAVA_HOME=/usr/lib/jvm/java-1.8.0
        export JOSHUA=%s
        export LC_ALL=en_US.UTF-8
        export LANG=en_US.UTF-8
        $JOSHUA/bin/bleu %s %s %d
    """ % (JOSHUA_dir, output_dir, reference_dir, num_references), shell=True)

    score = float(joshua_output.split("BLEU = ")[1])
    return 100 * score


def run_FKGL(output_dir):
    with open(output_dir) as f:
        output = f.readlines()
        output = [d.lower().strip() for d in output]

    output_final = " ".join(output)
    rd = Readability(output_final)
    score = rd.FleschKincaidGradeLevel()
    return score


def run_SARI(JOSHUA_dir, output_dir, reference_dir, source_dir,
             num_references=8):
    if num_references == 8:
        executable = None  # Instruction: PLEASE CHANGE THE DIRECTORIES HERE
    elif num_references == 1:
        executable = None  # Instruction: PLEASE CHANGE THE DIRECTORIES HERE
    else:
        raise ValueError("num_references must be 8 or 1")

    joshua_output = subprocess.check_output("""
       export JAVA_HOME=/usr/lib/jvm/java-1.8.0
       export JOSHUA=%s
       export LC_ALL=en_US.UTF-8
       export LANG=en_US.UTF-8
       %s %s %s %s
    """ % (JOSHUA_dir, executable, output_dir,
           reference_dir, source_dir), shell=True)

    score = float(joshua_output.split("STAR = ")[1])
    return 100 * score


def evaluate(mode,
             gen_file,
             ref_file=None,
             execute_dir=None,
             source_file=None,
             evaluation_task=None,
             deanonymize_file=True):

    # Instruction: PLEASE CHANGE THE DIRECTORIES HERE
    JOSHUA_dir = None
    WikiLarge_NER_MAP_FILE = None
    WikiSmall_NER_MAP_FILE = None
    Newsela_NER_MAP_FILE = None

    if evaluation_task in ["WikiLarge"]:
        ner_map_file = WikiLarge_NER_MAP_FILE
        num_test_references = 8
    elif evaluation_task in ["WikiSmall"]:
        ner_map_file = WikiSmall_NER_MAP_FILE
        num_test_references = 1
    elif evaluation_task in ["Newsela"]:
        ner_map_file = Newsela_NER_MAP_FILE
        num_test_references = 1

    if not execute_dir:
        execute_dir = JOSHUA_dir
    if mode == "val":
        num_references = 1
    else:
        num_references = num_test_references
        if deanonymize_file:
            gen_file = _deanonymize_file(
                file=gen_file, mode="test",
                ner_map_file=ner_map_file)

    bleu = run_BLEU(
        JOSHUA_dir=execute_dir,
        output_dir=gen_file,
        reference_dir=ref_file,
        num_references=num_references)
    fkgl = run_FKGL(
        output_dir=gen_file)
    sari = run_SARI(
        JOSHUA_dir=execute_dir,
        output_dir=gen_file,
        reference_dir=ref_file,
        source_dir=source_file,
        num_references=num_references)

    scores = {"BLEU": bleu,
              "FKGL": fkgl,
              "SARI": sari}

    return scores
