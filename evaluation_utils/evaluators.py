from evaluation_utils import sentence_simplification
SUPPORTED_TASKS = ["WikiLarge", "WikiSmall", "Newsela"]


def evaluate(mode,
             gen_file,
             ref_file=None,
             execute_dir=None,
             source_file=None,
             evaluation_task=None,
             deanonymize_file=True):
    """
    Evaluate the model on validation set

    Args:
        gen_file: model outputs
        ref_file: reference file
        execute_dir: directory to `ducrush` perl evaluation folder
                     or directory to `JOSHUA` program directory
        source_file: directory to WikiLarge evaluation source
        evaluation_task: task to run evaluation
    """
    if mode not in ["val", "test"]:
        raise ValueError("Unsupported mode ", mode)

    if evaluation_task not in SUPPORTED_TASKS:
        raise ValueError("%s is not supported" % evaluation_task)

    scores = sentence_simplification.evaluate(
        mode=mode,
        gen_file=gen_file,
        ref_file=ref_file,
        execute_dir=execute_dir,
        source_file=source_file,
        evaluation_task=evaluation_task,
        deanonymize_file=deanonymize_file)


    return scores
