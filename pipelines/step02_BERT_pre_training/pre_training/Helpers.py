
# deals with training_args "report_to" argument
def training_args_deprecation_fix(training_args):
    if training_args is None:
        training_args = {"output_dir": "tmp_trainer"}
    if "report_to" not in training_args.keys():
        training_args["report_to"] = ["all"]
    return training_args
