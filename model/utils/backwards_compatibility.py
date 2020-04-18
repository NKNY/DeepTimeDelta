# Project generally supports Python 2.X as SageMaker originally didn't have Python 3 support.
# Functions used to make the code backwards compatible located in this file

def merge_dicts(x, y, *args):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    for i in args:
        z.update(i)
    return z