# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks:
# SubTask 1: Slice the Tomato. (Skills Required: GoToObject, PickupObject, SliceObject, PutObject)
# We can execute SubTask 1.

# CODE
def slice_tomato():
    # 0: SubTask 1: Slice the Tomato
    # 1: Go to the Knife.
    GoToObject('Knife')
    # 2: Pick up the Knife.
    PickupObject('Knife')
    # 3: Go to the Tomato.
    GoToObject('Tomato')
    # 4: Slice the Tomato.
    SliceObject('Tomato')
    # 5: Go to the countertop.
    GoToObject('CounterTop')
    # 6: Put the Knife back on the CounterTop.
    PutObject('Knife', 'CounterTop')

# Execute SubTask 1
slice_tomato()

# Task slice the tomato is done