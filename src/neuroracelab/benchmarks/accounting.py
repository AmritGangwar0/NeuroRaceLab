from neuroracelab.core import RunAccounting


def aggregate(results):
    return RunAccounting(sum(x.environment_evaluations for x in results),sum(x.flops for x in results),sum(x.wall_time_seconds for x in results))
