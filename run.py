import importlib
import logging

import click

logger = logging.getLogger(__name__)

try:
    import colored_traceback

    colored_traceback.add_hook()
except ModuleNotFoundError:
    pass


@click.group()
def run():
    pass


@run.command("task")
@click.argument("dag")
@click.argument("task")
def task(dag, task):
    try:
        mod = importlib.import_module(f"dags.{dag}")
    except ModuleNotFoundError as ex:
        if "dag" not in ex.args[0]:
            raise
        click.secho(f"ERROR: Dag not found '{dag}'")
        logger.exception(ex)
        raise SystemExit(1)

    mod.run_dag(task)


@run.command("list-tasks")
@click.argument("dag")
def list_tasks(dag):

    dags = get_dags()

    if dag not in dags:
        click.secho(f"ERROR: DAG '{dag}' not found in dags.yaml")

    for task in dags[dag]:
        print(task)


def get_dags():
    from pathlib import Path

    import yaml

    dags_file = Path(__file__).parent / "dags.yaml"

    with dags_file.open("r") as fo:
        dags = yaml.safe_load(fo)
    return dags


if __name__ == "__main__":
    from dags import logging

    logging.init()
    run()
