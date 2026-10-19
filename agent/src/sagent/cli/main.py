from importlib.metadata import version
from typing import Optional

import typer
from typer import Typer

from sagent.cli.adapters import adapters_app
from sagent.cli.add import add_command
from sagent.cli.admin.admin import admin_app
from sagent.cli.analyze import analyze_command, check_command
from sagent.cli.auth import auth_app
from sagent.cli.cache import cache_app
from sagent.cli.datasets import datasets_app
from sagent.cli.download import download_command
from sagent.cli.init import init_command
from sagent.cli.jobs import jobs_app, start
from sagent.cli.publish import publish_command
from sagent.cli.remove import remove_command
from sagent.cli.sweeps import sweeps_app
from sagent.cli.sync import sync_command
from sagent.cli.tasks import tasks_app
from sagent.cli.traces import traces_app
from sagent.cli.trials import trials_app
from sagent.cli.view import view_command


def version_callback(value: bool) -> None:
    if value:
        print(version("sagent"))
        raise typer.Exit()


app = Typer(
    no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}
)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    pass


# Primary commands (singular)
app.add_typer(adapters_app, name="adapter", help="Manage adapters.")
app.add_typer(tasks_app, name="task", help="Manage tasks.")
app.add_typer(datasets_app, name="dataset", help="Manage datasets.")
app.add_typer(jobs_app, name="job", help="Manage jobs.")
app.add_typer(trials_app, name="trial", help="Manage trials.")
app.add_typer(cache_app, name="cache", help="Manage SAgent cache.")
app.add_typer(auth_app, name="auth", help="Manage authentication.")

# Plural aliases (hidden, backwards compat)
app.add_typer(adapters_app, name="adapters", help="Manage adapters.", hidden=True)
app.add_typer(tasks_app, name="tasks", help="Manage tasks.", hidden=True)
app.add_typer(datasets_app, name="datasets", help="Manage datasets.", hidden=True)
app.add_typer(jobs_app, name="jobs", help="Manage jobs.", hidden=True)
app.add_typer(trials_app, name="trials", help="Manage trials.", hidden=True)

# Hidden commands (plural only)
app.add_typer(traces_app, name="traces", help="Trace export utilities.", hidden=True)
app.add_typer(
    sweeps_app,
    name="sweeps",
    help="Run successive sweeps to focus on successes.",
    hidden=True,
)
app.add_typer(admin_app, name="admin")
app.command(name="check", help="Check task quality against a rubric.")(check_command)
app.command(name="analyze", help="Analyze trial trajectories.")(analyze_command)

app.command(name="init", help="Initialize a new task or dataset.")(init_command)
app.command(name="run", help="Start a job. Alias for `sagent job start`.")(start)
app.command(name="publish", help="Publish tasks and datasets to the SAgent registry.")(
    publish_command
)
app.command(name="add", help="Add tasks or datasets to a dataset.toml.")(add_command)
app.command(name="download", help="Download a task or dataset.")(download_command)
app.command(name="remove", help="Remove tasks from a dataset.toml.")(remove_command)
app.command(name="sync", help="Update task digests in a dataset manifest.")(
    sync_command
)
app.command(name="view", help="Start web server to browse trajectory files.")(
    view_command
)

if __name__ == "__main__":
    app()
