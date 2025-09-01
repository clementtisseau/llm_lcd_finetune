import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Dict, Optional, List

# --- Utility functions and classes from human_eval ---
# These are kept as-is because they provide robust sandboxing and timeout features.

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs): raise IOError
    def readline(self, *args, **kwargs): raise IOError
    def readlines(self, *args, **kwargs): raise IOError
    def readable(self, *args, **kwargs): return False

class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

@contextlib.contextmanager
def chdir(root):
    if root == ".": yield; return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    # Disables various destructive functions to prevent generated code from
    # interfering with the test environment.
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    # ... (the rest of the guard functions from your example)

# --- MBPP-specific execution logic ---

def unsafe_execute(problem: Dict, completion: str, timeout: float, result: List):
    """
    Executes the generated code against the problem's test cases.
    
    This function is intended to be run in a separate process to isolate it.
    """
    with create_tempdir():
        reliability_guard()
        
        # Combine the test setup, generated code, and test assertions
        # into a single executable script.
        test_setup = problem.get("test_setup_code", "")
        test_assertions = "\n".join(problem["test_list"])
        check_program = f"{test_setup}\n{completion}\n{test_assertions}"

        try:
            exec_globals = {}
            with swallow_io(), time_limit(timeout):
                exec(check_program, exec_globals)
            result.append(("passed", None))
        except TimeoutException:
            result.append(("failed", "TimeoutError"))
        except (SyntaxError, IndentationError):
            result.append(("failed", "Syntax Error"))
        except AssertionError:
            result.append(("failed", "Runtime Error")) # Assertion failures are runtime errors
        except NameError:
            result.append(("failed", "Name Error"))
        except Exception:
            result.append(("failed", "Runtime Error"))
        except BaseException:
            result.append(("failed", "Base Exception"))
 
def check_correctness(
    problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None
) -> Dict:
    """
    Evaluates a completion by running its test suite in a sandboxed process.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(problem, completion, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append(("failed", "TimeoutError"))

    status, error_type = result[0]

    return dict(
        task_id=problem["task_id"],
        passed=status == "passed",
        result=status,
        error_type=error_type,
        completion_id=completion_id,
        completion=completion,
    )