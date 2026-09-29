"""Execute one OCI program's tests inside a disposable external sandbox."""
import contextlib
import io
import json
import multiprocessing as mp
import os
import random
import resource
import sys
import tempfile
import time


def child(code, test, timeout, connection):
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    started = time.monotonic()
    phase = 'load'
    try:
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            os.chdir(folder)
            random.seed(42)
            sys.stdin = io.StringIO('')
            namespace = {'__name__': '__candidate__', '__file__': 'candidate.py'}
            with open(os.devnull, 'w') as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(code, 'candidate.py', 'exec'), namespace)
                phase = 'test'
                exec(compile(test, 'oci_test.py', 'exec'), namespace)
        result = dict(status='pass', phase=phase)
    except BaseException as error:
        result = dict(status='fail', phase=phase, exception=type(error).__name__, message=str(error)[:1000])
    result['seconds'] = time.monotonic() - started
    connection.send(result)
    connection.close()


def main():
    assert os.environ.get('OCI_UNIT_TEST_SANDBOX') == '1'
    request = json.load(sys.stdin)
    context = mp.get_context('fork')
    results = []
    for test in request['tests']:
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=child, args=(request['code'], test, request['timeout'], sender))
        started = time.monotonic()
        process.start()
        sender.close()
        process.join(request['timeout'])
        if process.is_alive():
            process.kill()
            process.join()
            result = dict(status='fail', phase='unknown', exception='Timeout', seconds=time.monotonic()-started)
        elif receiver.poll():
            try:
                result = receiver.recv()
            except EOFError:
                result = dict(status='fail', phase='unknown', exception='WorkerExit', exitcode=process.exitcode)
        else:
            result = dict(status='fail', phase='unknown', exception='WorkerExit', exitcode=process.exitcode)
        receiver.close()
        results.append(result)
        if request.get("stop_on_failure", False) and result["status"] != "pass":
            break
    print(json.dumps(dict(tests=results, all_pass=len(results)==len(request['tests']) and all(r['status']=='pass' for r in results)), sort_keys=True))


if __name__ == '__main__':
    main()
