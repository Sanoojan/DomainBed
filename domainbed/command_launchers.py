# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def multi_gpu_launcher_1_2(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = 2
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                expoo="export"
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx+2} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def gpu_launcher_2(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = 1
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                expoo="export"
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx+2} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'multi_gpu_1_2':multi_gpu_launcher_1_2,
    'gpu_2':gpu_launcher_2
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def multi_gpu_launcher_1_2(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = 2
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                expoo="export"
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES=2 {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def gpu_launcher_2(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = 1
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                expoo="export"
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx+2} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def gpu_launcher_1(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = 1
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                expoo="export"
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx+1} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'multi_gpu_1_2':multi_gpu_launcher_1_2,
    'gpu_2':gpu_launcher_2,
    'gpu_1':gpu_launcher_1
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
