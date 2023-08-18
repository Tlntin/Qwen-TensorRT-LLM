import time

import pynvml


def get_memory_info(handle):
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = round(mem_info.total / 1024 / 1024 / 1024, 2)
    used = round(mem_info.used / 1024 / 1024 / 1024, 2)
    free = round(mem_info.used / 1024 / 1024 / 1024, 2)
    return total, used, free


def mem_monitor(q1, q2):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    peak_used = 0
    while q1.empty():
        _, used, _ = get_memory_info(handle)
        peak_used = max(used, peak_used)
        time.sleep(0.1)

    pynvml.nvmlShutdown()
    q2.put(peak_used)
