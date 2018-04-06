from multiprocessing import Process
import sys
import classify

if __name__ == "__main__":
    print(' '.join(sys.argv))
    timeout = None
    timeout_pos = -1
    for i in range(len(sys.argv)):
        if sys.argv[i].lower() == "--timeout":
            timeout = int(sys.argv[i+1])
            timeout_pos = i
            break
    if timeout_pos != -1:
        del sys.argv[i:i+2]
    p = Process(target=classify.main, args=())
    p.start()
    try:
        p.join(timeout)
    except Exception:
        raise Exception("Exception while running the process.")
    if p.is_alive():
        print "Timeout Error: time has expired on this test."
        p.terminate()
