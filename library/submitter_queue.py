# -*- coding: utf-8 -*-
import os.path
import os
import tempfile 
import subprocess
import time
import threading
import re
from Queue import Queue
import logging

log = logging.getLogger(__name__)


OUT_RE = re.compile('^Your job (\d+)')

class Job(object):
    def __init__(self, subfile, debug = False):
        self._id = ''
        self.debug = debug
        self.subfile = subfile
        self._is_done = False

    def start(self):
        if self.debug:
            cmdline = ['condor_qsub', self.subfile]
        else:
            cmdline = ['condor_qsub', self.subfile, '-e', '/dev/null', '-o', '/dev/null']
        out = subprocess.check_output(cmdline)
        match = OUT_RE.match(out)
        if match is None:
            raise ValueError("Unexpected output from qsub {}".format(out))
        self._id = int(match.group(1))
        print 'Job {0} started with job id {1}'.format(self.subfile.split('.')[0], self._id)

    def cancel(self):
        if self._is_done:
            return
        print 'Canceling job {0} with id {1}'.format(self.subfile.split('.')[0], self._id)
        self._is_done = True

    def is_done(self):
        if self._is_done:
            return True
        res = subprocess.check_output(['condor_q'])
        match = res.find(str(self._id))
        if match == -1:
            print 'Job {0} with id {1} is done'.format(self.subfile.split('.')[0], self._id)
            self._is_done = True
            return True
        return False


class JobsPool(object):
    SLEEP_TIME = 300
    def __init__(self, max_jobs):
        self.remaining_jobs = Queue()
        for i in range(max_jobs):
            self.remaining_jobs.put(None)
        self.jobs_lock = threading.Lock()
        self.jobs = []
        self._cancel = False
        self._stop = False
        self._check_thread = threading.Thread(target=self._check_jobs)
        self._check_thread.start()

    def submit(self, subfile, debug = False):
        if self._stop:
            raise ValueError("Pool is canceled, no more jobs can be added")
        self.remaining_jobs.get(block=True, timeout=1e100)
        with self.jobs_lock:
            if self._stop:
                self.remaining_jobs.release()
                raise ValueError('Pool is canceled, no more jobs can be added')
            job = Job(subfile, debug = debug)
            job.start()
            self.jobs.append(job)
            
    def _check_jobs(self):
        while True:
            with self.jobs_lock:
                for job in list(self.jobs):
                    if self._cancel:
                        if not job.is_done():
                            try:
                                job.cancel()
                            except BaseException:
                                log.exception('Error while trying to cancel job %d', job.job_id)
                    if self._cancel or job.is_done():
                        self.jobs.remove(job)
                        self.remaining_jobs.put(None, block=False)
                if not self.jobs and self._stop:
                    return
            time.sleep(self.SLEEP_TIME)

    def cancel(self):
        with self.jobs_lock:
            self._cancel = True
            self._stop = True

    def close(self):
        print "close"
        self._stop = True

    def join(self, timeout=1e100):
        self.close()
        self._check_thread.join(timeout)


