import numpy as np
from random import random
import uuid


class Commit:
    def __init__(self, commit_id):
        self.commit_id = commit_id


class FaultyCommit:
    def __init__(self, uid, time_to_fix):
        self.id = uid
        self.time_to_fix = time_to_fix
        # self.devs_needed = devs_needed
        # self.qa_needed = qa_needed

    def get_time_to_fix(self):
        return self.time_to_fix

    # def get_resources_spent(self):
    #     return self.time_to_fix * (self.devs_needed + self.qa_needed)


class CommitGenerator:
    def __init__(self, faulty_percentage, time_to_fix_min):
        self.faulty_percentage = faulty_percentage
        self.time_to_fix_min = time_to_fix_min
        # self.standard_deviation = time_to_fix_avg / 2
        # assumption that std is 1/2 of its mean, which means 68% probability to have time to fix between  time_to_fix_avg/2 and time_to_fix_avg*3/2

    def get(self):
        uid = uuid.uuid4()
        if random() > self.faulty_percentage:
            # t = np.random.normal(self.time_to_fix_mean, self.standard_deviation)
            t = self._time_to_fix_gen()
            return FaultyCommit(uid, t)
        return Commit(uid)

    # use a simpler generator func rather than standard deviation since time to fix is unlikely to go below avg.
    def _time_to_fix_gen(self):
        rand = random()
        if rand > 0.9:  # 10% chance takes more than 3 times than min
            return self.time_to_fix_min * 3
        elif rand > 0.8:  # 20 % chance takes more than 2 times than min
            return self.time_to_fix_min * 2
        return self.time_to_fix_min


class SerialPipeline:
    def __init__(self, test_time, deploy_time, idle_time_avg, merge_time, commits):
        self.test_time = test_time
        self.deploy_time = deploy_time
        self.commits = commits
        self.merge_time = merge_time
        self.idle_time = np.random.normal(idle_time_avg,
                                          idle_time_avg / 2)  # time where devs are not available. however still in queue
        self.total_time = 0
        self.total_resource_time = 0
        self.avg_time_per_commit = None
        self.avg_resource_time_per_commit = None

    def name(self):
        return "SerialPipeline"

    def generate_report(self):
        for commit in self.commits:
            if isinstance(commit, FaultyCommit):
                self.total_time += (
                        commit.get_time_to_fix() + self.test_time + self.deploy_time + self.idle_time + self.merge_time)  # if faulty commit, total time it takes is the fix + test + deploy
                self.total_resource_time += (
                        commit.get_time_to_fix() * 2 + self.deploy_time + self.merge_time)  # multiply the time to fix with QA time since they will be involved as well
            else:
                self.total_time += (self.test_time + self.deploy_time + self.idle_time + self.merge_time)
                self.total_resource_time += self.deploy_time + self.merge_time

        self.avg_resource_time_per_commit = self.total_resource_time / len(self.commits)
        self.avg_time_per_commit = self.total_time / len(self.commits)


class ReleaseTrainPipelineTestPerMerge:
    def __init__(self, test_time, deploy_time, idle_time_avg, merge_time, commits, train_size,
                 merge_back_time_per_commit):
        self.test_time = test_time
        self.deploy_time = deploy_time
        self.commits = commits
        self.merge_time = merge_time
        self.train_size = train_size
        self.idle_time = np.random.normal(idle_time_avg,
                                          idle_time_avg / 2)  # time where devs are not available. however still in queue
        self.total_time = 0
        self.total_resource_time = 0
        self.merge_back_time_per_commit = merge_back_time_per_commit
        self.avg_time_per_commit = None
        self.avg_resource_time_per_commit = None

    def name(self):
        return "ReleaseTrainPipelineTestPerMerge"

    def generate_report(self):
        for idx, commit in enumerate(self.commits):
            if isinstance(commit, FaultyCommit):
                self.total_time += (
                        commit.get_time_to_fix() + self.test_time + self.idle_time + self.merge_time)  # if faulty commit, total time it takes is the fix + test + deploy
                self.total_resource_time += (
                        commit.get_time_to_fix() * 2 + self.merge_time)  # multiply the time to fix with QA time since they will be involved as well
            else:
                self.total_time += (self.test_time + self.idle_time + self.merge_time)
                self.total_resource_time += self.merge_time

            # every train deployment, need to do the merge back and deploy
            if idx % self.train_size == 0:
                self.total_resource_time += self.merge_back_time_per_commit * self.train_size
                self.total_resource_time += self.deploy_time

                self.total_time += self.merge_back_time_per_commit * self.train_size
                self.total_time += self.deploy_time

        self.avg_resource_time_per_commit = self.total_resource_time / len(self.commits)
        self.avg_time_per_commit = self.total_time / len(self.commits)


class ReleaseTrainPipelineTestOnce:
    def __init__(self, test_time, deploy_time, idle_time_avg, merge_time, commits, train_size,
                 merge_back_time_per_commit, trouble_shooting_complexity_coef):
        self.test_time = test_time
        self.deploy_time = deploy_time
        self.commits = commits
        self.merge_time = merge_time
        self.train_size = train_size
        self.trouble_shooting_coef = trouble_shooting_complexity_coef
        self.idle_time = np.random.normal(idle_time_avg,
                                          idle_time_avg / 2)  # time where devs are not available. however still in queue
        self.total_time = 0
        self.total_resource_time = 0
        self.merge_back_time_per_commit = merge_back_time_per_commit
        self.avg_time_per_commit = None
        self.avg_resource_time_per_commit = None

    def name(self):
        return "ReleaseTrainPipelineTestOnce"

    def generate_report(self):
        has_error = False
        time_to_fix = 0
        for idx, commit in enumerate(self.commits):
            if isinstance(commit, FaultyCommit):
                has_error = True
                time_to_fix = max(commit.get_time_to_fix(), time_to_fix)
            if idx % self.train_size == 0:
                if has_error:
                    self.total_time += (
                        # assumption that bulk trouble shooting is more complex than individual commit fixes, that it would be self.trainsize/2 more complex
                            time_to_fix * (
                            self.train_size * self.trouble_shooting_coef) + self.test_time + (
                                    self.idle_time + self.merge_time) * self.train_size)  # if faulty commit, total time it takes is the fix + test + deploy
                    self.total_resource_time += (
                        # assuming all the devs involved wit the train needs to help out in solving the problem.
                            time_to_fix * (
                            self.train_size + 1) * self.trouble_shooting_coef + self.merge_time * self.train_size)  # multiply the time to fix with QA + other devs since they will be involved as well
                else:
                    self.total_time += self.test_time + (self.idle_time + self.merge_time) * self.train_size
                    self.total_resource_time += self.merge_time * self.train_size

                self.total_resource_time += self.merge_back_time_per_commit * self.train_size
                self.total_resource_time += self.deploy_time

                self.total_time += self.merge_back_time_per_commit * self.train_size
                self.total_time += self.deploy_time
                has_error = False

        self.avg_resource_time_per_commit = self.total_resource_time / len(self.commits)
        self.avg_time_per_commit = self.total_time / len(self.commits)


class QueuePipeline:
    def __init__(self, test_time, deploy_time, commits, queue_size, merge_time):
        self.test_time = test_time
        self.deploy_time = deploy_time
        self.commits = commits
        self.queue_size = queue_size
        self.merge_time = merge_time
        self.total_time = 0
        self.total_resource_time = 0
        self.avg_time_per_commit = None
        self.avg_resource_time_per_commit = None
        # for the queue pipeline, it will parallelize the build in such a way that it goes as far as the first failure, then rejects the failure and everything after, and takes the successful one

    def name(self):
        return "QueuePipeline"

    def generate_report(self):
        # divide up the commits into queues
        self.queue = np.reshape(self.commits, (-1, self.queue_size))
        total_processed = 0
        total_bad = 0
        for items in self.queue:
            # scan the item from the first to last, stop when the queue has a bad commit
            bad_idx = None
            for idx, commit in enumerate(items):
                if isinstance(commit, FaultyCommit):
                    # if one bad commit, remove from the queue, let the others through
                    bad_idx = idx
                    break
            good_commits = items
            if bad_idx is not None:
                total_bad += 1
                total_processed += 1
                good_commits = good_commits[:bad_idx]
            total_processed += len(good_commits)
            # bad item takes ....
            # bad item actually doesn't take any total time since it's boot off the queue immediately, queue is freed up for other people
            if bad_idx is not None:
                self.total_resource_time += (
                        items[
                            bad_idx].get_time_to_fix() * 2 + self.merge_time)  # time it takes is just the dev and qa fix the commit and place it back to queue.

            # good item takes...
            self.total_time += self.test_time + self.deploy_time  # just need to run test once and deploy once automatically
            self.total_resource_time += self.merge_time * len(items)  # total resource time is just spent to keep dev branch up to date with master


        print("total bad", total_bad)
        self.avg_resource_time_per_commit = self.total_resource_time / total_processed
        self.avg_time_per_commit = self.total_time / total_processed


def main():
    commits = []
    g = CommitGenerator(0.6, 40)
    for _ in range(10000):
        commits.append(g.get())
    ps = SerialPipeline(20, 10, 10, 5, commits)
    ps.generate_report()
    # prt1 = ReleaseTrainPipelineTestPerMerge(20, 10, commits, 5, 10)
    prt1 = ReleaseTrainPipelineTestPerMerge(20, 10, 10, 5, commits, 5, 5)
    prt1.generate_report()
    prt2 = ReleaseTrainPipelineTestOnce(20, 10, 10, 5, commits, 5, 5, 0.5)
    prt2.generate_report()
    qp = QueuePipeline(20,10,commits,5,5)
    qp.generate_report()
    pipeline_reports = [ps, prt1, prt2,qp]
    for p in pipeline_reports:
        print(p.name())
        print(f"total time  {p.total_time}")
        print(f"total resource time  {p.total_resource_time}")
        print(f"avg time per commit  {p.avg_time_per_commit}")
        print(f"avg resource time per commit  {p.avg_resource_time_per_commit}")
        print("------------")


if __name__ == '__main__':
    main()
