import numpy as np
from random import random
import uuid


class Commit:
    # bug chance represent how likely the bug is not caught when it is going to deploy
    def __init__(self, commit_id, bugs, hidden_bugs, fix_time):
        self.commit_id = commit_id
        self.number_of_bugs = bugs
        self.hidden_bugs = hidden_bugs
        self.fix_time = fix_time

    def has_bug(self):
        return self.number_of_bugs > 1

    def fix(self):
        if self.number_of_bugs > 1:
            self.number_of_bugs -= 1
            return self.fix_time
        if self.hidden_bugs > 1:
            self.hidden_bugs -= 1
        return self.fix_time

    def has_hidden_bug(self):
        return self.hidden_bugs > 1


class BuildHistory:
    STARTED = "STARTED"
    UNIT_TESTED = "UNIT_TESTED"
    INTEGRATION_TESTED = "INTEGRATION_TESTED"
    SYNCED_BRANCHES = "SYNCED_BRANCHES"
    DEPLOYED = "DEPLOYED"
    IDLED = "IDLED"
    FIXED = "FIXED"

    def __init__(self, action, duration, resource=1):
        self.action = action
        self.duration = duration
        self.resource = resource


class Build:
    def __init__(self, commit, unit_test_duration, integration_test_duration, unit_test_flakiness,
                 integration_test_flakiness, deploy_duration):
        self.commit = commit
        self.unit_test_flakiness = unit_test_flakiness
        self.integration_test_flakiness = integration_test_flakiness
        self.unit_test_duration = unit_test_duration
        self.integration_test_duration = integration_test_duration
        self.deploy_duration = deploy_duration
        self.history = [BuildHistory(BuildHistory.STARTED, 0)]
        self.completed = False
        self.unit_test_passed = False
        self.integration_test_passed = False

    def unit_test(self):
        self.history.append(BuildHistory(BuildHistory.UNIT_TESTED, self.unit_test_duration))
        if random() < self.unit_test_flakiness:
            return False
        if self.commit.has_bug():
            return False
        self.unit_test_passed = True
        return True

    def integration_test(self):
        self.history.append(BuildHistory(BuildHistory.INTEGRATION_TESTED, self.integration_test_duration))
        if random() < self.integration_test_flakiness:
            return False
        if self.commit.has_hidden_bug():
            return False
        self.integration_test_passed = True
        return True

    def sync_branches(self, duration):
        self.history.append(BuildHistory(BuildHistory.SYNCED_BRANCHES, duration))

    def deploy(self):
        if self.unit_test_passed and self.integration_test_passed:
            self.completed = True
            return self.history.append(BuildHistory(BuildHistory.DEPLOYED, self.deploy_duration))
        raise Exception("cannot deploy if unit test and integration are not passed")

    def idle(self, duration):
        self.history.append(BuildHistory(BuildHistory.IDLED, duration))

    def fix(self, resources=1):
        fix_duration = self.commit.fix()
        self.history.append(BuildHistory(BuildHistory.FIXED, fix_duration, resources))

    def get_history(self):
        return self.history


class CommitGenerator:
    def __init__(self, bug_chance, test_coverage, avg_lines_of_code, bug_chance_per_line_of_code,
                 fix_time_per_lines_of_code):
        self.bug_chance = bug_chance
        self.test_coverage = test_coverage
        self.avg_lines_of_code = avg_lines_of_code
        self.bug_chance_per_line_of_code = bug_chance_per_line_of_code
        self.fix_time_per_lines_of_code = fix_time_per_lines_of_code

    def get(self):
        lines_of_code = np.random.normal(self.avg_lines_of_code, self.avg_lines_of_code / 2)
        number_of_bugs = lines_of_code * self.bug_chance_per_line_of_code
        hidden_bug = number_of_bugs * (1 - self.test_coverage)
        fix_time = max(self.fix_time_per_lines_of_code * lines_of_code, 1)  # takes at least 1 min to fix
        if random() > self.bug_chance:
            number_of_bugs = hidden_bug = 0

        uid = uuid.uuid4()
        return Commit(uid, hidden_bugs=hidden_bug, bugs=number_of_bugs, fix_time=fix_time)


class BuildPipelineReport:
    def __init__(self, name, avg_build_time, avg_resource_time):
        self.avg_resource_time = avg_resource_time
        self.avg_build_time = avg_build_time
        self.name = name


class SerialPipeline:
    def __init__(self, unit_test_time, integration_test_time, unit_test_flakiness, integration_test_flakiness,
                 deploy_time, idle_time_avg, sync_branches_time, commits):
        self.unit_test_time = unit_test_time
        self.integration_test_time = integration_test_time
        self.unit_test_flakiness = unit_test_flakiness
        self.integration_test_flakiness = integration_test_flakiness
        self.deploy_time = deploy_time
        self.commits = commits
        self.sync_branches_time = sync_branches_time
        self.idle_time = lambda: np.random.normal(idle_time_avg,
                                                  idle_time_avg / 2)  # time where devs are not available. however still in queue
        self.total_time = 0
        self.total_resource_time = 0
        self.avg_time_per_commit = None
        self.avg_resource_time_per_commit = None
        self.run_history = []

    def name(self):
        return "SerialPipeline"

    def execute_single_commit(self, commit):
        build = Build(commit=commit, unit_test_duration=self.unit_test_time,
                      integration_test_duration=self.integration_test_time,
                      unit_test_flakiness=self.unit_test_flakiness,
                      integration_test_flakiness=self.integration_test_flakiness, deploy_duration=self.deploy_time)
        # before starting, the developer might be idle
        build.idle(self.idle_time())
        # after idling, the developer needs to sync branches with master
        build.sync_branches(self.sync_branches_time)
        # after syncing with master, run unit tests
        unit_test_success = build.unit_test()
        while not unit_test_success:
            build.fix()  # developer will fix the problems themselves
            unit_test_success = build.unit_test()
        # once successful, merge with master and run integration tests
        integration_test_success = build.integration_test()
        while not integration_test_success:
            build.fix(1 + 1)  # dev + QA will troubleshoot integration test together
            integration_test_success = build.integration_test()
        build.idle(self.idle_time())  # dev probably idle while waiting for the integration test to pass
        build.deploy()
        return build

    def run(self):
        if self.run_history:
            return self.run_history
        for commit in self.commits:
            build = self.execute_single_commit(commit)
            self.run_history.append(build)
        return self.run_history

    def generate_report(self):
        builds_history = self.run()
        total_time = 0
        total_resource_time = 0
        for build in builds_history:
            for history in build.get_history():
                total_time += history.duration
                if history.action in [BuildHistory.FIXED, BuildHistory.DEPLOYED, BuildHistory.SYNCED_BRANCHES]:
                    total_resource_time += history.duration * history.resource
        return BuildPipelineReport(self.name(), total_time / len(builds_history),
                                   total_resource_time / len(builds_history))


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
            self.total_resource_time += self.merge_time * len(
                good_commits)  # total resource time is just spent to keep dev branch up to date with master

        self.avg_resource_time_per_commit = self.total_resource_time / total_processed
        self.avg_time_per_commit = self.total_time / total_processed


def main():
    commits = []
    g = CommitGenerator(bug_chance=0.1, bug_chance_per_line_of_code=1 / 200, avg_lines_of_code=500,
                        fix_time_per_lines_of_code=5 / 200, test_coverage=0.8)
    for _ in range(10000):
        commits.append(g.get())

    deploy_time = 10
    unit_test_flakiness = 0.1
    integration_test_flakiness = 0.2
    integration_test_time = 20
    unit_test_time = 10

    idle_time = 5
    merge_time = 5
    train_size = 5
    queue_size = 5
    merge_back_time_per_commit = 5
    debugging_complexity_coef_per_commit = 0.5

    ps = SerialPipeline(unit_test_flakiness=unit_test_flakiness,
                        integration_test_flakiness=integration_test_flakiness,
                        integration_test_time=integration_test_time, unit_test_time=unit_test_time,
                        deploy_time=deploy_time, idle_time_avg=idle_time, sync_branches_time=merge_time,
                        commits=commits)
    serial_pipline_report = ps.generate_report()

    pipeline_reports = [serial_pipline_report]
    for p in pipeline_reports:
        print(p.name)
        print(f"avg time per commit  {p.avg_build_time}")
        print(f"avg resource time per commit  {p.avg_resource_time}")
        print("------------")
        # prt1 = ReleaseTrainPipelineTestPerMerge(test_time, build_and_deploy_time, idle_time, merge_time, commits,
        #                                         train_size, merge_back_time_per_commit)
        # prt1.generate_report()
        # prt2 = ReleaseTrainPipelineTestOnce(test_time, build_and_deploy_time, idle_time, merge_time, commits,
        #                                     train_size,
        #                                     merge_back_time_per_commit, debugging_complexity_coef_per_commit)
        # prt2.generate_report()
        # qp = QueuePipeline(test_time, build_and_deploy_time, commits, queue_size, merge_time)
        # qp.generate_report()
        # pipeline_reports = [ps, prt1, prt2, qp]
        # for p in pipeline_reports:
        #     print(p.name())
        #     print(f"total time  {p.total_time}")
        #     print(f"total resource time  {p.total_resource_time}")
        #     print(f"avg time per commit  {p.avg_time_per_commit}")
        #     print(f"avg resource time per commit  {p.avg_resource_time_per_commit}")
        #     print("------------")


if __name__ == '__main__':
    main()
