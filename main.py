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

    def duplicate(self):
        return Commit(self.commit_id, self.number_of_bugs, self.hidden_bugs, self.fix_time)

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


def duplicate_commits(commits):
    dup = []
    for commit in commits:
        dup.append(commit.duplicate())
    return dup


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
    def __init__(self, test_coverage, avg_lines_of_code, bug_chance_per_line_of_code,
                 fix_time_per_lines_of_code):
        self.test_coverage = test_coverage
        self.avg_lines_of_code = avg_lines_of_code
        self.bug_chance_per_line_of_code = bug_chance_per_line_of_code
        self.fix_time_per_lines_of_code = fix_time_per_lines_of_code

    def get(self):
        lines_of_code = np.random.normal(self.avg_lines_of_code, self.avg_lines_of_code / 2)
        number_of_bugs = lines_of_code * self.bug_chance_per_line_of_code
        hidden_bug = number_of_bugs * (1 - self.test_coverage)
        fix_time = max(self.fix_time_per_lines_of_code * lines_of_code, 1)  # takes at least 1 min to fix

        uid = uuid.uuid4()
        return Commit(uid, hidden_bugs=hidden_bug, bugs=number_of_bugs, fix_time=fix_time)


class BuildPipelineReport:
    def __init__(self, name, avg_build_time, avg_resource_time, batch_time=0):
        self.avg_resource_time = avg_resource_time
        self.avg_build_time = avg_build_time
        self.name = name
        self.batch_time = batch_time
        if not batch_time:
            self.batch_time = avg_build_time


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


# ignore this model for now
# class ReleaseTrainPipelineTestPerMerge:
#     def __init__(self, test_time, deploy_time, idle_time_avg, merge_time, commits, train_size,
#                  merge_back_time_per_commit):
#         self.test_time = test_time
#         self.deploy_time = deploy_time
#         self.commits = commits
#         self.merge_time = merge_time
#         self.train_size = train_size
#         self.idle_time = np.random.normal(idle_time_avg,
#                                           idle_time_avg / 2)  # time where devs are not available. however still in queue
#         self.total_time = 0
#         self.total_resource_time = 0
#         self.merge_back_time_per_commit = merge_back_time_per_commit
#         self.avg_time_per_commit = None
#         self.avg_resource_time_per_commit = None
#
#     def name(self):
#         return "ReleaseTrainPipelineTestPerMerge"
#
#     def generate_report(self):
#         for idx, commit in enumerate(self.commits):
#             if isinstance(commit, FaultyCommit):
#                 self.total_time += (
#                         commit.get_time_to_fix() + self.test_time + self.idle_time + self.merge_time)  # if faulty commit, total time it takes is the fix + test + deploy
#                 self.total_resource_time += (
#                         commit.get_time_to_fix() * 2 + self.merge_time)  # multiply the time to fix with QA time since they will be involved as well
#             else:
#                 self.total_time += (self.test_time + self.idle_time + self.merge_time)
#                 self.total_resource_time += self.merge_time
#
#             # every train deployment, need to do the merge back and deploy
#             if idx % self.train_size == 0:
#                 self.total_resource_time += self.merge_back_time_per_commit * self.train_size
#                 self.total_resource_time += self.deploy_time
#
#                 self.total_time += self.merge_back_time_per_commit * self.train_size
#                 self.total_time += self.deploy_time
#
#         self.avg_resource_time_per_commit = self.total_resource_time / len(self.commits)
#         self.avg_time_per_commit = self.total_time / len(self.commits)


class ReleaseTrainPipelineTestOnce:
    def __init__(self,
                 unit_test_time,
                 integration_test_time,
                 unit_test_flakiness,
                 integration_test_flakiness,
                 deploy_time,
                 idle_time_avg,
                 sync_branches_time,
                 commits, train_size,
                 trouble_shooting_complexity_coef,
                 automated=False,
                 fail_fast=False,
                 ):
        self.sync_branches_time = sync_branches_time if not automated else 0
        self.integration_test_flakiness = integration_test_flakiness
        self.unit_test_flakiness = unit_test_flakiness
        self.integration_test_time = integration_test_time
        self.unit_test_time = unit_test_time
        self.deploy_time = deploy_time
        self.fail_fast = fail_fast
        self.commits = commits
        self.train_size = train_size
        self.trouble_shooting_coef = trouble_shooting_complexity_coef
        self.idle_time = lambda: np.random.normal(idle_time_avg,
                                                  idle_time_avg / 2) if automated else 0  # time where devs are not available. however still in queue
        self.run_history = []

    def name(self):
        return "ReleaseTrainPipelineTestOnce"

    def execute_single_batch(self, commits, combined_complexity_coef):
        # for a single batch, the devs would merge all to the train branch and hopefully nothing breaks
        max_fix_time = 0
        total_bugs = 0
        total_hidden_bugs = 0
        # combine commits into a single merge commit
        for commit in commits:
            total_bugs += commit.number_of_bugs
            total_hidden_bugs += commit.hidden_bugs
            max_fix_time = max(commit.fix_time, max_fix_time)

        merged_commit = Commit(uuid.uuid4(), hidden_bugs=total_hidden_bugs,
                               fix_time=max_fix_time * combined_complexity_coef, bugs=total_bugs)

        # create a new representation of the build for the merged commit
        build = Build(commit=merged_commit, unit_test_duration=self.unit_test_time,
                      integration_test_duration=self.integration_test_time,
                      unit_test_flakiness=self.unit_test_flakiness,
                      integration_test_flakiness=self.integration_test_flakiness, deploy_duration=self.deploy_time)

        if build.commit.has_bug() and self.fail_fast:
            return build

        for _ in commits:
            # before starting, the developer might be idle
            build.idle(self.idle_time())
            # after idling, the developer needs to sync branches with each other
            build.sync_branches(self.sync_branches_time)

        # merged commits together at this point

        # run unit tests
        unit_test_success = build.unit_test()
        if self.fail_fast and not unit_test_success:
            return build
        while not unit_test_success:
            build.fix()  # developer will fix the problems themselves
            unit_test_success = build.unit_test()
        # once successful, merge with master and run integration tests
        integration_test_success = build.integration_test()
        if self.fail_fast and not unit_test_success:
            return build
        while not integration_test_success:
            build.fix(1 + len(commits))  # all the devs on the train + QA will troubleshoot integration test together
            integration_test_success = build.integration_test()
            build.idle(self.idle_time())  # dev probably idle while waiting for the integration test to pass
        build.deploy()
        return build

    def run(self):
        if self.run_history:
            return self.run_history
        trains = np.reshape(self.commits, (-1, self.train_size))
        for train in trains:
            build = self.execute_single_batch(train, self.trouble_shooting_coef)
            self.run_history.append(build)
        return self.run_history

    def generate_report(self):
        builds_history = self.run()
        total_time = 0
        total_resource_time = 0
        for batch in builds_history:
            for history in batch.get_history():
                total_time += history.duration
                if history.action in [BuildHistory.FIXED, BuildHistory.DEPLOYED, BuildHistory.SYNCED_BRANCHES]:
                    total_resource_time += history.duration * history.resource
        avg_time_per_commit = total_time / len(builds_history) / self.train_size
        avg_resource_time_per_commit = total_resource_time / len(builds_history) / self.train_size
        avg_train_time = total_time / len(builds_history)
        return BuildPipelineReport(self.name(), avg_build_time=avg_time_per_commit, batch_time=avg_train_time,
                                   avg_resource_time=avg_resource_time_per_commit)


class QueuePipeline:
    def __init__(self,
                 queue_size,
                 unit_test_time,
                 integration_test_time,
                 unit_test_flakiness,
                 integration_test_flakiness,
                 deploy_time,
                 commits,
                 ):
        self.integration_test_flakiness = integration_test_flakiness
        self.unit_test_flakiness = unit_test_flakiness
        self.integration_test_time = integration_test_time
        self.unit_test_time = unit_test_time
        self.commits = commits
        self.deploy_time = deploy_time
        self.run_history = []
        self.queue_size = queue_size
        # for the queue pipeline, it will parallelize the build in such a way that it goes as far as the first failure, then rejects the failure and everything after, and takes the successful one

    def name(self):
        return "QueuePipeline"

    def execute_single_batch(self, commits):
        # emulate the batch build for 1 to n number of commmits
        # pick the earliest one that hasn't errored
        # the ones that did error, fix them until they are all fixed. but note the fixes does not count for queue time!!
        parallel_builds = []
        for idx, _ in enumerate(commits):
            batch = duplicate_commits(commits[:idx + 1])
            pipeline = ReleaseTrainPipelineTestOnce(
                unit_test_flakiness=self.unit_test_flakiness, train_size=len(batch),
                integration_test_time=self.integration_test_time,
                integration_test_flakiness=self.integration_test_flakiness,
                deploy_time=self.deploy_time, commits=batch, idle_time_avg=0,
                sync_branches_time=0, trouble_shooting_complexity_coef=0,
                unit_test_time=self.unit_test_time, automated=True, fail_fast=True)
            parallel_builds.append(pipeline)

        latest_success_build = None
        batch_success = 0
        for pipeline in parallel_builds:
            build = pipeline.execute_single_batch(pipeline.commits, 0)
            if build.completed:
                latest_success_build = build
                batch_success = len(pipeline.commits)
            else:
                break
        return latest_success_build, batch_success

    def run(self):
        total_success = 0
        if self.run_history:
            return self.run_history
        trains = np.reshape(self.commits, (-1, self.queue_size))
        for train in trains:
            build, successes = self.execute_single_batch(train)
            total_success += successes
            self.run_history.append(build)
        return self.run_history, total_success

    def generate_report(self):
        builds_history, successes = self.run()
        total_time = 0
        total_resource_time = 0
        total_batches = 0
        for batch in builds_history:
            if not batch:
                continue
            total_batches += 1
            for history in batch.get_history():
                total_time += history.duration
                if history.action in [BuildHistory.FIXED]:
                    total_resource_time += history.duration * history.resource
        avg_time_per_commit = total_time / successes
        avg_resource_time_per_commit = 0
        avg_train_time = total_time / total_batches
        return BuildPipelineReport(self.name(), avg_build_time=avg_time_per_commit, batch_time=avg_train_time,
                                   avg_resource_time=avg_resource_time_per_commit)
        # divide up the commits into queues


def generate_base_line_report():
    commits = []
    g = CommitGenerator(bug_chance_per_line_of_code=1 / 200, avg_lines_of_code=300,
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

    ps = SerialPipeline(unit_test_flakiness=unit_test_flakiness,
                        integration_test_flakiness=integration_test_flakiness,
                        integration_test_time=integration_test_time, unit_test_time=unit_test_time,
                        deploy_time=deploy_time, idle_time_avg=idle_time, sync_branches_time=merge_time,
                        commits=commits)
    return ps.generate_report()


def main():
    base_report = generate_base_line_report()
    commits = []
    g = CommitGenerator(bug_chance_per_line_of_code=1 / 200, avg_lines_of_code=300,
                        fix_time_per_lines_of_code=5 / 200, test_coverage=0.8)
    for _ in range(10000):
        commits.append(g.get())

    commits_for_train = duplicate_commits(commits)
    commits_for_queue = duplicate_commits(commits)

    deploy_time = 10
    unit_test_flakiness = 0.1
    integration_test_flakiness = 0.2
    integration_test_time = 20
    unit_test_time = 10

    idle_time = 5
    merge_time = 5
    train_size = 5
    queue_size = 5

    # the amount of time it takes to fix an individual bug when a train of commits are merged together (i.e complexity added when problem solving not only your ticket)
    debugging_complexity_coef = 1

    ps = SerialPipeline(unit_test_flakiness=unit_test_flakiness,
                        integration_test_flakiness=integration_test_flakiness,
                        integration_test_time=integration_test_time, unit_test_time=unit_test_time,
                        deploy_time=deploy_time, idle_time_avg=idle_time, sync_branches_time=merge_time,
                        commits=commits)
    serial_pipline_report = ps.generate_report()

    rt = ReleaseTrainPipelineTestOnce(
        train_size=train_size,
        unit_test_flakiness=unit_test_flakiness,
        integration_test_flakiness=integration_test_flakiness,
        integration_test_time=integration_test_time, unit_test_time=unit_test_time,
        deploy_time=deploy_time, idle_time_avg=idle_time, sync_branches_time=merge_time,
        commits=commits_for_train, trouble_shooting_complexity_coef=debugging_complexity_coef,
    )
    release_train_report = rt.generate_report()

    rq = QueuePipeline(
        queue_size=queue_size,
        unit_test_flakiness=unit_test_flakiness,
        integration_test_flakiness=integration_test_flakiness,
        integration_test_time=integration_test_time, unit_test_time=unit_test_time,
        deploy_time=deploy_time,
        commits=commits_for_queue,
    )

    queue_report = rq.generate_report()
    pipeline_reports = [serial_pipline_report, release_train_report, queue_report]

    # print base report
    print("base report")
    print(f"avg time per commit  {base_report.avg_build_time}")
    print(f"avg resource time per commit  {base_report.avg_resource_time}")
    print(f"avg time per batch  {base_report.batch_time}")
    print("------------")

    # print experiment reports
    for p in pipeline_reports:
        print(p.name)
        print(
            f"avg time per commit  {p.avg_build_time}: [ {round(p.avg_build_time / base_report.avg_build_time * 100, 2)}% compare to base ] ")
        print(
            f"avg resource time per commit  {p.avg_resource_time}: [ {round(p.avg_resource_time / base_report.avg_resource_time * 100, 2)}% compare to base ]")
        print(
            f"avg time per batch  {p.batch_time}:[ {round(p.batch_time / base_report.batch_time * 100, 2)}% compare to base ]")
        print("------------")


if __name__ == '__main__':
    main()
