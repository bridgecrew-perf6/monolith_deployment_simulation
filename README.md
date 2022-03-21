# monolith_deployment_simulation

CI/CD in monolith application is difficult. 
- test time goes up as size of monolith increases
- CI serially becomes a bottleneck for large teams working on the same application
- CI using feature branch is an alternative but merging commits together to deploy would cause sorts of issues
- what are some other possible solutions? how can we measure the success/effectiveness of our CI/CD strategies based on the context of company's heuristics 
  - success rate per commit in deployment
  - time spent from commit to be merged in master (unittests, integration tests, merging, resolving merge conflicts)
  - etc. 
