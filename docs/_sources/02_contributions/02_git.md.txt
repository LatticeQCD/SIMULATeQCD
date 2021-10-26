# How to use git

## Contribute to the ParallelGPUCode using git

If you want to start a new project, create a new branch. This should be done like that
```shell
git checkout -b <name_of_your_new_branch>
```
If you have done some changes, you should add these changes to your local repository using this command
```shell
git add /path/to/changed/file
```
Do NOT call `git add .` and **do NOT include the build folder**! You should always keep track of what files you have changed 
and should be pushed...
After that you should commit these changes to your local repository. This is done like this
```shell
git commit -m "Some words about what has been changed"
```
**Never leave the message empty!** Using **many small commits with detailed messages is always better** than doing one large commit!
Finally you can push your changes to the git:
```
git push origin <name_of_your_new_branch>
```
Frequently pull new changes from the master into your local branch, in order to not fall behind: 
```shell
git pull origin master 
```
There should never be an "old version" of the code in use. This makes eventually merging your code into the master easier and safer.

## Pushing to the master

If you don't exactly know what you're doing then you should **not push your code the master**.

**One always has to check that all executables still compile and that all tests pass without errors!**

You can automatically check this by simply running: 

```shell
make everything
cd testing
bash ../scripts/TEST_run.bash
```

**This should complete without any errors or warnings!** 
You need access to a machine with 4 GPUs for 2-3 hours (make a job script!). 


 ````{admonition} Pushing to the master
 :class: toggle

Pushing to the master branch is done like this:
```shell
git pull origin master
git merge <name_of_your_branch>
git push origin master
```
 
In the case that there are merge conflicts (another user has modified a file that you have also changed in your branch), you need to resolve these after calling @git merge@, then stage and commit the resolved files and then push again. 
````

If at any point you are not sure what to do, do not hesitate to ask someone who knows how to use git correctly.
