#tests
git-tests

```

show manuals:
	git help <verb>
	git <verb> --help
	man git-<verb>
show short manual:
	git <verb> -h

```

diff:
```
	git diff : 			shows changes that are unstaged.
	git diff --staged : shows changes that are staged. [--cached and --staged are synonyms]
	git diff --cached :	shows changes that are staged. [--cached and --staged are synonyms]
```

commit:
```
	git commit -v : puts the diff of changes in commit message editor.
	git commit -m : type inline commit messages.
	git commit -a : automatically stage all files that are already tracked before doing the commit, 
					skipping the `git add` part.
					if a new file is added, it is not staged/indexed.
```

rm:
```
	git rm :	remove targetted file from working-dir, and stage this removal.
				if the file was removed manually, this change will require `git rm` to be staged.
	git rm --cached : 	keep file in working-dir, but remove it from staging area.
```

log:
```
	git log :	lists the commits made in that repository in reverse chronological order
	git log -p : shows the difference (the patch output) introduced in each commit.
	git log -2 :	limit number of log entries to show.


Common options to git log:
-p  			Show the patch introduced with each commit.
--stat			Show statistics for files modified in each commit.
--shortstat		Display only the changed/insertions/deletions line from the --stat command.
--name-only		Show the list of files modified after the commit information.
--name-status	Show the list of files affected with added/modified/deleted information as well.
--abbrev-commit	Show only the first few characters of the SHA-1 checksum instead of all 40.
--relative-date	Display the date in a relative format (for example, “2 weeks ago”) instead of using the full date format.
--graph			Display an ASCII graph of the branch and merge history beside the log output.
--pretty		Show commits in an alternate format. 
				Option values include oneline, short, full, fuller, and format (where you specify your own format).
--oneline		Shorthand for --pretty=oneline --abbrev-commit used together.

```
- see more here: https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History


amend:
```
git commit --amend :	if no changes since last commit, and want to add more files to last commit
						You end up with a single commit — the second commit replaces the results of the first.
```

revert:
```
	git revert <comit-hash> : 	create a new commit that reverts the changes of the
								commit being targetted.
```

