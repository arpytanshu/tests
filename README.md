#tests
git-tests


show manuals:
	git help <verb>
	git <verb> --help
	man git-<verb>
show short manual:
	git <verb> -h


diff:
	git diff : 		shows changes that are unstaged.
	git diff --staged : 	shows changes that are staged. [--cached and --staged are synonyms]
	git diff --cached :	shows changes that are staged. [--cached and --staged are synonyms]

commit:
	git commit -v : puts the diff of changes in commit message editor.
	git commit -m : type inline commit messages.
	git commit -a : automatically stage all files that are already tracked before doing the commit, 
			skipping the `git add` part.
			if a new file is added, it is not staged/indexed.


rm:
	git rm :	remove targetted file from working-dir, and stage this removal.
			if the file was removed manually, this change will require `git rm` to be staged.
	git rm --cached : 	keep file in working-dir, but remove it from staging area.


log:
	git log : 	lists the commits made in that repository in reverse chronological order
	git log -p : 	shows the difference (the patch output) introduced in each commit.

revert:
	git revert <comit-hash> : 	create a new commit that reverts the changes of the commit being targetted.

