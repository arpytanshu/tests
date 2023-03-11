# tests
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
